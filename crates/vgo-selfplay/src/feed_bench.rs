#![forbid(unsafe_code)]

//! Multithreaded CPU capacity benchmark for feeding inference.
//!
//! Positions are loaded from a replay-v5 shard before timing. The benchmark
//! reports both the raw allocating rasterization ceiling and the complete host
//! handoff path through the production evaluator, broker, response channel, and
//! contiguous ONNX-style input gather. The packing service returns immediately;
//! no model runtime, device transfer, policy output, or MCTS work is included.

use std::{
    fs,
    hint::black_box,
    io,
    path::{Path, PathBuf},
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use vgo_core::{Color, Position, Stone};
use vgo_inference::{
    BatchContract, BatchService, BatchedEvaluator, BatchedEvaluatorPool, BrokerConfig,
    BrokerMetrics, InferenceInput, InferenceOutput,
};
use vgo_raster::{RasterConfig, RasterKind, rasterize};
use vgo_search::{EvaluationError, Evaluator};

const REPLAY_HEADER_BYTES: usize = 32;
const REPLAY_VERSION: u32 = 5;
const STONE_CAPACITY: usize = 128;
const STONE_BYTES: usize = 8 + 8 + 1;
const POLICY_CAPACITY: usize = 64;
const POLICY_ENTRY_BYTES: usize = 5 * 4;
const TRAILING_SCALAR_BYTES: usize = 4 + 4 + 8 + 4 + 8;
const REPLAY_V5_STRIDE: usize = 8
    + 8
    + 1
    + 4
    + 1
    + 4
    + STONE_CAPACITY * STONE_BYTES
    + 4
    + POLICY_CAPACITY * POLICY_ENTRY_BYTES
    + TRAILING_SCALAR_BYTES;

#[derive(Clone, Debug, Parser)]
#[command(about = "Measure multithreaded CPU capacity for feeding inference")]
struct Config {
    /// Replay-v5 dataset whose real positions form the untimed fixture corpus.
    #[arg(long)]
    dataset: PathBuf,
    /// Producer thread counts to benchmark.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,24,32,48,64")]
    threads: Vec<usize>,
    /// Timed samples per thread count; the median throughput is reported.
    #[arg(long, default_value_t = 3)]
    samples: usize,
    /// Target duration of each timed sample.
    #[arg(long, default_value_t = 1_500)]
    sample_millis: u64,
    /// Positions submitted together by one caller, matching MCTS leaf batching.
    #[arg(long, default_value_t = 4)]
    group: usize,
    /// Maximum positions gathered for one fake backend call.
    #[arg(long, default_value_t = 64)]
    batch: usize,
    /// Independent broker/packing lanes in the host-pipeline measurement.
    #[arg(long, default_value_t = 2)]
    lanes: usize,
    /// Broker collection deadline in milliseconds.
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct TimedCount {
    elapsed: Duration,
    positions: u64,
}

impl TimedCount {
    fn positions_per_second(self) -> f64 {
        self.positions as f64 / self.elapsed.as_secs_f64()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct HostSample {
    timed: TimedCount,
    broker: BrokerMetrics,
}

#[derive(Clone, Debug)]
struct ScalingResult {
    threads: usize,
    raster_positions_per_second: f64,
    host_positions_per_second: f64,
    host_average_batch: f64,
    host_encoding_milliseconds_per_position: f64,
    host_queue_milliseconds_per_position: f64,
    host_service_milliseconds_per_batch: f64,
}

struct PackingService {
    contract: BatchContract,
    states: Vec<f32>,
}

impl PackingService {
    fn new(raster: RasterConfig, maximum_batch: usize) -> Self {
        Self {
            contract: BatchContract {
                raster,
                // The response path still receives a valid dense policy, but
                // keeping it at one cell excludes production policy-transfer
                // cost from this input-feed benchmark.
                policy: RasterConfig::square(1),
                maximum_batch,
            },
            states: Vec::with_capacity(maximum_batch * raster.channels() * raster.pixels()),
        }
    }
}

impl BatchService for PackingService {
    fn contract(&self) -> BatchContract {
        self.contract
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        self.states.clear();
        for input in batch {
            self.states.extend_from_slice(input.raster().data());
        }
        black_box(self.states.as_slice());
        batch
            .iter()
            .map(|input| InferenceOutput::new(input.id(), 0.0, vec![0.0; 2]))
            .collect()
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated replay u32"))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("four-byte slice"),
    ))
}

fn read_f64(bytes: &[u8], offset: usize) -> io::Result<f64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated replay f64"))?;
    Ok(f64::from_le_bytes(
        value.try_into().expect("eight-byte slice"),
    ))
}

fn load_positions(path: &Path) -> io::Result<Vec<Position>> {
    let bytes = fs::read(path)?;
    if bytes.len() < REPLAY_HEADER_BYTES || &bytes[..8] != b"VGORPLY1" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected a VGO replay dataset",
        ));
    }
    let version = read_u32(&bytes, 8)?;
    if version != REPLAY_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected replay version {REPLAY_VERSION}, found {version}"),
        ));
    }
    let samples = read_u32(&bytes, 12)? as usize;
    if samples == 0 || !(bytes.len() - REPLAY_HEADER_BYTES).is_multiple_of(samples) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay length is not an integral number of records",
        ));
    }
    let stride = (bytes.len() - REPLAY_HEADER_BYTES) / samples;
    let stones_offset = 8 + 8 + 1 + 4 + 1 + 4;
    if stride != REPLAY_V5_STRIDE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected replay-v5 record stride {REPLAY_V5_STRIDE}, found {stride}"),
        ));
    }

    let mut positions = Vec::with_capacity(samples);
    for index in 0..samples {
        let base = REPLAY_HEADER_BYTES + index * stride;
        let radius = read_f64(&bytes, base)?;
        let komi = read_f64(&bytes, base + 8)?;
        let to_move = match bytes[base + 16] {
            0 => Color::Black,
            1 => Color::White,
            value => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid color code {value} in replay position {index}"),
                ));
            }
        };
        let count = read_u32(&bytes, base + 8 + 8 + 1 + 4 + 1)? as usize;
        if count > STONE_CAPACITY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("replay position {index} contains {count} stones"),
            ));
        }
        let mut stones = Vec::with_capacity(count);
        for slot in 0..count {
            let stone = base + stones_offset + slot * STONE_BYTES;
            let color = match bytes[stone + 16] {
                0 => Color::Black,
                1 => Color::White,
                value => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid stone color {value} in replay position {index}"),
                    ));
                }
            };
            stones.push(Stone::new(
                read_f64(&bytes, stone)?,
                read_f64(&bytes, stone + 8)?,
                color,
            ));
        }
        // Consecutive-pass and phase fields are deliberately not reconstructed:
        // Position exposes no public deserializer for them, and Compact's five
        // planes depend only on these geometry-visible fields. This loader must
        // not be reused for Semantic raster or MCTS benchmarks.
        positions.push(Position::new(radius, stones, to_move).with_komi(komi));
    }
    Ok(positions)
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn corpus_step(length: usize) -> usize {
    if length == 1 {
        return 1;
    }
    let mut step = (length / 3).max(1) | 1;
    while greatest_common_divisor(step, length) != 1 {
        step += 2;
        if step >= length {
            step = 1;
        }
    }
    step
}

fn run_raster_sample(
    positions: &[Position],
    raster: RasterConfig,
    threads: usize,
    duration: Duration,
) -> TimedCount {
    let ready = Arc::new(Barrier::new(threads + 1));
    let start = Arc::new(Barrier::new(threads + 1));
    let stop = Arc::new(AtomicBool::new(false));

    let (elapsed, positions_done) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        let step = corpus_step(positions.len());
        for worker in 0..threads {
            let ready = Arc::clone(&ready);
            let start = Arc::clone(&start);
            let stop = Arc::clone(&stop);
            handles.push(scope.spawn(move || {
                let mut cursor = worker * positions.len() / threads;
                let warm = rasterize(&positions[cursor], raster);
                black_box(warm.data());
                ready.wait();
                start.wait();

                let mut count = 0_u64;
                let mut checksum = 0.0_f64;
                while !stop.load(Ordering::Relaxed) {
                    let encoded = rasterize(black_box(&positions[cursor]), raster);
                    checksum += f64::from(encoded.data()[cursor % encoded.data().len()]);
                    black_box(&encoded);
                    count += 1;
                    cursor = (cursor + step) % positions.len();
                }
                black_box(checksum);
                count
            }));
        }
        ready.wait();
        let started = Instant::now();
        start.wait();
        thread::sleep(duration);
        stop.store(true, Ordering::Relaxed);
        let count = handles
            .into_iter()
            .map(|handle| handle.join().expect("raster worker"))
            .sum();
        (started.elapsed(), count)
    });
    TimedCount {
        elapsed,
        positions: positions_done,
    }
}

fn build_pool(
    lanes: usize,
    callers: usize,
    raster: RasterConfig,
    maximum_batch: usize,
    maximum_delay: Duration,
) -> BatchedEvaluatorPool {
    let lanes = (0..lanes)
        .map(|_| {
            BatchedEvaluator::spawn(
                BrokerConfig {
                    maximum_delay,
                    queue_capacity: (callers * 4).max(maximum_batch * 2),
                },
                PackingService::new(raster, maximum_batch),
            )
            .expect("start packing broker")
        })
        .collect();
    BatchedEvaluatorPool::new(lanes).expect("compatible packing lanes")
}

fn metrics_delta(after: BrokerMetrics, before: BrokerMetrics) -> BrokerMetrics {
    BrokerMetrics {
        requests: after.requests.saturating_sub(before.requests),
        batches: after.batches.saturating_sub(before.batches),
        positions: after.positions.saturating_sub(before.positions),
        maximum_batch: after.maximum_batch,
        failures: after.failures.saturating_sub(before.failures),
        encoding_nanoseconds: after
            .encoding_nanoseconds
            .saturating_sub(before.encoding_nanoseconds),
        queue_nanoseconds: after
            .queue_nanoseconds
            .saturating_sub(before.queue_nanoseconds),
        inference_nanoseconds: after
            .inference_nanoseconds
            .saturating_sub(before.inference_nanoseconds),
    }
}

fn run_host_sample(
    groups: &[Vec<Position>],
    raster: RasterConfig,
    threads: usize,
    lanes: usize,
    maximum_batch: usize,
    maximum_delay: Duration,
    duration: Duration,
) -> HostSample {
    let pool = Arc::new(build_pool(
        lanes,
        threads,
        raster,
        maximum_batch,
        maximum_delay,
    ));
    let warm_positions = groups
        .iter()
        .flatten()
        .cycle()
        .take(maximum_batch)
        .cloned()
        .collect::<Vec<_>>();
    for _ in 0..lanes {
        black_box(
            pool.evaluate_batch(&warm_positions)
                .expect("warm packing lane at maximum batch"),
        );
    }
    let ready = Arc::new(Barrier::new(threads + 1));
    let start = Arc::new(Barrier::new(threads + 1));
    let stop = Arc::new(AtomicBool::new(false));

    let (elapsed, positions_done, before) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        let step = corpus_step(groups.len());
        for worker in 0..threads {
            let pool = Arc::clone(&pool);
            let ready = Arc::clone(&ready);
            let start = Arc::clone(&start);
            let stop = Arc::clone(&stop);
            handles.push(scope.spawn(move || {
                let mut cursor = worker * groups.len() / threads;
                black_box(
                    pool.evaluate_batch(&groups[cursor])
                        .expect("packing evaluator warmup"),
                );
                ready.wait();
                start.wait();

                let mut count = 0_u64;
                while !stop.load(Ordering::Relaxed) {
                    let evaluations = pool
                        .evaluate_batch(black_box(&groups[cursor]))
                        .expect("packing evaluator");
                    count += evaluations.len() as u64;
                    black_box(evaluations);
                    cursor = (cursor + step) % groups.len();
                }
                count
            }));
        }
        ready.wait();
        let before = pool.metrics();
        let started = Instant::now();
        start.wait();
        thread::sleep(duration);
        stop.store(true, Ordering::Relaxed);
        let count = handles
            .into_iter()
            .map(|handle| handle.join().expect("host-pipeline worker"))
            .sum();
        (started.elapsed(), count, before)
    });
    let broker = metrics_delta(pool.metrics(), before);
    assert_eq!(
        broker.positions, positions_done,
        "broker metrics must account for every completed timed position"
    );
    HostSample {
        timed: TimedCount {
            elapsed,
            positions: positions_done,
        },
        broker,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn summarize(
    threads: usize,
    raster_samples: &[TimedCount],
    host_samples: &[HostSample],
) -> ScalingResult {
    let raster_positions_per_second = median(
        raster_samples
            .iter()
            .map(|sample| sample.positions_per_second())
            .collect(),
    );
    let host_positions_per_second = median(
        host_samples
            .iter()
            .map(|sample| sample.timed.positions_per_second())
            .collect(),
    );
    let broker = host_samples
        .iter()
        .fold(BrokerMetrics::default(), |mut total, sample| {
            total.requests += sample.broker.requests;
            total.batches += sample.broker.batches;
            total.positions += sample.broker.positions;
            total.maximum_batch = total.maximum_batch.max(sample.broker.maximum_batch);
            total.failures += sample.broker.failures;
            total.encoding_nanoseconds += sample.broker.encoding_nanoseconds;
            total.queue_nanoseconds += sample.broker.queue_nanoseconds;
            total.inference_nanoseconds += sample.broker.inference_nanoseconds;
            total
        });
    let positions = broker.positions.max(1) as f64;
    let batches = broker.batches.max(1) as f64;
    ScalingResult {
        threads,
        raster_positions_per_second,
        host_positions_per_second,
        host_average_batch: broker.positions as f64 / batches,
        host_encoding_milliseconds_per_position: broker.encoding_nanoseconds as f64
            / 1.0e6
            / positions,
        host_queue_milliseconds_per_position: broker.queue_nanoseconds as f64 / 1.0e6 / positions,
        host_service_milliseconds_per_batch: broker.inference_nanoseconds as f64 / 1.0e6 / batches,
    }
}

fn stone_statistics(positions: &[Position]) -> (usize, f64, usize, usize) {
    let mut counts = positions
        .iter()
        .map(|position| position.stones().len())
        .collect::<Vec<_>>();
    counts.sort_unstable();
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    (
        counts[0],
        mean,
        counts[counts.len() / 2],
        counts[counts.len() - 1],
    )
}

fn print_json(config: &Config, positions: &[Position], results: &[ScalingResult]) {
    let (minimum, mean, median_stones, maximum) = stone_statistics(positions);
    println!("{{");
    println!("  \"schema\": \"vgo.cpu-feed-scaling.v1\",");
    println!("  \"dataset\": {:?},", config.dataset.display().to_string());
    println!("  \"positions\": {},", positions.len());
    println!(
        "  \"stones\": {{\"minimum\": {minimum}, \"mean\": {mean:.3}, \
         \"median\": {median_stones}, \"maximum\": {maximum}}},"
    );
    println!(
        "  \"raster\": {{\"kind\": \"compact\", \"channels\": 5, \"height\": 128, \"width\": 128}},"
    );
    println!(
        "  \"host_pipeline\": {{\"group\": {}, \"maximum_batch\": {}, \
         \"lanes\": {}, \"delay_ms\": {}}},",
        config.group, config.batch, config.lanes, config.delay_ms
    );
    println!(
        "  \"timing\": {{\"samples\": {}, \"sample_millis\": {}}},",
        config.samples, config.sample_millis
    );
    println!("  \"results\": [");
    for (index, result) in results.iter().enumerate() {
        let comma = if index + 1 == results.len() { "" } else { "," };
        println!(
            concat!(
                "    {{\"threads\": {}, \"raster_positions_per_second\": {:.3}, ",
                "\"host_positions_per_second\": {:.3}, \"host_average_batch\": {:.3}, ",
                "\"host_encoding_milliseconds_per_position\": {:.6}, ",
                "\"host_queue_milliseconds_per_position\": {:.6}, ",
                "\"host_service_milliseconds_per_batch\": {:.6}}}{}"
            ),
            result.threads,
            result.raster_positions_per_second,
            result.host_positions_per_second,
            result.host_average_batch,
            result.host_encoding_milliseconds_per_position,
            result.host_queue_milliseconds_per_position,
            result.host_service_milliseconds_per_batch,
            comma,
        );
    }
    println!("  ]");
    println!("}}");
}

fn print_table(config: &Config, positions: &[Position], results: &[ScalingResult]) {
    let (minimum, mean, median_stones, maximum) = stone_statistics(positions);
    println!(
        "fixture: {} real replay positions, stones min={minimum} mean={mean:.1} \
         median={median_stones} max={maximum}",
        positions.len()
    );
    println!(
        "host path: group={} batch={} lanes={} delay={}ms",
        config.group, config.batch, config.lanes, config.delay_ms
    );
    println!(
        "{:>7} {:>14} {:>14} {:>10} {:>11} {:>10} {:>11}",
        "threads", "raster pos/s", "host pos/s", "avg batch", "encode ms", "queue ms", "service ms"
    );
    for result in results {
        println!(
            "{:>7} {:>14.0} {:>14.0} {:>10.1} {:>11.3} {:>10.3} {:>11.3}",
            result.threads,
            result.raster_positions_per_second,
            result.host_positions_per_second,
            result.host_average_batch,
            result.host_encoding_milliseconds_per_position,
            result.host_queue_milliseconds_per_position,
            result.host_service_milliseconds_per_batch,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::parse();
    if config.samples == 0
        || config.sample_millis == 0
        || config.group == 0
        || config.batch == 0
        || config.lanes == 0
        || config.threads.contains(&0)
    {
        return Err("benchmark counts, batch sizes, lanes, and threads must be positive".into());
    }
    if config.group > config.batch {
        return Err("--group must not exceed --batch".into());
    }
    config.threads.sort_unstable();
    config.threads.dedup();

    let positions = load_positions(&config.dataset)?;
    if positions.len() < config.group {
        return Err("dataset does not contain one complete request group".into());
    }
    let groups = positions
        .chunks_exact(config.group)
        .map(<[Position]>::to_vec)
        .collect::<Vec<_>>();
    let raster = RasterConfig::square_of(128, RasterKind::Compact);
    let duration = Duration::from_millis(config.sample_millis);
    let maximum_delay = Duration::from_millis(config.delay_ms);
    let mut results = Vec::with_capacity(config.threads.len());

    for &threads in &config.threads {
        let mut raster_samples = Vec::with_capacity(config.samples);
        let mut host_samples = Vec::with_capacity(config.samples);
        for _ in 0..config.samples {
            raster_samples.push(run_raster_sample(&positions, raster, threads, duration));
            host_samples.push(run_host_sample(
                &groups,
                raster,
                threads,
                config.lanes,
                config.batch,
                maximum_delay,
                duration,
            ));
        }
        results.push(summarize(threads, &raster_samples, &host_samples));
    }

    if config.json {
        print_json(&config, &positions, &results);
    } else {
        print_table(&config, &positions, &results);
    }
    Ok(())
}
