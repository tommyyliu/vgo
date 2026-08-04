// Drives the real BatchedEvaluatorPool with a fixed number of concurrent
// callers, so lane counts are compared through the production broker path
// rather than a hand-rolled loop. This answers whether a second session
// overlaps one lane's staging memcpy with another lane's GPU execution.
use std::{
    env,
    hint::black_box,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use vgo_core::{Color, Position, Stone};
use vgo_inference::{
    BatchedEvaluator, BatchedEvaluatorPool, BrokerConfig, OnnxBatchService, OnnxProvider,
    OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::Evaluator;

fn fixture_positions() -> Vec<Position> {
    let radius = 1.0 / 6.0;
    let coordinates = [radius, 0.5, 1.0 - radius];
    let stones = coordinates
        .into_iter()
        .flat_map(|y| coordinates.into_iter().map(move |x| (x, y)))
        .enumerate()
        .map(|(index, (x, y))| {
            Stone::new(x, y, if index % 2 == 0 { Color::Black } else { Color::White })
        })
        .collect::<Vec<_>>();
    (0..=stones.len())
        .map(|count| Position::new(radius, stones[..count].to_vec(), Color::Black))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let get = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let model = PathBuf::from(get("--model", ""));
    let resolution: usize = get("--resolution", "128").parse()?;
    let policy_resolution: usize = get("--policy-resolution", "128").parse()?;
    let batch: usize = get("--batch", "64").parse()?;
    let lanes: usize = get("--lanes", "1").parse()?;
    // Concurrent callers. Each submits `--group` positions and blocks for the
    // reply, mirroring one MCTS actor with a leaf batch.
    let callers: usize = get("--callers", "64").parse()?;
    let group: usize = get("--group", "4").parse()?;
    let seconds: f64 = get("--seconds", "8").parse()?;
    let delay_ms: u64 = get("--delay-ms", "1").parse()?;
    let cache = PathBuf::from(get("--cache-directory", "artifacts/onnx-cache"));

    let raster = RasterConfig::square_of(resolution, RasterKind::Compact);

    let mut built = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        let service = OnnxBatchService::load(&OnnxServiceConfig {
            model: model.clone(),
            raster,
            policy: Some(RasterConfig::square(policy_resolution)),
            maximum_batch: batch,
            provider: OnnxProvider::TensorRt,
            device_id: 0,
            fp16: true,
            cache_directory: cache.clone(),
        })?;
        built.push(BatchedEvaluator::spawn(
            BrokerConfig {
                maximum_delay: Duration::from_millis(delay_ms),
                queue_capacity: (callers * 4).max(batch * 2),
            },
            service,
        )?);
    }
    let pool = Arc::new(BatchedEvaluatorPool::new(built)?);

    let positions = fixture_positions();
    let evaluations = Arc::new(AtomicUsize::new(0));

    // Warm every lane before timing.
    for _ in 0..(20 * lanes) {
        black_box(pool.evaluate_batch(&positions[..group.min(positions.len())])?);
    }

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = Instant::now();
    std::thread::scope(|scope| {
        for caller in 0..callers {
            let pool = Arc::clone(&pool);
            let evaluations = Arc::clone(&evaluations);
            let stop = Arc::clone(&stop);
            let positions = positions.clone();
            scope.spawn(move || {
                let mut cursor = caller;
                while !stop.load(Ordering::Relaxed) {
                    let batch_positions = (0..group)
                        .map(|i| positions[(cursor + i) % positions.len()].clone())
                        .collect::<Vec<_>>();
                    cursor = cursor.wrapping_add(group);
                    match pool.evaluate_batch(&batch_positions) {
                        Ok(outputs) => {
                            evaluations.fetch_add(outputs.len(), Ordering::Relaxed);
                            black_box(outputs);
                        }
                        Err(_) => break,
                    }
                }
            });
        }
        while started.elapsed().as_secs_f64() < seconds {
            std::thread::sleep(Duration::from_millis(20));
        }
        stop.store(true, Ordering::Relaxed);
    });
    let elapsed = started.elapsed().as_secs_f64();

    let total = evaluations.load(Ordering::Relaxed);
    let metrics = pool.lane_metrics();
    let batches: u64 = metrics.iter().map(|m| m.batches).sum();
    let positions_done: u64 = metrics.iter().map(|m| m.positions).sum();
    let inference_ns: u64 = metrics.iter().map(|m| m.inference_nanoseconds).sum();
    let queue_ns: u64 = metrics.iter().map(|m| m.queue_nanoseconds).sum();

    println!(
        "lanes={lanes} callers={callers} group={group} batch={batch} delay_ms={delay_ms}"
    );
    println!("  positions/s   {:9.0}", total as f64 / elapsed);
    println!("  average batch {:9.1}", positions_done as f64 / batches.max(1) as f64);
    println!(
        "  broker infer  {:9.3} ms/batch",
        inference_ns as f64 / 1e6 / batches.max(1) as f64
    );
    println!(
        "  queue wait    {:9.3} ms/position",
        queue_ns as f64 / 1e6 / positions_done.max(1) as f64
    );
    Ok(())
}
