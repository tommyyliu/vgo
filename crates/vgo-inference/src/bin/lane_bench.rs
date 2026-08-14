// Drives the real BatchedEvaluatorPool with a fixed number of concurrent
// callers, so slot counts are compared through the production broker path
// rather than a hand-rolled loop. This answers whether a second session
// overlaps one slot's staging memcpy with another slot's GPU execution.
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
    BatchedEvaluatorPool, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
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
            Stone::new(
                x,
                y,
                if index % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            )
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
        built.push(service);
    }
    let pool = Arc::new(BatchedEvaluatorPool::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(delay_ms),
            queue_capacity: (callers * 4).max(batch * 2),
        },
        built,
    )?);

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
    let metrics = pool.metrics();
    let batches = metrics.batches.max(1);
    let positions_done = metrics.positions.max(1);

    println!("lanes={lanes} callers={callers} group={group} batch={batch} delay_ms={delay_ms}");
    println!("  positions/s   {:9.0}", total as f64 / elapsed);
    println!(
        "  average batch {:9.1}",
        metrics.positions as f64 / batches as f64
    );
    println!(
        "  dispatch       {:9} full / {} deadline / {} drain",
        metrics.full_batches, metrics.deadline_batches, metrics.drain_batches
    );
    println!(
        "  queue wait     {:9.3} ms/position ({:.3} channel + {:.3} broker)",
        metrics.queue_nanoseconds as f64 / 1e6 / positions_done as f64,
        metrics.channel_nanoseconds as f64 / 1e6 / positions_done as f64,
        metrics.broker_queue_nanoseconds as f64 / 1e6 / positions_done as f64,
    );
    println!(
        "  broker batch   {:9.3} ms collect + {:.4} ms submit",
        metrics.batch_collection_nanoseconds as f64 / 1e6 / batches as f64,
        metrics.batch_submission_nanoseconds as f64 / 1e6 / batches as f64,
    );
    println!(
        "  inference      {:9.3} ms/batch ({:.3} pack + {:.3} session + {:.3} output + {:.3} other)",
        metrics.inference_nanoseconds as f64 / 1e6 / batches as f64,
        metrics.input_packing_nanoseconds as f64 / 1e6 / batches as f64,
        metrics.session_run_nanoseconds as f64 / 1e6 / batches as f64,
        metrics.output_materialization_nanoseconds as f64 / 1e6 / batches as f64,
        metrics.inference_unattributed_nanoseconds() as f64 / 1e6 / batches as f64,
    );
    println!(
        "  broker waits   {:9.3}s idle-input / {:.3}s overlap-input / {:.3}s completion",
        metrics.idle_request_wait_nanoseconds as f64 / 1e9,
        metrics.overlap_request_wait_nanoseconds as f64 / 1e9,
        metrics.completion_wait_nanoseconds as f64 / 1e9,
    );
    Ok(())
}
