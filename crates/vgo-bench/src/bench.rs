//! Time each large CPU path on a fixed set of real positions.
//!
//! Every path the generator spends CPU on, measured alone and single-threaded
//! so numbers from different commits are comparable: applying a move, the
//! board analysis, the legal-set vertex index, the settled plane, the packed
//! raster at both sizes the loop has used, the search's fine grid, and a whole
//! search with a fixed policy standing in for the network. These default paths
//! leave inference out. With `--model`, run concurrent neural searches instead,
//! including encoding, broker scheduling and actual model inference.
//!
//! Positions come from `performance/positions.txt`, 16 at radius 1/38 and 16 at
//! 1/18, and each path is reported per board size because cost scales with
//! stone count. `performance/run.py` runs this, keeps the history, and compares.
//!
//!     vgo-bench [--positions PATH] [--seconds S] [--only NAME]
//!
//! Built with `--features profiling`, `--profile-output PATH` also writes the
//! sampled stacks as folded text, one `frame;frame;frame count` per line.

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use vgo_bench::{COARSE_POOL, FixedEvaluator, POLICY_RESOLUTION, SEARCH_SIMULATIONS, load};
use vgo_core::{Analysis, LegalSetIndex, Point, Position, nearest_legal_placement, place};
use vgo_inference::{
    BatchedEvaluatorPool, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{
    DensePolicy, RasterConfig, RasterKind, packed::rasterize_packed, settled_for_raster,
};
use vgo_search::{
    Evaluation, EvaluationError, Evaluator, Policy, SearchConfig, search_with_evaluator,
};

#[derive(Parser)]
struct Arguments {
    /// Write sampled CPU stacks as folded text. Requires the `profiling` feature.
    #[arg(long)]
    profile_output: Option<PathBuf>,
    /// Fixture positions, one per line.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../performance/positions.txt"))]
    positions: PathBuf,
    /// Time budget per path and board size, after one warm-up pass.
    #[arg(long, default_value_t = 2.0)]
    seconds: f64,
    /// Run only the paths whose name contains this.
    #[arg(long)]
    only: Option<String>,
    /// Run complete neural searches on the saved positions instead of CPU-only paths.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    resolution: usize,
    #[arg(long, default_value_t = 32)]
    actors: usize,
    #[arg(long, default_value_t = 1000)]
    simulations: u32,
    #[arg(long, default_value_t = 32)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 2)]
    inference_slots: usize,
    #[arg(long, default_value_t = 0.1)]
    root_exploration_noise: f64,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

/// `FixedEvaluator` plus the encoding the network path pays for each leaf, so a
/// search costs what it costs the CPU in generation, with only inference absent.
struct RasterizingEvaluator {
    inner: FixedEvaluator,
    raster: RasterConfig,
}

impl Evaluator for RasterizingEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        black_box(rasterize_packed(position, self.raster));
        self.inner.evaluate(position)
    }
}

#[cfg(feature = "profiling")]
fn start_profiler(output: Option<&std::path::Path>) -> Option<pprof::ProfilerGuard<'static>> {
    output.map(|_| {
        pprof::ProfilerGuardBuilder::default()
            .frequency(499)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .expect("start profiler")
    })
}

#[cfg(feature = "profiling")]
fn write_profile(guard: Option<pprof::ProfilerGuard<'static>>, output: Option<&std::path::Path>) {
    use std::io::Write;
    let (Some(guard), Some(path)) = (guard, output) else {
        return;
    };
    let report = guard.report().build().expect("build profile");
    let mut file = std::fs::File::create(path).expect("create profile");
    for (frames, count) in &report.data {
        let mut names: Vec<String> = frames
            .frames
            .iter()
            .rev()
            .flat_map(|inlined| inlined.iter().rev().map(|symbol| symbol.name()))
            .collect();
        names.iter_mut().for_each(|name| *name = name.replace(';', ":"));
        writeln!(file, "{} {count}", names.join(";")).expect("write profile");
    }
}

struct Timing {
    per_op_us: f64,
    spread: f64,
    passes: usize,
}

/// Median time per operation over whole passes of `work`, which runs `ops`
/// operations. Whole passes rather than single calls, so a pass always covers
/// every stage of the game and the median is not dominated by the cheap ones.
fn time(ops: usize, budget: Duration, mut work: impl FnMut()) -> Timing {
    work();
    let mut passes = Vec::new();
    let started = Instant::now();
    while passes.len() < 3 || (started.elapsed() < budget && passes.len() < 50) {
        let pass = Instant::now();
        work();
        passes.push(pass.elapsed().as_secs_f64());
    }
    passes.sort_by(f64::total_cmp);
    let median = passes[passes.len() / 2];
    let low = passes[passes.len() / 10];
    let high = passes[(passes.len() * 9) / 10];
    Timing {
        per_op_us: median * 1e6 / ops as f64,
        spread: (high - low) / median,
        passes: passes.len(),
    }
}

fn main() {
    let arguments = Arguments::parse();
    #[cfg(not(feature = "profiling"))]
    assert!(
        arguments.profile_output.is_none(),
        "--profile-output requires the `profiling` feature"
    );
    #[cfg(feature = "profiling")]
    let profiler = start_profiler(arguments.profile_output.as_deref());
    run_benchmarks(&arguments);
    #[cfg(feature = "profiling")]
    write_profile(profiler, arguments.profile_output.as_deref());
}

fn run_benchmarks(arguments: &Arguments) {
    let positions = load(&arguments.positions);
    if arguments.model.is_some() {
        neural_benchmark(arguments, positions);
        return;
    }
    let budget = Duration::from_secs_f64(arguments.seconds);
    let evaluator = FixedEvaluator::new();
    let policy_grid = RasterConfig::square(POLICY_RESOLUTION);
    let mut search = SearchConfig::canary(SEARCH_SIMULATIONS);
    search.coarse_pool = COARSE_POOL;
    search.widening_coefficient = 6.0;
    search.maximum_candidates = 321;
    search.leaf_batch = 4;

    let mut groups: Vec<(String, Vec<Position>)> = Vec::new();
    for position in positions {
        let board = format!("r{}", (1.0 / position.radius()).round() as u32);
        match groups.iter_mut().find(|(name, _)| *name == board) {
            Some((_, list)) => list.push(position),
            None => groups.push((board, vec![position])),
        }
    }

    let mut results = Vec::new();
    for (board, positions) in &groups {
        // One legal move per position, chosen outside the timed region.
        let moves: Vec<Point> = positions
            .iter()
            .enumerate()
            .map(|(index, position)| {
                let target = Point::new(
                    0.13 + 0.57 * ((index * 7) % 11) as f64 / 11.0,
                    0.21 + 0.6 * ((index * 5) % 13) as f64 / 13.0,
                );
                nearest_legal_placement(position, target).point
            })
            .collect();
        let raster = |size| RasterConfig::square_of(size, RasterKind::CompactRadius);
        let count = positions.len();
        let mut run = |name: &str, ops: usize, work: &mut dyn FnMut()| {
            if arguments
                .only
                .as_deref()
                .is_some_and(|only| !name.contains(only))
            {
                return;
            }
            let timing = time(ops, budget, work);
            eprintln!(
                "{board:>4} {name:<16} {:>10.1} us/op  (±{:.0}%, {} passes)",
                timing.per_op_us,
                50.0 * timing.spread,
                timing.passes
            );
            results.push(format!(
                "\"{board}.{name}\":{{\"us\":{:.2},\"spread\":{:.3},\"passes\":{}}}",
                timing.per_op_us, timing.spread, timing.passes
            ));
        };
        run("place", count, &mut || {
            for (position, point) in positions.iter().zip(&moves) {
                let _ = black_box(place(position, point.x, point.y));
            }
        });
        run("analysis", count, &mut || {
            for position in positions {
                let _ = black_box(Analysis::new(position));
            }
        });
        run("legal_index", count, &mut || {
            for position in positions {
                let _ = black_box(LegalSetIndex::build(position));
            }
        });
        for size in [128, 256] {
            run(&format!("settled_{size}"), count, &mut || {
                for position in positions {
                    black_box(settled_for_raster(position, raster(size)));
                }
            });
            run(&format!("raster_{size}"), count, &mut || {
                for position in positions {
                    black_box(rasterize_packed(position, raster(size)));
                }
            });
        }
        run("fine_grid", count, &mut || {
            let policy = DensePolicy::new(policy_grid, evaluator.logits.to_vec());
            for position in positions {
                black_box(policy.fine_grid(position, COARSE_POOL));
            }
        });
        run("search_200", count, &mut || {
            for (index, position) in positions.iter().enumerate() {
                let _ = black_box(search_with_evaluator(
                    position,
                    search,
                    index as u64,
                    &evaluator,
                ));
            }
        });
        for size in [128, 256] {
            let rasterizing = RasterizingEvaluator {
                inner: FixedEvaluator::new(),
                raster: raster(size),
            };
            run(&format!("search_raster_{size}"), count, &mut || {
                for (index, position) in positions.iter().enumerate() {
                    let _ = black_box(search_with_evaluator(
                        position,
                        search,
                        index as u64,
                        &rasterizing,
                    ));
                }
            });
        }
    }
    println!("{{{}}}", results.join(","));
}

fn neural_benchmark(arguments: &Arguments, positions: Vec<Position>) {
    assert!(arguments.actors > 0 && arguments.simulations > 0 && arguments.inference_slots > 0);
    // The same primary-context flag as continuous generation, before ORT loads.
    // SAFETY: initialization is idempotent, the device handle is checked, and
    // the primary-context flags are set before any native session is loaded.
    if !matches!(arguments.provider, OnnxProvider::Cpu) {
        unsafe {
            use cudarc::driver::sys;
            sys::cuInit(0).result().expect("CUDA initialization");
            let device = cudarc::driver::result::device::get(0).expect("CUDA device");
            sys::cuDevicePrimaryCtxSetFlags_v2(
                device,
                sys::CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as u32,
            )
            .result()
            .expect("CUDA blocking sync before session load");
        }
    }
    let mut services = Vec::new();
    for _ in 0..arguments.inference_slots {
        services.push(
            OnnxBatchService::load(&OnnxServiceConfig {
                model: arguments.model.clone().unwrap(),
                raster: RasterConfig::square_of(arguments.resolution, RasterKind::CompactRadius),
                policy: Some(RasterConfig::square(POLICY_RESOLUTION)),
                maximum_batch: arguments.maximum_batch,
                provider: arguments.provider,
                device_id: 0,
                fp16: true,
                cache_directory: arguments.cache_directory.clone(),
            })
            .expect("neural benchmark session"),
        );
    }
    let pool = Arc::new(
        BatchedEvaluatorPool::spawn(
            BrokerConfig {
                maximum_delay: Duration::from_millis(1),
                queue_capacity: arguments.actors * 8,
            },
            services,
        )
        .expect("neural benchmark broker"),
    );
    let mut config = SearchConfig::canary(arguments.simulations);
    config.coarse_pool = COARSE_POOL;
    config.widening_coefficient = 6.0;
    config.maximum_candidates = 321;
    config.leaf_batch = 4;
    config.root_exploration_noise = arguments.root_exploration_noise;
    let mut results = serde_json::Map::new();
    for units in [18, 38] {
        let boards: Vec<_> = positions
            .iter()
            .filter(|position| (1.0 / position.radius()).round() as u32 == units)
            .cloned()
            .collect();
        assert!(!boards.is_empty());
        let baseline = pool.metrics();
        let timing = time(
            arguments.actors,
            Duration::from_secs_f64(arguments.seconds),
            || {
                thread::scope(|scope| {
                    for actor in 0..arguments.actors {
                        let evaluator = Arc::clone(&pool);
                        let position = &boards[actor % boards.len()];
                        scope.spawn(move || {
                            black_box(
                                search_with_evaluator(
                                    position,
                                    config,
                                    actor as u64,
                                    evaluator.as_ref(),
                                )
                                .expect("neural search"),
                            );
                        });
                    }
                });
            },
        );
        let metrics = pool.metrics().delta_since(baseline);
        let name = format!(
            "r{units}.neural_search_{}x{}",
            arguments.actors, arguments.simulations
        );
        eprintln!(
            "{name}: {:.2} searches/s, {} timed passes",
            1e6 / timing.per_op_us,
            timing.passes
        );
        results.insert(
            name,
            serde_json::json!({
                "us": timing.per_op_us, "spread": timing.spread, "passes": timing.passes,
                "positions": metrics.positions, "failures": metrics.failures,
                "broker": vgo_selfplay::telemetry::broker_json(metrics, arguments.maximum_batch),
            }),
        );
    }
    println!("{}", serde_json::Value::Object(results));
}
