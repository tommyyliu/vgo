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
//! Positions come from `benchmarks/positions.txt`, 16 at radius 1/38 and 16 at
//! 1/18, and each path is reported per board size because cost scales with
//! stone count. `scripts/bench.py` runs this, keeps the history, and compares.
//!
//!     vgo-bench [--positions PATH] [--seconds S] [--only NAME]
//!
//! `--digest` instead hashes every raster a set of searches asks for, plus each
//! search's result. An optimization that claims to change nothing has to leave
//! this output unchanged: a model only ever sees these bytes.
//!
//! `--fuzz-games N` plays seeded games across board sizes and both rulesets and
//! hashes every ply's geometry, legal set, liveness, score and rasters. The
//! fixture digest above only sees two board sizes under one ruleset.
//!
//! Built with `--features profiling`, `--profile-output PATH` also writes the
//! sampled stacks as folded text, one `frame;frame;frame count` per line.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use vgo_core::{
    Analysis, Color, LegalSetIndex, Phase, Point, Position, Ruleset, Stone,
    nearest_legal_placement, pass, place,
};
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
    /// Print digests of rasters and search results, and run no timings.
    #[arg(long)]
    digest: bool,
    /// Print a digest of this many seeded whole games, one line each, and run
    /// no timings. Two builds that claim identical behaviour must print the same
    /// lines; the first differing line names the board and rules to look at.
    #[arg(long)]
    fuzz_games: Option<u64>,
    /// Write sampled CPU stacks as folded text. Requires the `profiling` feature.
    #[arg(long)]
    profile_output: Option<PathBuf>,
    /// Fixture positions, one per line.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benchmarks/positions.txt"))]
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

/// Policy grid and search settings the loop generates with.
const POLICY_RESOLUTION: usize = 128;
const COARSE_POOL: usize = 16;
const SEARCH_SIMULATIONS: u32 = 200;

fn load(path: &PathBuf) -> Vec<Position> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let number = |index: usize| -> f64 { fields[index].parse().expect("number") };
            let colour = |value: f64| {
                if value == 0.0 {
                    Color::Black
                } else {
                    Color::White
                }
            };
            let count = number(4) as usize;
            let stones = (0..count)
                .map(|stone| {
                    let base = 5 + 3 * stone;
                    Stone::new(number(base), number(base + 1), colour(number(base + 2)))
                })
                .collect();
            Position::new(number(0), stones, colour(number(2)))
                .with_komi(number(1))
                .with_passes(number(3) as u32)
        })
        .collect()
}

/// A smooth, fixed placement field, so search exercises the spatial proposal
/// path exactly as it does with the network without needing one.
struct FixedEvaluator {
    logits: Arc<Vec<f32>>,
}

impl FixedEvaluator {
    fn new() -> Self {
        let side = POLICY_RESOLUTION;
        let mut logits = Vec::with_capacity(side * side + 1);
        for row in 0..side {
            for column in 0..side {
                let (x, y) = (column as f32 / side as f32, row as f32 / side as f32);
                logits.push((7.0 * x).sin() * (5.0 * y).cos() + 0.5 * (13.0 * (x + y)).sin());
            }
        }
        logits.push(-2.0);
        Self {
            logits: Arc::new(logits),
        }
    }
}

impl Evaluator for FixedEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        // Varies with the position so backed-up values are not all equal.
        let value = ((position.stones().len() % 7) as f64 - 3.0) / 10.0;
        let policy = DensePolicy::new(
            RasterConfig::square(POLICY_RESOLUTION),
            self.logits.to_vec(),
        );
        Ok(Evaluation::new(value, Box::new(policy)))
    }
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

/// Hashes each leaf's raster at both loop sizes, in the order search asks for them.
struct DigestEvaluator {
    inner: FixedEvaluator,
    hasher: Mutex<DefaultHasher>,
    leaves: AtomicU64,
}

impl Evaluator for DigestEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        let mut hasher = self.hasher.lock().unwrap();
        for size in [128, 256] {
            let raster = rasterize_packed(
                position,
                RasterConfig::square_of(size, RasterKind::CompactRadius),
            );
            raster.bits().hash(&mut *hasher);
            for value in raster.dense().iter().chain(raster.scalars()) {
                value.to_bits().hash(&mut *hasher);
            }
        }
        drop(hasher);
        self.leaves.fetch_add(1, Ordering::Relaxed);
        self.inner.evaluate(position)
    }
}

fn digest(positions: &[Position]) {
    let mut search = SearchConfig::canary(SEARCH_SIMULATIONS);
    search.coarse_pool = COARSE_POOL;
    search.widening_coefficient = 6.0;
    search.maximum_candidates = 321;
    search.leaf_batch = 4;
    let evaluator = DigestEvaluator {
        inner: FixedEvaluator::new(),
        hasher: Mutex::new(DefaultHasher::new()),
        leaves: AtomicU64::new(0),
    };
    let mut results = DefaultHasher::new();
    for (index, position) in positions.iter().enumerate() {
        let result = search_with_evaluator(position, search, index as u64, &evaluator)
            .expect("digest search");
        format!("{result:?}").hash(&mut results);
    }
    println!(
        "positions={} leaves={} rasters={:016x} searches={:016x}",
        positions.len(),
        evaluator.leaves.load(Ordering::Relaxed),
        evaluator.hasher.lock().unwrap().finish(),
        results.finish(),
    );
}

fn hash_point(point: Point, hasher: &mut DefaultHasher) {
    point.x.to_bits().hash(hasher);
    point.y.to_bits().hash(hasher);
}

fn hash_analysis(analysis: &Analysis, hasher: &mut DefaultHasher) {
    for cell in analysis.geometry.cells.iter() {
        cell.area.to_bits().hash(hasher);
        cell.polygon.iter().for_each(|point| hash_point(*point, hasher));
        for edge in &cell.edges {
            hash_point(edge.start, hasher);
            hash_point(edge.end, hasher);
            format!("{:?}", edge.source).hash(hasher);
        }
    }
    analysis.geometry.adjacency.hash(hasher);
    analysis.geometry.groups.hash(hasher);
    analysis.legal_vertices.iter().for_each(|point| hash_point(*point, hasher));
    for groups in [&analysis.alive_groups, &analysis.settled_groups] {
        let mut sorted: Vec<usize> = groups.iter().copied().collect();
        sorted.sort_unstable();
        sorted.hash(hasher);
    }
    analysis.score.black.to_bits().hash(hasher);
    analysis.score.white.to_bits().hash(hasher);
    format!("{:?}", analysis.outcome).hash(hasher);
}

fn hash_rasters(position: &Position, hasher: &mut DefaultHasher) {
    for size in [64, 100, 128, 256] {
        let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
        let raster = rasterize_packed(position, config);
        raster.bits().hash(hasher);
        for value in raster.dense().iter().chain(raster.scalars()) {
            value.to_bits().hash(hasher);
        }
        settled_for_raster(position, config).hash(hasher);
    }
}

/// One seeded game. Returns plies played, stones captured and the digest.
///
/// `tangent` plays on a legal-set vertex whenever it can: those points touch two
/// constraints exactly, which is where ties and degenerate cells come from.
fn fuzz_game(seed: u64, radius: f64, ruleset: Ruleset, tangent: bool) -> (usize, usize, u64) {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut search = SearchConfig::canary(24);
    search.coarse_pool = COARSE_POOL;
    search.widening_coefficient = 6.0;
    search.maximum_candidates = 321;
    search.leaf_batch = 4;
    let evaluator = FixedEvaluator::new();
    let mut hasher = DefaultHasher::new();
    let mut position = Position::new(radius, Vec::new(), Color::Black)
        .with_komi(next() * 0.2 - 0.1)
        .with_ruleset(ruleset);
    let mut vertices: Vec<Point> = Vec::new();
    let (mut plies, mut captured) = (0, 0);
    while plies < 400 && position.phase() == Phase::Playing {
        let query = if tangent && !vertices.is_empty() && next() < 0.8 {
            vertices[(next() * vertices.len() as f64) as usize]
        } else {
            Point::new(next(), next())
        };
        let nearest = nearest_legal_placement(&position, query);
        hash_point(nearest.point, &mut hasher);
        let played = if nearest.legal && next() >= 0.01 {
            place(&position, nearest.point.x, nearest.point.y)
                .or_else(|_| pass(&position))
        } else {
            pass(&position)
        };
        let result = played.expect("a pass is always available");
        hash_analysis(&result.analysis, &mut hasher);
        vertices.clone_from(&result.analysis.legal_vertices);
        captured += result.captured;
        position = result.position;
        hash_rasters(&position, &mut hasher);
        plies += 1;
        if plies % 10 == 0 && position.phase() == Phase::Playing {
            let result = search_with_evaluator(&position, search, seed, &evaluator)
                .expect("fuzz search");
            format!("{result:?}").hash(&mut hasher);
        }
    }
    (plies, captured, hasher.finish())
}

fn fuzz(games: u64) {
    const RADII: [f64; 5] = [1.0 / 12.0, 1.0 / 18.0, 1.0 / 26.0, 1.0 / 38.0, 1.0 / 50.0];
    let next_game = AtomicU64::new(0);
    let lines = Mutex::new(Vec::new());
    thread::scope(|scope| {
        for _ in 0..thread::available_parallelism().map_or(1, usize::from) {
            scope.spawn(|| loop {
                let game = next_game.fetch_add(1, Ordering::Relaxed);
                if game >= games {
                    break;
                }
                let radius = RADII[(game % 5) as usize];
                let ruleset = if game / 5 % 2 == 0 { Ruleset::Vgo } else { Ruleset::Official };
                let tangent = game / 10 % 2 == 1;
                let (plies, captured, digest) = fuzz_game(game + 1, radius, ruleset, tangent);
                lines.lock().unwrap().push((game, format!(
                    "{game:06} r=1/{:.0} {ruleset:?} tangent={tangent} plies={plies} captured={captured} {digest:016x}",
                    1.0 / radius,
                )));
            });
        }
    });
    let mut lines = lines.into_inner().unwrap();
    lines.sort();
    lines.iter().for_each(|(_, line)| println!("{line}"));
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
    if let Some(games) = arguments.fuzz_games {
        fuzz(games);
        return;
    }
    let positions = load(&arguments.positions);
    if arguments.digest {
        digest(&positions);
        return;
    }
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
