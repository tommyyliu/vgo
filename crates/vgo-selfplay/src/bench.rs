//! Time each large CPU path on a fixed set of real positions.
//!
//! Every path the generator spends CPU on, measured alone and single-threaded
//! so numbers from different commits are comparable: applying a move, the
//! board analysis, the legal-set vertex index, the settled plane, the packed
//! raster at both sizes the loop has used, the search's fine grid, and a whole
//! search with a fixed policy standing in for the network. Inference is left
//! out on purpose: it measures the GPU and the driver, not this code.
//!
//! Positions come from `benchmarks/positions.txt`, 16 at radius 1/38 and 16 at
//! 1/18, and each path is reported per board size because cost scales with
//! stone count. `scripts/bench.sh` runs this, keeps the history, and compares.
//!
//!     vgo-bench [--positions PATH] [--seconds S] [--only NAME]

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use vgo_core::{
    Analysis, Color, LegalSetIndex, Point, Position, Stone, nearest_legal_placement, place,
};
use vgo_raster::{DensePolicy, RasterConfig, RasterKind, packed::rasterize_packed, settled_for_raster};
use vgo_search::{
    Evaluation, EvaluationError, Evaluator, Policy, SearchConfig, search_with_evaluator,
};

#[derive(Parser)]
struct Arguments {
    /// Fixture positions, one per line.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benchmarks/positions.txt"))]
    positions: PathBuf,
    /// Time budget per path and board size, after one warm-up pass.
    #[arg(long, default_value_t = 2.0)]
    seconds: f64,
    /// Run only the paths whose name contains this.
    #[arg(long)]
    only: Option<String>,
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
            let colour = |value: f64| if value == 0.0 { Color::Black } else { Color::White };
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
        Self { logits: Arc::new(logits) }
    }
}

impl Evaluator for FixedEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        // Varies with the position so backed-up values are not all equal.
        let value = ((position.stones().len() % 7) as f64 - 3.0) / 10.0;
        let policy = DensePolicy::new(RasterConfig::square(POLICY_RESOLUTION), self.logits.to_vec());
        Ok(Evaluation::new(value, Box::new(policy)))
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
    let positions = load(&arguments.positions);
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
                let target = Point::new(0.13 + 0.57 * ((index * 7) % 11) as f64 / 11.0, 0.21 + 0.6 * ((index * 5) % 13) as f64 / 13.0);
                nearest_legal_placement(position, target).point
            })
            .collect();
        let raster = |size| RasterConfig::square_of(size, RasterKind::CompactRadius);
        let count = positions.len();
        let mut run = |name: &str, ops: usize, work: &mut dyn FnMut()| {
            if arguments.only.as_deref().is_some_and(|only| !name.contains(only)) {
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
                let _ = black_box(search_with_evaluator(position, search, index as u64, &evaluator));
            }
        });
    }
    println!("{{{}}}", results.join(","));
}
