//! Hash engine output, so two builds can be compared bit for bit.
//!
//! An optimization that claims to change nothing has to leave this output
//! unchanged, because a model only ever sees these bytes.
//!
//!     vgo-verify                      # digest of searches on the fixtures
//!     vgo-verify --fuzz-games N       # one line per seeded whole game
//!
//! The fixture digest sees two board sizes under one ruleset. The fuzz plays
//! games across board sizes and both rulesets and hashes every ply's geometry,
//! legal set, liveness, score and rasters. `correctness/check.sh` compares both
//! against `correctness/golden.txt`.
//!
//! These compare builds. They find differences and prove nothing: claims about
//! the engine belong in `spec/LEMMAS.md`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use clap::Parser;
use vgo_bench::{COARSE_POOL, FixedEvaluator, SEARCH_SIMULATIONS, load};
use vgo_core::{
    Analysis, Color, Phase, Point, Position, Ruleset, nearest_legal_placement, pass, place,
};
use vgo_raster::{RasterConfig, RasterKind, packed::rasterize_packed, settled_for_raster};
use vgo_search::{Evaluation, EvaluationError, Evaluator, SearchConfig, search_with_evaluator};

#[derive(Parser)]
struct Arguments {
    /// Digest this many seeded whole games instead of the fixtures.
    #[arg(long)]
    fuzz_games: Option<u64>,
    /// Fixture positions, one per line.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../performance/positions.txt"))]
    positions: PathBuf,
}

fn main() {
    let arguments = Arguments::parse();
    match arguments.fuzz_games {
        Some(games) => fuzz(games),
        None => digest(&load(&arguments.positions)),
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
        cell.polygon
            .iter()
            .for_each(|point| hash_point(*point, hasher));
        for edge in &cell.edges {
            hash_point(edge.start, hasher);
            hash_point(edge.end, hasher);
            format!("{:?}", edge.source).hash(hasher);
        }
    }
    analysis.geometry.adjacency.hash(hasher);
    analysis.geometry.groups.hash(hasher);
    analysis
        .legal_vertices
        .iter()
        .for_each(|point| hash_point(*point, hasher));
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
            place(&position, nearest.point.x, nearest.point.y).or_else(|_| pass(&position))
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
            let result =
                search_with_evaluator(&position, search, seed, &evaluator).expect("fuzz search");
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
                let ruleset = if (game / 5).is_multiple_of(2) { Ruleset::Vgo } else { Ruleset::Official };
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
