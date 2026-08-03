#![forbid(unsafe_code)]

//! CPU microbenchmarks for the steady-state work around self-play generation.
//!
//! This deliberately uses no model runtime. The spatial evaluator below emits a
//! deterministic policy map in-process, so the benchmark covers everything on
//! either side of GPU inference without making CUDA, ONNX Runtime, or a Python
//! environment part of the measurement.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use clap::Parser;
use vgo_core::{
    Analysis, Color, Position, SettledRegion, Stone, is_legal_placement, legal_set_vertices,
};
use vgo_raster::{
    CHANNEL_COUNT, RasterConfig, RasterKind, rasterize, rasterize_any_into, rasterize_into,
};
use vgo_search::{
    Action, Evaluation, EvaluationError, Evaluator, FineGrid, Policy, SearchConfig,
    generate_candidates, sample_candidates, search_at_ply,
};

#[derive(Clone, Debug, Parser)]
#[command(about = "Benchmark CPU stages used by self-play generation")]
struct Config {
    /// Timed samples reported for each case. The median is the headline result.
    #[arg(long, default_value_t = 7)]
    samples: usize,
    /// Approximate duration of each timed sample after automatic calibration.
    #[arg(long, default_value_t = 100)]
    sample_millis: u64,
    /// Untimed calls before calibration, to warm code and allocator paths.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Emit machine-readable JSON rather than the comparison table.
    #[arg(long)]
    json: bool,
    /// Run only stages whose name contains this text.
    #[arg(long)]
    filter: Option<String>,
}

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    iterations: u64,
    median_ns: f64,
    minimum_ns: f64,
}

fn run_case(
    config: &Config,
    name: &'static str,
    mut operation: impl FnMut(),
) -> Option<BenchResult> {
    assert!(config.samples > 0, "--samples must be positive");
    assert!(config.sample_millis > 0, "--sample-millis must be positive");
    if config
        .filter
        .as_ref()
        .is_some_and(|filter| !name.contains(filter))
    {
        return None;
    }
    for _ in 0..config.warmup {
        operation();
    }

    let target = Duration::from_millis(config.sample_millis);
    let mut iterations = 1_u64;
    loop {
        let started = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let elapsed = started.elapsed();
        if elapsed >= target || iterations >= 1_u64 << 32 {
            break;
        }
        let elapsed_ns = elapsed.as_nanos().max(1);
        let target_ns = target.as_nanos();
        let multiplier = (target_ns / elapsed_ns).clamp(2, 16) as u64;
        iterations = iterations.saturating_mul(multiplier);
    }

    let mut samples = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        let started = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        samples.push(started.elapsed().as_secs_f64() * 1.0e9 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    Some(BenchResult {
        name,
        iterations,
        median_ns: samples[samples.len() / 2],
        minimum_ns: samples[0],
    })
}

fn print_results(config: &Config, results: &[BenchResult]) {
    if config.json {
        println!("{{");
        println!("  \"samples\": {},", config.samples);
        println!("  \"sample_millis\": {},", config.sample_millis);
        println!("  \"results\": [");
        for (index, result) in results.iter().enumerate() {
            let comma = if index + 1 == results.len() { "" } else { "," };
            println!(
                "    {{\"stage\": {:?}, \"iterations\": {}, \"median_ns\": {:.3}, \
                 \"minimum_ns\": {:.3}}}{comma}",
                result.name, result.iterations, result.median_ns, result.minimum_ns,
            );
        }
        println!("  ]");
        println!("}}");
        return;
    }

    println!(
        "{:<35} {:>12} {:>12} {:>12} {:>10}",
        "stage", "median us", "min us", "operations/s", "iterations"
    );
    for result in results {
        println!(
            "{:<35} {:>12.3} {:>12.3} {:>12.1} {:>10}",
            result.name,
            result.median_ns / 1.0e3,
            result.minimum_ns / 1.0e3,
            1.0e9 / result.median_ns,
            result.iterations,
        );
    }
}

fn next_random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

/// A deterministic valid position at the active run's radius. Fixture creation
/// is outside every timed region.
fn scattered_position(stones: usize) -> Position {
    let radius = 1.0 / 18.0;
    let mut placed = Vec::with_capacity(stones);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut attempts = 0;
    while placed.len() < stones && attempts < 100_000 {
        attempts += 1;
        let x = radius + (1.0 - 2.0 * radius) * next_random(&mut state);
        let y = radius + (1.0 - 2.0 * radius) * next_random(&mut state);
        let probe = Position::new(radius, placed.clone(), Color::Black);
        if is_legal_placement(&probe, x, y) {
            placed.push(Stone::new(
                x,
                y,
                if placed.len() % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            ));
        }
    }
    assert_eq!(placed.len(), stones, "fixture could not place every stone");
    Position::new(radius, placed, Color::Black).with_komi(0.12)
}

fn policy_logit(row: usize, column: usize) -> f32 {
    let x = column as f32 + 0.5;
    let y = row as f32 + 0.5;
    (x * 0.173).sin() + (y * 0.117).cos() + ((row ^ column) % 7) as f32 * 0.03
}

#[derive(Clone, Copy)]
struct SpatialEvaluator {
    resolution: usize,
}

impl Evaluator for SpatialEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        Ok(Evaluation::new(
            0.0,
            Box::new(SpatialPolicy {
                position: position.clone(),
                resolution: self.resolution,
            }),
        ))
    }
}

struct SpatialPolicy {
    position: Position,
    resolution: usize,
}

impl Policy for SpatialPolicy {
    fn logit(&self, action: Action) -> f64 {
        match action {
            Action::Pass => -1.0,
            Action::Place(point) => {
                let column = (point.x * self.resolution as f64).floor() as usize;
                let row = (point.y * self.resolution as f64).floor() as usize;
                f64::from(policy_logit(
                    row.min(self.resolution - 1),
                    column.min(self.resolution - 1),
                ))
            }
        }
    }

    fn fine_grid(&self, _position: &Position, coarse: usize) -> Option<FineGrid> {
        Some(FineGrid::build(
            &self.position,
            self.resolution,
            self.resolution,
            coarse,
            policy_logit,
        ))
    }
}

fn main() {
    let config = Config::parse();
    let position = scattered_position(20);
    let raster_position = scattered_position(35);
    let semantic_96 = RasterConfig::square(96);
    let compact_96 = RasterConfig::square_of(96, RasterKind::Compact);
    let compact_128 = RasterConfig::square_of(128, RasterKind::Compact);
    let mut semantic_buffer = vec![0.0_f32; CHANNEL_COUNT * semantic_96.pixels()];
    let mut compact_96_buffer = vec![0.0_f32; compact_96.channels() * compact_96.pixels()];
    let mut compact_128_buffer = vec![0.0_f32; compact_128.channels() * compact_128.pixels()];
    let settled_vertices = legal_set_vertices(&raster_position);
    let settled_regions: Vec<_> = (0..raster_position.stones().len())
        .map(|index| SettledRegion::new(&raster_position, index, &settled_vertices))
        .collect();
    let mut contour_buffer = Vec::new();
    let sampling_grid = FineGrid::build(&position, 32, 32, 4, policy_logit);
    let placement = generate_candidates(&position, 32, 50_001)
        .into_iter()
        .find_map(|candidate| match candidate.action {
            Action::Place(point) => Some(Action::Place(point)),
            Action::Pass => None,
        })
        .expect("fixture has a legal placement");
    let evaluator = SpatialEvaluator { resolution: 32 };
    let search_config = SearchConfig {
        simulations: 32,
        initial_candidates: 4,
        maximum_candidates: 32,
        widening_coefficient: 2.0,
        widening_exponent: 0.5,
        exploration: 1.5,
        maximum_depth: 32,
        coarse_pool: 4,
        temperature: 1.0,
        temperature_plies: 30,
        leaf_batch: 1,
    };

    let mut results = Vec::new();
    results.extend(run_case(&config, "analysis/20-stone", || {
        let analysis = Analysis::new(black_box(&position));
        black_box((analysis.score, analysis.legal_vertices.len()));
    }));
    results.extend(run_case(&config, "raster-semantic/96/35/into", || {
        rasterize_into(
            black_box(&raster_position),
            semantic_96,
            &mut semantic_buffer,
        );
        black_box(semantic_buffer[6 * semantic_96.pixels() + 101]);
    }));
    results.extend(run_case(&config, "raster-compact/96/35/alloc", || {
        let encoded = rasterize(black_box(&raster_position), compact_96);
        black_box(encoded.data()[2 * compact_96.pixels() + 101]);
    }));
    results.extend(run_case(&config, "raster-compact/96/35/into", || {
        rasterize_any_into(
            black_box(&raster_position),
            compact_96,
            &mut compact_96_buffer,
        );
        black_box(compact_96_buffer[2 * compact_96.pixels() + 101]);
    }));
    results.extend(run_case(&config, "raster-compact/128/35/alloc", || {
        let encoded = rasterize(black_box(&raster_position), compact_128);
        black_box(encoded.data()[2 * compact_128.pixels() + 101]);
    }));
    results.extend(run_case(&config, "raster-compact/128/35/into", || {
        rasterize_any_into(
            black_box(&raster_position),
            compact_128,
            &mut compact_128_buffer,
        );
        black_box(compact_128_buffer[2 * compact_128.pixels() + 101]);
    }));
    results.extend(run_case(&config, "settled-region-build/35", || {
        let regions: Vec<_> = (0..raster_position.stones().len())
            .map(|index| SettledRegion::new(&raster_position, index, &settled_vertices))
            .collect();
        black_box(regions);
    }));
    results.extend(run_case(&config, "settled-contours/128/35/alloc", || {
        let points: usize = settled_regions
            .iter()
            .map(|region| region.contour_within(1.0 / (3.0 * 128.0)).len())
            .sum();
        black_box(points);
    }));
    results.extend(run_case(&config, "settled-contours/128/35/reuse", || {
        let mut points = 0;
        for region in &settled_regions {
            region.contour_within_into(1.0 / (3.0 * 128.0), &mut contour_buffer);
            points += contour_buffer.len();
        }
        black_box(points);
    }));
    results.extend(run_case(&config, "fine-grid/32x32/pool-4", || {
        let grid = FineGrid::build(black_box(&position), 32, 32, 4, policy_logit);
        black_box(grid);
    }));

    let mut sample_state = 0xd1b5_4a32_d192_ed03_u64;
    results.extend(run_case(&config, "sample/one-candidate", || {
        let sampled = sample_candidates(&sampling_grid, 1, || next_random(&mut sample_state));
        black_box(sampled);
    }));
    results.extend(run_case(&config, "sample/32-candidates", || {
        let sampled = sample_candidates(&sampling_grid, 32, || next_random(&mut sample_state));
        black_box(sampled);
    }));
    results.extend(run_case(&config, "candidate-sequence/32", || {
        black_box(generate_candidates(black_box(&position), 32, 50_001));
    }));
    results.extend(run_case(&config, "move-apply/placement", || {
        let transition = placement.apply(black_box(&position));
        black_box((transition.captured, transition.position.stones().len()));
    }));
    results.extend(run_case(&config, "mcts/32-sim/spatial", || {
        let result = search_at_ply(black_box(&position), search_config, 50_001, &evaluator, 12)
            .expect("in-process evaluator cannot fail");
        black_box((result.action, result.stats));
    }));

    print_results(&config, &results);
}
