#![forbid(unsafe_code)]

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::{ArgAction, Parser};
use vgo_core::{Color, Outcome, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, BrokerMetrics, OnnxBatchService, OnnxProvider,
    OnnxServiceConfig,
};
use vgo_raster::RasterConfig;
use vgo_search::{EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::play_game as run_playout;

#[derive(Debug, Parser)]
#[command(about = "Run a color-swapped held-out arena for an ONNX candidate")]
struct Arguments {
    #[arg(long)]
    candidate: PathBuf,
    /// Repeatable. Each opponent plays `--pairs` color-swapped pairs against the
    /// same loaded candidate and emits its own JSON record. Batching them here
    /// amortizes the provider's per-process model load and warmup, which is
    /// ~21s against ~0.93s per additional pair: six opponents in one process
    /// costs about what one does. Omit entirely to play the naive evaluator.
    #[arg(long)]
    opponent: Vec<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pairs: usize,
    #[arg(long, default_value_t = 16)]
    simulations: u32,
    /// Fine cells per coarse sampling region; zero uses legacy candidates.
    #[arg(long, default_value_t = 0)]
    coarse_pool: usize,
    /// Leaves evaluated together per simulation round; above one a single game
    /// keeps that many evaluations in flight. Both seats must use the same value
    /// for a fair comparison, since it changes which nodes get explored.
    #[arg(long, default_value_t = 1)]
    leaf_batch: usize,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Placement grid the policy head emits; independent of the render
    /// resolution. Must match the value the model was exported with.
    #[arg(long, default_value_t = 32)]
    policy_resolution: usize,
    #[arg(long, default_value_t = 96)]
    resolution: usize,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
    #[arg(long, default_value_t = 900_001)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

#[derive(Clone, Copy)]
struct GameResult {
    completed: bool,
    outcome: Option<Outcome>,
    candidate_color: Color,
    plies: u32,
}

fn load_model(model: PathBuf, arguments: &Arguments) -> Result<BatchedEvaluator, EvaluationError> {
    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: RasterConfig::square(arguments.resolution),
        policy: Some(RasterConfig::square(arguments.policy_resolution)),
        maximum_batch: arguments.maximum_batch,
        provider: arguments.provider,
        device_id: arguments.device_id,
        fp16: arguments.fp16,
        cache_directory: arguments.cache_directory.clone(),
    })?;
    BatchedEvaluator::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(arguments.delay_ms),
            queue_capacity: (arguments.threads * 4).max(arguments.maximum_batch * 2),
        },
        service,
    )
}

fn search_config(simulations: u32, coarse_pool: usize, leaf_batch: usize) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config.leaf_batch = leaf_batch.max(1);
    config
}

fn validate_arguments(arguments: &Arguments) -> Result<(), &'static str> {
    if arguments.pairs == 0
        || arguments.simulations == 0
        || arguments.maximum_plies == 0
        || arguments.threads == 0
        || arguments.maximum_batch == 0
        || arguments.resolution == 0
        || arguments.policy_resolution == 0
        || arguments.device_id < 0
    {
        return Err("arena counts, simulations, and dimensions must be positive");
    }
    if arguments.coarse_pool > arguments.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution");
    }
    if !arguments.radius.is_finite() || arguments.radius <= 0.0 || arguments.radius >= 0.5 {
        return Err("--radius must be finite and between zero and one half");
    }
    Ok(())
}

fn play_game(
    candidate: &BatchedEvaluator,
    opponent: Option<&BatchedEvaluator>,
    candidate_color: Color,
    seed: u64,
    arguments: &Arguments,
) -> Result<GameResult, EvaluationError> {
    let naive = NaiveEvaluator;
    let playout = run_playout(
        Position::new(arguments.radius, Vec::new(), Color::Black),
        arguments.maximum_plies,
        |position, _| {
            let evaluator: &dyn Evaluator = if position.to_move() == candidate_color {
                candidate
            } else if let Some(opponent) = opponent {
                opponent
            } else {
                &naive
            };
            search_with_evaluator(
                position,
                search_config(
                    arguments.simulations,
                    arguments.coarse_pool,
                    arguments.leaf_batch,
                ),
                seed,
                evaluator,
            )
        },
        |_| {},
    )?;
    Ok(GameResult {
        completed: playout.completed(),
        outcome: playout.outcome,
        candidate_color,
        plies: playout.stats.plies,
    })
}

fn wilson_interval(points: f64, games: usize) -> (f64, f64) {
    if games == 0 {
        return (0.0, 1.0);
    }
    let count = games as f64;
    let proportion = points / count;
    let z = 1.959_963_984_540_054;
    let denominator = 1.0 + z * z / count;
    let center = (proportion + z * z / (2.0 * count)) / denominator;
    let margin = z
        * ((proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)).sqrt())
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arc::new(Arguments::parse());
    if let Err(message) = validate_arguments(&arguments) {
        return Err(message.into());
    }
    let candidate = load_model(arguments.candidate.clone(), &arguments)?;
    // One record per opponent, or a single naive-evaluator record when none are
    // given. The candidate is loaded once and reused across every match.
    let opponents: Vec<Option<PathBuf>> = if arguments.opponent.is_empty() {
        vec![None]
    } else {
        arguments.opponent.iter().cloned().map(Some).collect()
    };
    for (index, opponent_path) in opponents.into_iter().enumerate() {
        let opponent = opponent_path
            .clone()
            .map(|model| load_model(model, &arguments))
            .transpose()?;
        // Distinct seeds per opponent, or every match would replay one game set.
        let seed_base = arguments.seed + (index as u64) * 1_000_003;
        run_match(
            &arguments,
            &candidate,
            opponent.as_ref(),
            opponent_path.as_deref(),
            seed_base,
        )?;
    }
    Ok(())
}

fn run_match(
    arguments: &Arc<Arguments>,
    candidate: &BatchedEvaluator,
    opponent: Option<&BatchedEvaluator>,
    opponent_path: Option<&std::path::Path>,
    seed_base: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let games = arguments.pairs * 2;
    let next_game = Arc::new(AtomicUsize::new(0));
    // Broker counters are cumulative over the process, and the candidate is
    // shared across every opponent, so report this match's delta.
    let baseline = candidate.metrics();
    let started = Instant::now();
    let mut handles = Vec::with_capacity(arguments.threads);
    for _ in 0..arguments.threads {
        let arguments = Arc::clone(arguments);
        let candidate = candidate.clone();
        let opponent = opponent.cloned();
        let next_game = Arc::clone(&next_game);
        handles.push(thread::spawn(move || {
            let mut results = Vec::new();
            loop {
                let game = next_game.fetch_add(1, Ordering::Relaxed);
                if game >= arguments.pairs * 2 {
                    break;
                }
                let candidate_color = if game.is_multiple_of(2) {
                    Color::Black
                } else {
                    Color::White
                };
                results.push(play_game(
                    &candidate,
                    opponent.as_ref(),
                    candidate_color,
                    seed_base + (game / 2) as u64,
                    &arguments,
                )?);
            }
            Ok::<_, EvaluationError>(results)
        }));
    }
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("arena worker"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let elapsed = started.elapsed().as_secs_f64();
    let mut wins = 0_usize;
    let mut losses = 0_usize;
    let mut draws = 0_usize;
    let mut completed = 0_usize;
    let mut points = 0.0;
    let mut plies = 0_u64;
    for result in &results {
        plies += u64::from(result.plies);
        if !result.completed {
            continue;
        }
        completed += 1;
        match result.outcome.expect("completed game has outcome").winner {
            Some(winner) if winner == result.candidate_color => {
                wins += 1;
                points += 1.0;
            }
            Some(_) => losses += 1,
            None => {
                draws += 1;
                points += 0.5;
            }
        }
    }
    let score = if completed == 0 {
        0.0
    } else {
        points / completed as f64
    };
    let interval = wilson_interval(points, completed);
    let current = candidate.metrics();
    let candidate_metrics = BrokerMetrics {
        requests: current.requests - baseline.requests,
        batches: current.batches - baseline.batches,
        positions: current.positions - baseline.positions,
        maximum_batch: current.maximum_batch,
        failures: current.failures - baseline.failures,
        encoding_nanoseconds: current.encoding_nanoseconds - baseline.encoding_nanoseconds,
        queue_nanoseconds: current.queue_nanoseconds - baseline.queue_nanoseconds,
        inference_nanoseconds: current.inference_nanoseconds - baseline.inference_nanoseconds,
    };
    let opponent_name = if opponent_path.is_some() {
        "onnx"
    } else {
        "naive"
    };
    let opponent_model = opponent_path
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"vgo.arena.v1\",\n",
            "  \"opponent\": \"{}\",\n",
            "  \"opponent_model\": \"{}\",\n",
            "  \"pairs\": {},\n",
            "  \"games\": {},\n",
            "  \"completed\": {},\n",
            "  \"truncated\": {},\n",
            "  \"candidate_wins\": {},\n",
            "  \"candidate_losses\": {},\n",
            "  \"draws\": {},\n",
            "  \"candidate_score\": {:.6},\n",
            "  \"score_ci95\": [{:.6}, {:.6}],\n",
            "  \"simulations_per_move\": {},\n",
            "  \"coarse_pool\": {},\n",
            "  \"average_plies\": {:.3},\n",
            "  \"wall_seconds\": {:.6},\n",
            "  \"model_evaluations\": {},\n",
            "  \"model_batches\": {},\n",
            "  \"encoding_seconds\": {:.3},\n",
            "  \"queue_seconds\": {:.3},\n",
            "  \"inference_seconds\": {:.3},\n",
            "  \"failures\": {}\n",
            "}}"
        ),
        opponent_name,
        opponent_model,
        arguments.pairs,
        games,
        completed,
        games - completed,
        wins,
        losses,
        draws,
        score,
        interval.0,
        interval.1,
        arguments.simulations,
        arguments.coarse_pool,
        plies as f64 / games as f64,
        elapsed,
        candidate_metrics.requests,
        candidate_metrics.batches,
        candidate_metrics.encoding_nanoseconds as f64 / 1e9,
        candidate_metrics.queue_nanoseconds as f64 / 1e9,
        candidate_metrics.inference_nanoseconds as f64 / 1e9,
        candidate_metrics.failures,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Arguments, search_config, validate_arguments, wilson_interval};

    #[test]
    fn empty_arena_interval_is_uninformative() {
        assert_eq!(wilson_interval(0.0, 0), (0.0, 1.0));
    }

    #[test]
    fn coarse_pool_cli_defaults_to_legacy_and_accepts_an_override() {
        let default = Arguments::try_parse_from(["vgo-arena", "--candidate", "candidate.onnx"])
            .expect("default CLI parses");
        assert_eq!(default.coarse_pool, 0);

        let configured = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--coarse-pool",
            "8",
        ])
        .expect("coarse sampling options parse");
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn coarse_sampling_is_applied_to_search_config() {
        let configured = search_config(19, 8, 1);
        assert_eq!(configured.simulations, 19);
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn invalid_coarse_sampling_config_is_rejected_before_arena() {
        // The pool counts fine cells per coarse region on the placement grid,
        // which is now independent of the render resolution.
        let oversized_pool = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--policy-resolution",
            "16",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(
            validate_arguments(&oversized_pool),
            Err("--coarse-pool must not exceed --policy-resolution")
        );

        // A pool larger than the raster but within the placement grid is fine:
        // the two resolutions are unrelated.
        let coarse_raster = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--resolution",
            "16",
            "--policy-resolution",
            "32",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(validate_arguments(&coarse_raster), Ok(()));
    }

    #[test]
    fn invalid_radius_is_rejected_before_arena() {
        for radius in ["0", "0.5", "NaN"] {
            let arguments = Arguments::try_parse_from([
                "vgo-arena",
                "--candidate",
                "candidate.onnx",
                "--radius",
                radius,
            ])
            .expect("CLI syntax parses");
            assert_eq!(
                validate_arguments(&arguments),
                Err("--radius must be finite and between zero and one half")
            );
        }
    }
}
