#![forbid(unsafe_code)]

//! Play one model against itself at two different playout budgets.
//!
//! `vgo-arena` drives both seats at a single `--simulations`, so it cannot
//! answer whether search depth is buying strength. This binary holds the
//! evaluator fixed and varies only the simulation count, which isolates the
//! search path -- in particular the coarse->fine candidate sampler selected by
//! `--coarse-pool`. If the high-playout seat does not beat the low-playout seat,
//! the fault is in search rather than in policy quality.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{ArgAction, Parser};
use vgo_core::{Color, Outcome, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::RasterConfig;
use vgo_search::{EvaluationError, SearchConfig, search_with_evaluator};
use vgo_selfplay::play_game as run_playout;

#[derive(Debug, Parser)]
#[command(about = "Play one ONNX model against itself at two playout budgets")]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    /// Simulations for the high-playout seat.
    #[arg(long, default_value_t = 128)]
    high: u32,
    /// Simulations for the low-playout seat.
    #[arg(long, default_value_t = 16)]
    low: u32,
    /// Colour-swapped pairs; each pair is two games.
    #[arg(long, default_value_t = 16)]
    pairs: usize,
    /// Fine cells per coarse sampling region; zero uses legacy candidates.
    #[arg(long, default_value_t = 0)]
    coarse_pool: usize,
    #[arg(long = "max-plies", default_value_t = 128)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 1)]
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
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

#[derive(Clone, Copy)]
struct GameResult {
    completed: bool,
    outcome: Option<Outcome>,
    high_color: Color,
    plies: u32,
}

#[derive(Clone, Copy, Default)]
struct Report {
    games: usize,
    completed: usize,
    high_wins: usize,
    low_wins: usize,
    draws: usize,
    high_points: f64,
    plies: u64,
}

impl Report {
    fn add(&mut self, game: GameResult) {
        self.games += 1;
        self.plies += u64::from(game.plies);
        if !game.completed {
            return;
        }
        self.completed += 1;
        match game.outcome.and_then(|outcome| outcome.winner) {
            Some(winner) if winner == game.high_color => {
                self.high_wins += 1;
                self.high_points += 1.0;
            }
            Some(_) => self.low_wins += 1,
            None => {
                self.draws += 1;
                self.high_points += 0.5;
            }
        }
    }
}

fn validate_arguments(arguments: &Arguments) -> Result<(), &'static str> {
    if arguments.pairs == 0
        || arguments.high == 0
        || arguments.low == 0
        || arguments.maximum_plies == 0
        || arguments.threads == 0
        || arguments.maximum_batch == 0
        || arguments.resolution == 0
        || arguments.policy_resolution == 0
    {
        return Err("duel counts, simulations, and dimensions must be positive");
    }
    if arguments.coarse_pool > arguments.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution");
    }
    if !arguments.radius.is_finite() || arguments.radius <= 0.0 || arguments.radius >= 0.5 {
        return Err("--radius must be finite and between zero and one half");
    }
    Ok(())
}

fn load_model(arguments: &Arguments) -> Result<BatchedEvaluator, EvaluationError> {
    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model: arguments.model.clone(),
        raster: RasterConfig::square(arguments.resolution),
        policy: Some(RasterConfig::square(arguments.policy_resolution)),
        maximum_batch: arguments.maximum_batch,
        provider: arguments.provider,
        device_id: 0,
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

fn search_config(simulations: u32, coarse_pool: usize) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config
}

fn play_game(
    evaluator: &BatchedEvaluator,
    high_color: Color,
    seed: u64,
    arguments: &Arguments,
) -> Result<GameResult, EvaluationError> {
    let playout = run_playout(
        Position::new(arguments.radius, Vec::new(), Color::Black),
        arguments.maximum_plies,
        |position, _| {
            let simulations = if position.to_move() == high_color {
                arguments.high
            } else {
                arguments.low
            };
            search_with_evaluator(
                position,
                search_config(simulations, arguments.coarse_pool),
                seed,
                evaluator,
            )
        },
        |_| {},
    )?;
    Ok(GameResult {
        completed: playout.completed(),
        outcome: playout.outcome,
        high_color,
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
    let evaluator = Arc::new(load_model(&arguments)?);
    let started = Instant::now();

    let next_pair = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let workers = arguments.threads.min(arguments.pairs);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let next_pair = Arc::clone(&next_pair);
        let evaluator = Arc::clone(&evaluator);
        let arguments = Arc::clone(&arguments);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            loop {
                let pair = next_pair.fetch_add(1, Ordering::Relaxed);
                if pair >= arguments.pairs {
                    break;
                }
                let seed = arguments.seed.wrapping_add(pair as u64);
                for high_color in [Color::Black, Color::White] {
                    let result = play_game(&evaluator, high_color, seed, &arguments);
                    if sender.send(result).is_err() {
                        return;
                    }
                }
            }
        }));
    }
    drop(sender);

    let mut report = Report::default();
    let mut failure = None;
    for result in receiver {
        match result {
            Ok(game) => report.add(game),
            Err(error) => failure = Some(error),
        }
    }
    for handle in handles {
        handle.join().expect("duel worker did not panic");
    }
    if let Some(error) = failure {
        return Err(error.into());
    }

    let score = if report.completed == 0 {
        0.0
    } else {
        report.high_points / report.completed as f64
    };
    let (lower, upper) = wilson_interval(report.high_points, report.completed);
    let wall = started.elapsed().as_secs_f64();
    println!(
        concat!(
            "{{\n",
            "  \"model\": {:?},\n",
            "  \"provider\": {:?},\n",
            "  \"coarse_pool\": {},\n",
            "  \"high_simulations\": {},\n",
            "  \"low_simulations\": {},\n",
            "  \"pairs\": {},\n",
            "  \"games\": {},\n",
            "  \"completed\": {},\n",
            "  \"truncated\": {},\n",
            "  \"high_wins\": {},\n",
            "  \"low_wins\": {},\n",
            "  \"draws\": {},\n",
            "  \"high_score\": {:.6},\n",
            "  \"score_ci95\": [{:.6}, {:.6}],\n",
            "  \"average_plies\": {:.3},\n",
            "  \"wall_seconds\": {:.3}\n",
            "}}"
        ),
        arguments.model.display().to_string(),
        arguments.provider.as_str(),
        arguments.coarse_pool,
        arguments.high,
        arguments.low,
        arguments.pairs,
        report.games,
        report.completed,
        report.games - report.completed,
        report.high_wins,
        report.low_wins,
        report.draws,
        score,
        lower,
        upper,
        report.plies as f64 / report.games.max(1) as f64,
        wall,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use vgo_core::Color;

    use super::{Arguments, GameResult, Report, validate_arguments, wilson_interval};

    fn parsed(extra: &[&str]) -> Arguments {
        let mut argv = vec!["vgo-playout-duel", "--model", "model.onnx"];
        argv.extend_from_slice(extra);
        Arguments::try_parse_from(argv).expect("CLI parses")
    }

    #[test]
    fn defaults_use_the_legacy_candidate_path() {
        let arguments = parsed(&[]);
        assert_eq!(arguments.coarse_pool, 0);
        assert_eq!(arguments.high, 128);
        assert_eq!(arguments.low, 16);
        assert!(validate_arguments(&arguments).is_ok());
    }

    #[test]
    fn oversized_coarse_pool_is_rejected() {
        let arguments = parsed(&["--coarse-pool", "256", "--policy-resolution", "128"]);
        assert!(validate_arguments(&arguments).is_err());
        // The pool is bounded by the placement grid, not the render raster.
        let decoupled = parsed(&["--coarse-pool", "17", "--resolution", "16", "--policy-resolution", "32"]);
        assert!(validate_arguments(&decoupled).is_ok());
    }

    #[test]
    fn zero_simulations_are_rejected() {
        assert!(validate_arguments(&parsed(&["--low", "0"])).is_err());
        assert!(validate_arguments(&parsed(&["--high", "0"])).is_err());
    }

    #[test]
    fn truncated_games_do_not_score() {
        let mut report = Report::default();
        report.add(GameResult {
            completed: false,
            outcome: None,
            high_color: Color::Black,
            plies: 128,
        });
        assert_eq!(report.games, 1);
        assert_eq!(report.completed, 0);
        assert_eq!(report.high_points, 0.0);
    }

    #[test]
    fn draws_score_one_half_for_each_seat() {
        let mut report = Report::default();
        report.add(GameResult {
            completed: true,
            outcome: Some(vgo_core::Outcome {
                winner: None,
                margin: 0.0,
            }),
            high_color: Color::Black,
            plies: 40,
        });
        assert_eq!(report.draws, 1);
        assert_eq!(report.high_points, 0.5);
    }

    #[test]
    fn empty_confidence_interval_is_uninformative() {
        assert_eq!(wilson_interval(0.0, 0), (0.0, 1.0));
    }
}
