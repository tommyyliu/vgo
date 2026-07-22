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
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::RasterConfig;
use vgo_search::{EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::play_game as run_playout;

#[derive(Debug, Parser)]
#[command(about = "Run a color-swapped held-out arena for an ONNX candidate")]
struct Arguments {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    opponent: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pairs: usize,
    #[arg(long, default_value_t = 16)]
    simulations: u32,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
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
    candidate_color: Color,
    plies: u32,
}

fn load_model(model: PathBuf, arguments: &Arguments) -> Result<BatchedEvaluator, EvaluationError> {
    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: RasterConfig::square(arguments.resolution),
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
                SearchConfig::canary(arguments.simulations),
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
    if arguments.pairs == 0
        || arguments.simulations == 0
        || arguments.maximum_plies == 0
        || arguments.threads == 0
        || arguments.maximum_batch == 0
        || arguments.resolution == 0
    {
        return Err("arena counts, simulations, and dimensions must be positive".into());
    }
    let candidate = load_model(arguments.candidate.clone(), &arguments)?;
    let opponent = arguments
        .opponent
        .clone()
        .map(|model| load_model(model, &arguments))
        .transpose()?;
    let games = arguments.pairs * 2;
    let next_game = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut handles = Vec::with_capacity(arguments.threads);
    for _ in 0..arguments.threads {
        let arguments = Arc::clone(&arguments);
        let candidate = candidate.clone();
        let opponent = opponent.clone();
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
                    arguments.seed + (game / 2) as u64,
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
    let candidate_metrics = candidate.metrics();
    let opponent_name = if arguments.opponent.is_some() {
        "onnx"
    } else {
        "naive"
    };
    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"vgo.arena.v1\",\n",
            "  \"opponent\": \"{}\",\n",
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
            "  \"average_plies\": {:.3},\n",
            "  \"wall_seconds\": {:.6},\n",
            "  \"model_evaluations\": {},\n",
            "  \"model_batches\": {},\n",
            "  \"failures\": {}\n",
            "}}"
        ),
        opponent_name,
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
        plies as f64 / games as f64,
        elapsed,
        candidate_metrics.requests,
        candidate_metrics.batches,
        candidate_metrics.failures,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::wilson_interval;

    #[test]
    fn empty_arena_interval_is_uninformative() {
        assert_eq!(wilson_interval(0.0, 0), (0.0, 1.0));
    }
}
