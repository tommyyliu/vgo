#![forbid(unsafe_code)]

use std::{
    collections::HashSet,
    env,
    path::PathBuf,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, Phase, Point, Position};
use vgo_inference::{PythonEvaluator, PythonProcessConfig};
use vgo_raster::RasterConfig;
use vgo_search::{Action, EvaluationError, Evaluator, SearchConfig, search_with_evaluator};

fn path_argument(arguments: &[String], name: &str, default: PathBuf) -> PathBuf {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map_or(default, |pair| PathBuf::from(&pair[1]))
}

fn value_argument<T>(arguments: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(pair) = arguments.windows(2).find(|pair| pair[0] == name) else {
        return Ok(default);
    };
    pair[1]
        .parse()
        .map_err(|error| format!("invalid value for {name}: {error}"))
}

fn hash_word(mut hash: u64, word: u64) -> u64 {
    for byte in word.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn position_hash(position: &Position) -> u64 {
    let mut hash = hash_word(0xcbf2_9ce4_8422_2325, position.radius().to_bits());
    hash = hash_word(hash, u64::from(position.consecutive_passes()));
    hash = hash_word(hash, u64::from(position.to_move() == Color::White));
    let mut stones = position
        .stones()
        .iter()
        .map(|stone| {
            (
                stone.x.to_bits(),
                stone.y.to_bits(),
                u64::from(stone.color == Color::White),
            )
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    for (x, y, color) in stones {
        hash = hash_word(hash, x);
        hash = hash_word(hash, y);
        hash = hash_word(hash, color);
    }
    hash
}

fn play_model_game(
    evaluator: &PythonEvaluator,
    seed: u64,
    simulations: u32,
    maximum_plies: u32,
) -> Result<(bool, u32), EvaluationError> {
    let mut position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let mut seen = HashSet::new();
    seen.insert(position_hash(&position));
    for ply in 0..maximum_plies {
        let result = search_with_evaluator(
            &position,
            SearchConfig::canary(simulations),
            seed,
            evaluator,
        )?;
        let mut selected = None;
        for action in result.actions_by_preference(position.to_move()) {
            let transition = action.apply(&position);
            if transition.position.phase() == Phase::Finished
                || !seen.contains(&position_hash(&transition.position))
            {
                selected = Some(transition);
                break;
            }
        }
        let transition = selected.unwrap_or_else(|| Action::Pass.apply(&position));
        position = transition.position;
        if position.phase() == Phase::Finished {
            return Ok((true, ply + 1));
        }
        seen.insert(position_hash(&position));
    }
    Ok((false, maximum_plies))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let root = env::current_dir()?;
    let python = path_argument(
        &arguments,
        "--python",
        root.join("training/.venv/Scripts/python.exe"),
    );
    let working_directory = path_argument(&arguments, "--training", root.join("training"));
    let checkpoint = path_argument(
        &arguments,
        "--checkpoint",
        root.join("artifacts/raster-demo/model.pt"),
    );
    let actors = value_argument(&arguments, "--actors", 32_usize)?;
    let games = value_argument(&arguments, "--games", actors)?;
    let simulations = value_argument(&arguments, "--simulations", 8_u32)?;
    let maximum_plies = value_argument(&arguments, "--max-plies", 48_u32)?;
    let maximum_batch = value_argument(&arguments, "--maximum-batch", 16_usize)?;
    let maximum_delay_ms = value_argument(&arguments, "--delay-ms", 1_u64)?;
    let torch_threads = value_argument(&arguments, "--torch-threads", 16_usize)?;
    let probe_requests = value_argument(&arguments, "--probe-requests", 1_usize)?;
    if actors == 0
        || games == 0
        || simulations == 0
        || maximum_plies == 0
        || maximum_batch == 0
        || torch_threads == 0
    {
        return Err(
            "actor, game, simulation, ply, batch, and thread counts must be positive".into(),
        );
    }

    let evaluator = PythonEvaluator::spawn(PythonProcessConfig {
        python,
        working_directory,
        checkpoint,
        raster: RasterConfig::square(48),
        maximum_batch,
        maximum_delay: Duration::from_millis(maximum_delay_ms),
        queue_capacity: (actors * 4).max(maximum_batch * 2),
        torch_threads,
    })?;

    let probe_barrier = Arc::new(Barrier::new(probe_requests + 1));
    let mut probe_handles = Vec::with_capacity(probe_requests);
    for _ in 0..probe_requests {
        let evaluator = evaluator.clone();
        let barrier = Arc::clone(&probe_barrier);
        probe_handles.push(thread::spawn(move || {
            let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
            barrier.wait();
            let evaluation = evaluator.evaluate(&position).expect("model evaluation");
            (
                evaluation.current_value,
                evaluation.policy_logit(Action::Pass),
                evaluation.policy_logit(Action::Place(Point::new(0.5, 0.5))),
            )
        }));
    }
    probe_barrier.wait();
    let probe_outputs = probe_handles
        .into_iter()
        .map(|handle| handle.join().expect("evaluation worker"))
        .collect::<Vec<_>>();
    let first = probe_outputs.first().copied().unwrap_or((0.0, 0.0, 0.0));
    assert!(probe_outputs.iter().all(|output| *output == first));

    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let warmup_search = search_with_evaluator(&position, SearchConfig::canary(4), 91, &evaluator)?;
    let before_actors = evaluator.metrics();
    let next_game = Arc::new(AtomicUsize::new(0));
    let actor_barrier = Arc::new(Barrier::new(actors + 1));
    let mut actor_handles = Vec::with_capacity(actors);
    for _ in 0..actors {
        let evaluator = evaluator.clone();
        let barrier = Arc::clone(&actor_barrier);
        let next_game = Arc::clone(&next_game);
        actor_handles.push(thread::spawn(move || {
            barrier.wait();
            let mut reports = Vec::new();
            loop {
                let game = next_game.fetch_add(1, Ordering::Relaxed);
                if game >= games {
                    break;
                }
                reports.push(play_model_game(
                    &evaluator,
                    1_000 + game as u64,
                    simulations,
                    maximum_plies,
                )?);
            }
            Ok::<_, EvaluationError>(reports)
        }));
    }
    let actor_started = Instant::now();
    actor_barrier.wait();
    let actor_reports = actor_handles
        .into_iter()
        .map(|handle| handle.join().expect("game actor"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let actor_wall_seconds = actor_started.elapsed().as_secs_f64();
    let completed_games = actor_reports
        .iter()
        .filter(|(completed, _)| *completed)
        .count();
    let actor_plies = actor_reports
        .iter()
        .map(|(_, plies)| u64::from(*plies))
        .sum::<u64>();
    let metrics = evaluator.metrics();
    let actor_requests = metrics.requests - before_actors.requests;
    let actor_batches = metrics.batches - before_actors.batches;
    let actor_queue_nanoseconds = metrics.queue_nanoseconds - before_actors.queue_nanoseconds;
    let actor_inference_nanoseconds =
        metrics.inference_nanoseconds - before_actors.inference_nanoseconds;

    println!(
        concat!(
            "{{\n",
            "  \"probe_requests\": {},\n",
            "  \"model_value\": {:.6},\n",
            "  \"pass_logit\": {:.6},\n",
            "  \"center_logit\": {:.6},\n",
            "  \"warmup_evaluations\": {},\n",
            "  \"actors\": {},\n",
            "  \"games\": {},\n",
            "  \"simulations_per_move\": {},\n",
            "  \"maximum_batch_config\": {},\n",
            "  \"maximum_delay_ms\": {},\n",
            "  \"torch_threads\": {},\n",
            "  \"completed_games\": {},\n",
            "  \"actor_plies\": {},\n",
            "  \"actor_wall_seconds\": {:.6},\n",
            "  \"actor_requests\": {},\n",
            "  \"actor_batches\": {},\n",
            "  \"average_batch\": {:.3},\n",
            "  \"maximum_batch\": {},\n",
            "  \"failures\": {},\n",
            "  \"games_per_second\": {:.3},\n",
            "  \"plies_per_second\": {:.3},\n",
            "  \"evaluations_per_second\": {:.3},\n",
            "  \"queue_milliseconds_total\": {:.3},\n",
            "  \"inference_milliseconds_total\": {:.3}\n",
            "}}"
        ),
        probe_requests,
        first.0,
        first.1,
        first.2,
        warmup_search.stats.evaluations,
        actors,
        games,
        simulations,
        maximum_batch,
        maximum_delay_ms,
        torch_threads,
        completed_games,
        actor_plies,
        actor_wall_seconds,
        actor_requests,
        actor_batches,
        actor_requests as f64 / actor_batches.max(1) as f64,
        metrics.maximum_batch,
        metrics.failures,
        completed_games as f64 / actor_wall_seconds,
        actor_plies as f64 / actor_wall_seconds,
        actor_requests as f64 / actor_wall_seconds,
        actor_queue_nanoseconds as f64 / 1_000_000.0,
        actor_inference_nanoseconds as f64 / 1_000_000.0,
    );
    Ok(())
}
