#![forbid(unsafe_code)]

use std::{
    collections::HashSet,
    env,
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, Phase, Point, Position};
use vgo_inference::{PythonEvaluator, PythonProcessConfig};
use vgo_raster::RasterConfig;
use vgo_search::{Action, EvaluationError, Evaluator, SearchConfig, search_with_evaluator};

fn argument(arguments: &[String], name: &str, default: PathBuf) -> PathBuf {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map_or(default, |pair| PathBuf::from(&pair[1]))
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

fn play_model_game(evaluator: &PythonEvaluator, seed: u64) -> Result<(bool, u32), EvaluationError> {
    let mut position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let mut seen = HashSet::new();
    seen.insert(position_hash(&position));
    for ply in 0..48 {
        let result = search_with_evaluator(&position, SearchConfig::canary(8), seed, evaluator)?;
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
    Ok((false, 48))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let root = env::current_dir()?;
    let python = argument(
        &arguments,
        "--python",
        root.join("training/.venv/Scripts/python.exe"),
    );
    let working_directory = argument(&arguments, "--training", root.join("training"));
    let checkpoint = argument(
        &arguments,
        "--checkpoint",
        root.join("artifacts/raster-demo/model.pt"),
    );
    let evaluator = PythonEvaluator::spawn(PythonProcessConfig {
        python,
        working_directory,
        checkpoint,
        raster: RasterConfig::square(48),
        maximum_batch: 16,
        maximum_delay: Duration::from_millis(5),
        queue_capacity: 64,
        torch_threads: 2,
    })?;

    let workers = 16;
    let barrier = Arc::new(Barrier::new(workers + 1));
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let evaluator = evaluator.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
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
    barrier.wait();
    let outputs = handles
        .into_iter()
        .map(|handle| handle.join().expect("evaluation worker"))
        .collect::<Vec<_>>();
    let first = outputs[0];
    assert!(outputs.iter().all(|output| *output == first));

    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let search = search_with_evaluator(&position, SearchConfig::canary(16), 91, &evaluator)?;
    let actor_barrier = Arc::new(Barrier::new(workers + 1));
    let mut actor_handles = Vec::with_capacity(workers);
    for actor in 0..workers {
        let evaluator = evaluator.clone();
        let barrier = Arc::clone(&actor_barrier);
        actor_handles.push(thread::spawn(move || {
            barrier.wait();
            play_model_game(&evaluator, 1_000 + actor as u64)
        }));
    }
    let actor_started = Instant::now();
    actor_barrier.wait();
    let actor_reports = actor_handles
        .into_iter()
        .map(|handle| handle.join().expect("game actor"))
        .collect::<Result<Vec<_>, _>>()?;
    let completed_games = actor_reports
        .iter()
        .filter(|(completed, _)| *completed)
        .count();
    let actor_plies = actor_reports
        .iter()
        .map(|(_, plies)| u64::from(*plies))
        .sum::<u64>();
    let actor_wall_seconds = actor_started.elapsed().as_secs_f64();
    let metrics = evaluator.metrics();
    println!(
        concat!(
            "{{\n",
            "  \"concurrent_requests\": {},\n",
            "  \"model_value\": {:.6},\n",
            "  \"pass_logit\": {:.6},\n",
            "  \"center_logit\": {:.6},\n",
            "  \"search_simulations\": {},\n",
            "  \"search_evaluations\": {},\n",
            "  \"actor_games\": {},\n",
            "  \"completed_games\": {},\n",
            "  \"actor_plies\": {},\n",
            "  \"actor_wall_seconds\": {:.6},\n",
            "  \"broker_requests\": {},\n",
            "  \"broker_batches\": {},\n",
            "  \"maximum_batch\": {},\n",
            "  \"failures\": {},\n",
            "  \"queue_milliseconds\": {:.3},\n",
            "  \"inference_milliseconds\": {:.3}\n",
            "}}"
        ),
        workers,
        first.0,
        first.1,
        first.2,
        search.stats.simulations,
        search.stats.evaluations,
        workers,
        completed_games,
        actor_plies,
        actor_wall_seconds,
        metrics.requests,
        metrics.batches,
        metrics.maximum_batch,
        metrics.failures,
        metrics.queue_nanoseconds as f64 / 1_000_000.0,
        metrics.inference_nanoseconds as f64 / 1_000_000.0,
    );
    Ok(())
}
