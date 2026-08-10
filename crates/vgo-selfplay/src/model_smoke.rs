#![forbid(unsafe_code)]

use std::{
    mem::size_of,
    path::PathBuf,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::{ArgAction, Parser};
use vgo_core::{Color, Point, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
    PythonBatchService, PythonProcessConfig, TorchDevice,
};
use vgo_raster::RasterConfig;
use vgo_search::{Action, EvaluationError, Evaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::play_game as run_playout;

/// Path to the training venv's interpreter, relative to the repo root.
/// The layout differs by platform: `bin/` on Unix, `Scripts/` on Windows.
#[cfg(windows)]
const VENV_PYTHON: &str = "training/.venv/Scripts/python.exe";
#[cfg(not(windows))]
const VENV_PYTHON: &str = "training/.venv/bin/python3";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InferenceRuntime {
    Python,
    Onnx,
}

impl InferenceRuntime {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Onnx => "onnx",
        }
    }
}

impl std::str::FromStr for InferenceRuntime {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "python" => Ok(Self::Python),
            "onnx" => Ok(Self::Onnx),
            _ => Err(format!("unsupported inference runtime: {value}")),
        }
    }
}

#[derive(Debug, Parser)]
struct Arguments {
    // bin/ on Unix, Scripts/ on Windows -- see VENV_PYTHON.
    #[arg(long, default_value = VENV_PYTHON)]
    python: PathBuf,
    #[arg(long, default_value = "training")]
    training: PathBuf,
    #[arg(long, default_value = "artifacts/raster-demo/model.pt")]
    checkpoint: PathBuf,
    #[arg(long, default_value = "artifacts/raster-demo/model.onnx")]
    model: PathBuf,
    #[arg(long, default_value = "python")]
    runtime: InferenceRuntime,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value_t = 64)]
    actors: usize,
    #[arg(long)]
    games: Option<usize>,
    #[arg(long, default_value_t = 8)]
    simulations: u32,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 8)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value_t = 1)]
    torch_threads: usize,
    #[arg(long, default_value = "cuda")]
    device: TorchDevice,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    compile: bool,
    #[arg(long, default_value_t = 1)]
    probe_requests: usize,
    #[arg(long, default_value_t = 128)]
    resolution: usize,
    #[arg(long)]
    policy_resolution: Option<usize>,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
}

fn rooted(root: &std::path::Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn play_model_game(
    evaluator: &BatchedEvaluator,
    seed: u64,
    simulations: u32,
    maximum_plies: u32,
    radius: f64,
) -> Result<(bool, u32), EvaluationError> {
    let report = run_playout(
        Position::new(radius, Vec::new(), Color::Black),
        maximum_plies,
        |position, _| {
            search_with_evaluator(position, SearchConfig::canary(simulations), seed, evaluator)
        },
        |_| {},
    )?;
    Ok((report.completed(), report.stats.plies))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let root = std::env::current_dir()?;
    let python = rooted(&root, arguments.python);
    let working_directory = rooted(&root, arguments.training);
    let checkpoint = rooted(&root, arguments.checkpoint);
    let model = rooted(&root, arguments.model);
    let runtime = arguments.runtime;
    let provider = arguments.provider;
    let fp16 = arguments.fp16;
    let actors = arguments.actors;
    let games = arguments.games.unwrap_or_else(|| actors.saturating_mul(8));
    let simulations = arguments.simulations;
    let maximum_plies = arguments.maximum_plies;
    let maximum_batch = arguments.maximum_batch;
    let maximum_delay_ms = arguments.delay_ms;
    let torch_threads = arguments.torch_threads;
    let device = arguments.device;
    let compile = arguments.compile;
    let probe_requests = arguments.probe_requests;
    let resolution = arguments.resolution;
    let policy_resolution = arguments.policy_resolution.unwrap_or(resolution);
    let radius = arguments.radius;
    if actors == 0
        || games == 0
        || simulations == 0
        || maximum_plies == 0
        || maximum_batch == 0
        || torch_threads == 0
        || resolution == 0
        || policy_resolution == 0
        || !(0.0_f64..=0.5).contains(&radius)
        || radius == 0.0
    {
        return Err(
            "counts and raster/policy resolutions must be positive; radius must be in (0, 0.5]"
                .into(),
        );
    }

    let raster = RasterConfig::square(resolution);
    let policy = RasterConfig::square(policy_resolution);
    let broker = BrokerConfig {
        maximum_delay: Duration::from_millis(maximum_delay_ms),
        queue_capacity: (actors * 4).max(maximum_batch * 2),
    };
    let evaluator = match runtime {
        InferenceRuntime::Python => {
            let service = PythonBatchService::spawn(&PythonProcessConfig {
                python,
                working_directory,
                checkpoint,
                raster,
                policy: Some(policy),
                maximum_batch,
                torch_threads,
                device,
                compile,
            })?;
            BatchedEvaluator::spawn(broker, service)?
        }
        InferenceRuntime::Onnx => {
            let service = OnnxBatchService::load(&OnnxServiceConfig {
                model,
                raster,
                policy: Some(policy),
                maximum_batch,
                provider,
                device_id: 0,
                fp16,
                cache_directory: root.join("artifacts/onnx-cache"),
            })?;
            BatchedEvaluator::spawn(broker, service)?
        }
    };

    let probe_barrier = Arc::new(Barrier::new(probe_requests + 1));
    let mut probe_handles = Vec::with_capacity(probe_requests);
    for _ in 0..probe_requests {
        let evaluator = evaluator.clone();
        let barrier = Arc::clone(&probe_barrier);
        probe_handles.push(thread::spawn(move || {
            let position = Position::new(radius, Vec::new(), Color::Black);
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

    let position = Position::new(radius, Vec::new(), Color::Black);
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
                    radius,
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
    let actor_encoding_nanoseconds =
        metrics.encoding_nanoseconds - before_actors.encoding_nanoseconds;
    let actor_queue_nanoseconds = metrics.queue_nanoseconds - before_actors.queue_nanoseconds;
    let actor_inference_nanoseconds =
        metrics.inference_nanoseconds - before_actors.inference_nanoseconds;
    let python_torch_threads = match runtime {
        InferenceRuntime::Python => torch_threads.to_string(),
        InferenceRuntime::Onnx => String::from("null"),
    };
    let python_device = match runtime {
        InferenceRuntime::Python => format!("\"{}\"", device.as_str()),
        InferenceRuntime::Onnx => String::from("null"),
    };
    let python_compiled = match runtime {
        InferenceRuntime::Python => compile.to_string(),
        InferenceRuntime::Onnx => String::from("null"),
    };

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
            "  \"inference_runtime\": \"{}\",\n",
            "  \"inference_provider\": \"{}\",\n",
            "  \"inference_fp16\": {},\n",
            "  \"python_torch_threads\": {},\n",
            "  \"python_device\": {},\n",
            "  \"python_compiled\": {},\n",
            "  \"raster_resolution\": {},\n",
            "  \"policy_resolution\": {},\n",
            "  \"stone_radius\": {:.9},\n",
            "  \"input_bytes_per_position\": {},\n",
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
            "  \"encoding_milliseconds_total\": {:.3},\n",
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
        runtime.as_str(),
        match runtime {
            InferenceRuntime::Python => device.as_str(),
            InferenceRuntime::Onnx => provider.as_str(),
        },
        runtime == InferenceRuntime::Onnx && fp16,
        python_torch_threads,
        python_device,
        python_compiled,
        resolution,
        policy_resolution,
        radius,
        10 * resolution * resolution * size_of::<f32>(),
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
        actor_encoding_nanoseconds as f64 / 1_000_000.0,
        actor_queue_nanoseconds as f64 / 1_000_000.0,
        actor_inference_nanoseconds as f64 / 1_000_000.0,
    );
    Ok(())
}
