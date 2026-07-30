#![forbid(unsafe_code)]

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{ArgAction, Parser};
use sha2::{Digest, Sha256};
use vgo_core::{Color, Outcome, Position};
use vgo_inference::{
    BatchedEvaluator, BatchedEvaluatorPool, BrokerConfig, BrokerMetrics, OnnxBatchService,
    OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{CHANNELS, RasterConfig, RasterKind, SemanticRaster, action_pixel};
use vgo_search::{
    Action, EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, SearchResult, search_at_ply,
};
use vgo_selfplay::{
    ResignRule, play_game_with_resignation as run_playout_with_resignation,
};

mod replay_stream;

use replay_stream::{
    LabeledSample, PublishedReplay, REPLAY_VERSION, ReplayStream, sync_parent_directory,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationRuntime {
    Naive,
    Onnx,
}

impl GenerationRuntime {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Onnx => "onnx",
        }
    }
}

impl std::str::FromStr for GenerationRuntime {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "naive" => Ok(Self::Naive),
            "onnx" => Ok(Self::Onnx),
            _ => Err(format!("unsupported generation runtime: {value}")),
        }
    }
}

struct PendingSample {
    position: Position,
    root_black_value: f64,
    policy: Vec<f32>,
    policy_mask: Vec<f32>,
    visits: Vec<f32>,
    beta: Vec<f32>,
    proposal_counts: Vec<u32>,
    to_move: Color,
    selected_action: u32,
    game: u64,
    ply: u32,
    seed: u64,
}

struct GameSamples {
    samples: Vec<LabeledSample>,
    completed: bool,
    /// Counterfactual resignation outcomes for this game, one entry per
    /// candidate threshold. Only produced for games exempt from resignation,
    /// which are the only ones whose true result is known independently of the
    /// rule being measured.
    calibration: Vec<ResignTrial>,
}

#[derive(Clone, Debug, Parser)]
#[command(about = "Generate a labeled semantic-raster dataset from self-play")]
struct Config {
    #[arg(long, default_value_t = 96)]
    samples: usize,
    #[arg(long, default_value_t = 96)]
    resolution: usize,
    /// Channel layout written into the shard. `semantic` is the ten engineered
    /// channels; `rgb` is the board as a player sees it -- stone discs over
    /// Voronoi fill, three channels, no derived fields. Games are unaffected:
    /// the same seed and model produce the same play either way, so two runs
    /// differing only here are paired data over identical positions.
    #[arg(long, default_value = "semantic")]
    raster_kind: RasterKind,
    /// Placement grid the policy head emits, independent of the render
    /// resolution. The board is only ~9 stones across, so 128x128 of placement
    /// precision mostly splits single moves across many cells while spreading
    /// the fixed proposal budget too thin to ever revisit one. Rendering stays
    /// at `--resolution` so the Voronoi boundary channels keep their detail.
    #[arg(long, default_value_t = 32)]
    policy_resolution: usize,
    #[arg(long, default_value_t = 256)]
    simulations: u32,
    /// Leaves evaluated together per simulation round. Above one, a single game
    /// keeps that many evaluations in flight instead of one, which fills
    /// inference batches without needing more concurrent games.
    #[arg(long, default_value_t = 1)]
    leaf_batch: usize,
    /// Fine cells per coarse sampling region; zero uses legacy candidates.
    #[arg(long, default_value_t = 0)]
    coarse_pool: usize,
    /// Softmax temperature on root visit counts for the opening plies. Zero is
    /// deterministic argmax, which makes every game from a given position
    /// identical; a positive value is what gives self-play its diversity.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Plies over which `--temperature` applies; selection is argmax afterwards.
    #[arg(long, default_value_t = 30)]
    temperature_plies: u32,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    /// Concede once the side to move has been losing by at least this much for
    /// `--resign-window` consecutive plies. Zero disables resignation.
    ///
    /// Do not set this from intuition. A value head that is confidently wrong
    /// concedes won games, and the samples then carry the wrong label; measure
    /// the false-positive rate on games exempted by --resign-disable-fraction
    /// and pick a threshold that keeps it acceptable.
    #[arg(long, default_value_t = 0.0)]
    resign_threshold: f64,
    /// Consecutive plies that must all agree before conceding. One ply's root
    /// value is noisy; requiring a run of them is what keeps noise from ending
    /// games.
    #[arg(long, default_value_t = 5)]
    resign_window: u32,
    /// Fraction of games played to a real finish regardless of the threshold.
    /// These are the only games that can measure how often resignation would
    /// have been wrong, so a run that resigns should always keep some.
    #[arg(long, default_value_t = 0.1)]
    resign_disable_fraction: f64,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
    #[arg(long, default_value_t = 50_001)]
    seed: u64,
    #[arg(long, default_value_t = 4)]
    examples: usize,
    #[arg(long, default_value = "artifacts/raster-demo")]
    output: PathBuf,
    #[arg(long, default_value = "naive")]
    runtime: GenerationRuntime,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value_t = 8)]
    maximum_batch: usize,
    /// Independent inference brokers and native execution sessions. Multiple
    /// lanes let host packing, device transfers, and execution overlap without
    /// adding another per-position tensor copy.
    #[arg(long, default_value_t = 2)]
    inference_slots: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value_t = 8)]
    actors: usize,
    /// Maximum completed-game payloads waiting for the replay writer. Actors
    /// backpressure here instead of allowing a multi-gigabyte in-memory shard.
    #[arg(long, default_value_t = 2)]
    writer_queue_games: usize,
    #[arg(long)]
    maximum_games: Option<u64>,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

struct PolicyTarget {
    /// Normalized visit distribution over cells (the legacy target).
    policy: Vec<f32>,
    /// 1.0 for any cell that received a candidate, else 0.0.
    mask: Vec<f32>,
    /// Raw visit counts per cell (unnormalized), for off-policy reweighting.
    visits: Vec<f32>,
    /// Coarse->fine sampling probability beta per cell; 0.0 for legacy/pass
    /// candidates (which have no factored sampling probability).
    beta: Vec<f32>,
    /// Number of raw coarse->fine proposal draws landing in each cell. Legacy
    /// candidates and pass have zero multiplicity.
    proposal_counts: Vec<u32>,
}

/// Thresholds the calibrator evaluates each shard.
///
/// Spanning the range where a value head this saturated might plausibly be
/// trusted. The pipeline picks the lowest whose measured error rate is
/// acceptable -- lower concedes earlier and saves more.
const CALIBRATION_THRESHOLDS: [f64; 6] = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98];

/// What resignation would have done to one exempt game at one threshold.
#[derive(Clone, Copy, Debug)]
struct ResignTrial {
    threshold: f64,
    /// The rule would have conceded this game.
    fired: bool,
    /// It would have conceded for the side that actually won: a false positive,
    /// and the label the loop would have learned is the wrong one.
    wrong: bool,
    /// Plies that would have been skipped had it fired.
    plies_saved: u32,
}

/// Replays the resign rule over a finished game at each candidate threshold.
///
/// Only meaningful for games played to a real result, which is why this runs on
/// the exempt set: the rule's error rate is how often it would have conceded
/// for the eventual winner, and that is unknowable for a game the rule already
/// ended.
fn calibration_trials(
    pending: &[PendingSample],
    window: u32,
    black_won: bool,
) -> Vec<ResignTrial> {
    CALIBRATION_THRESHOLDS
        .iter()
        .map(|&threshold| {
            let mut streak = 0_u32;
            let mut fired_at = None;
            for (index, sample) in pending.iter().enumerate() {
                let mover_value = if sample.to_move == Color::Black {
                    sample.root_black_value
                } else {
                    -sample.root_black_value
                };
                if mover_value <= -threshold {
                    streak += 1;
                } else {
                    streak = 0;
                }
                if streak >= window {
                    fired_at = Some((index, sample.to_move));
                    break;
                }
            }
            match fired_at {
                None => ResignTrial {
                    threshold,
                    fired: false,
                    wrong: false,
                    plies_saved: 0,
                },
                Some((index, conceding)) => {
                    let conceding_won =
                        (conceding == Color::Black) == black_won;
                    ResignTrial {
                        threshold,
                        fired: true,
                        wrong: conceding_won,
                        plies_saved: (pending.len() - index - 1) as u32,
                    }
                }
            }
        })
        .collect()
}

/// Whether a game is exempt from resignation, decided from its seed.
///
/// Hashing the seed rather than counting games keeps the exempt set stable
/// across reruns and independent of actor scheduling, which matters because
/// these games are the calibration sample: they have to be a fair draw, not
/// whichever games happened to land on a particular worker.
fn resign_exempt(game_seed: u64, fraction: f64) -> bool {
    if fraction <= 0.0 {
        return false;
    }
    if fraction >= 1.0 {
        return true;
    }
    // SplitMix64 finalizer: cheap, and well distributed for consecutive seeds.
    let mut value = game_seed ^ 0x9e37_79b9_7f4a_7c15;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / (1_u64 << 53) as f64 <= fraction
}

/// Closing plies that must agree before a truncated game is adjudicated.
///
/// Long enough that a transient swing cannot decide the game, short enough to
/// stay inside the endgame the cap interrupted.
const ADJUDICATION_PLIES: usize = 12;

/// How far from even every one of those plies must be.
///
/// The value head saturates on about a third of positions and agrees with the
/// outcome only ~76% of the time, so a small margin would adjudicate on noise.
/// This is deliberately strict: a game that does not clear it stays discarded,
/// which is the behaviour it replaces.
const ADJUDICATION_MARGIN: f64 = 0.5;

/// Awards a truncated game when both seats agree on the leader.
///
/// Returns `None` -- leaving the game discarded -- unless every one of the last
/// `plies` root evaluations lies on the same side of zero by at least `margin`.
/// Because the plies alternate between the two players' searches, that
/// unanimity means the side that is behind agrees it is behind.
fn adjudicate(pending: &[PendingSample], plies: usize, margin: f64) -> Option<Outcome> {
    if pending.len() < plies {
        return None;
    }
    let closing = &pending[pending.len() - plies..];
    let leader_is_black = closing[0].root_black_value > 0.0;
    let agreed = closing.iter().all(|sample| {
        let value = sample.root_black_value;
        value.abs() >= margin && (value > 0.0) == leader_is_black
    });
    if !agreed {
        return None;
    }
    // Both seats must be represented, or this is one player's opinion repeated.
    let mut seen_black = false;
    let mut seen_white = false;
    for sample in closing {
        match sample.to_move {
            Color::Black => seen_black = true,
            Color::White => seen_white = true,
        }
    }
    if !(seen_black && seen_white) {
        return None;
    }
    Some(Outcome {
        winner: Some(if leader_is_black {
            Color::Black
        } else {
            Color::White
        }),
        // No area was counted, so there is no margin to report.
        margin: 0.0,
    })
}

fn policy_target(result: &SearchResult, config: RasterConfig) -> PolicyTarget {
    let size = config.pixels() + 1;
    let mut policy = vec![0.0_f32; size];
    let mut mask = vec![0.0_f32; size];
    let mut visits = vec![0.0_f32; size];
    let mut beta = vec![0.0_f32; size];
    let mut proposal_counts = vec![0_u32; size];
    let total = result
        .children
        .iter()
        .map(|child| child.visits)
        .sum::<u32>();
    for child in &result.children {
        let index = match child.action {
            Action::Pass => config.pixels(),
            Action::Place(point) => action_pixel(point.x, point.y, config),
        };
        policy[index] += child.visits as f32 / total as f32;
        mask[index] = 1.0;
        visits[index] += child.visits as f32;
        if let Some(b) = child.beta {
            beta[index] = b as f32;
        }
        proposal_counts[index] = proposal_counts[index]
            .checked_add(child.proposal_count)
            .expect("proposal multiplicity must fit in u32");
    }
    PolicyTarget {
        policy,
        mask,
        visits,
        beta,
        proposal_counts,
    }
}

fn action_index(action: Action, config: RasterConfig) -> u32 {
    match action {
        Action::Pass => config.pixels() as u32,
        Action::Place(point) => action_pixel(point.x, point.y, config) as u32,
    }
}

fn search_config(
    simulations: u32,
    coarse_pool: usize,
    temperature: f64,
    temperature_plies: u32,
    leaf_batch: usize,
) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config.temperature = temperature;
    config.temperature_plies = temperature_plies;
    config.leaf_batch = leaf_batch.max(1);
    config
}

fn validate_config(config: &Config) -> Result<(), &'static str> {
    if config.samples == 0
        || config.resolution == 0
        || config.simulations == 0
        || config.maximum_plies == 0
        || config.maximum_batch == 0
        || config.actors == 0
        || config.writer_queue_games == 0
        || config.maximum_games.is_some_and(|games| games == 0)
    {
        return Err("generation counts, simulations, and dimensions must be positive");
    }
    if u32::try_from(config.samples).is_err() {
        return Err("--samples exceeds the replay-v3 header capacity");
    }
    if config.policy_resolution == 0 {
        return Err("--policy-resolution must be positive");
    }
    if config.inference_slots == 0 {
        return Err("--inference-slots must be positive");
    }
    if config.device_id < 0 {
        return Err("--device-id must be nonnegative");
    }
    if config.policy_resolution > config.resolution {
        return Err("--policy-resolution must not exceed --resolution");
    }
    // The pool counts fine cells per coarse region on the policy grid, which is
    // what the sampler actually walks -- not the render raster.
    if config.coarse_pool > config.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution");
    }
    if !config.radius.is_finite() || config.radius <= 0.0 || config.radius >= 0.5 {
        return Err("--radius must be finite and between zero and one half");
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err("--temperature must be finite and not negative");
    }
    Ok(())
}

fn generate_game(
    config: &Config,
    evaluator: &dyn Evaluator,
    game_index: u64,
    stopped: &AtomicBool,
) -> Result<GameSamples, EvaluationError> {
    // Policy targets, the recorded action index, and the replay policy vector all
    // live on the placement grid, which may be coarser than the render raster.
    let policy_config = RasterConfig::square(config.policy_resolution);
    let search_config = search_config(
        config.simulations,
        config.coarse_pool,
        config.temperature,
        config.temperature_plies,
        config.leaf_batch,
    );
    let game_seed = config.seed.wrapping_add(game_index);
    let mut pending = Vec::new();
    // Exempt a deterministic fraction of games from resignation. Deriving the
    // choice from the game seed rather than a counter keeps it reproducible and
    // independent of how games are distributed across actors, so a rerun
    // exempts exactly the same games.
    let exempt_from_resignation = resign_exempt(game_seed, config.resign_disable_fraction);
    let resign = if config.resign_threshold > 0.0 && !exempt_from_resignation {
        ResignRule {
            threshold: config.resign_threshold,
            window: config.resign_window,
            disable_fraction: config.resign_disable_fraction,
        }
    } else {
        ResignRule::disabled()
    };
    let playout = run_playout_with_resignation(
        Position::new(config.radius, Vec::new(), Color::Black),
        config.maximum_plies,
        resign,
        |position, ply| {
            if stopped.load(Ordering::Acquire) {
                return Err(EvaluationError::new("replay generation cancelled"));
            }
            search_at_ply(position, search_config, game_seed, evaluator, ply)
        },
        |step| {
            let target = policy_target(step.search, policy_config);
            pending.push(PendingSample {
                // Store the position; rendering is a training-time choice now,
                // which also takes the rasterizer off the self-play hot path.
                position: step.position.clone(),
                policy: target.policy,
                policy_mask: target.mask,
                visits: target.visits,
                beta: target.beta,
                proposal_counts: target.proposal_counts,
                to_move: step.position.to_move(),
                // Root evaluation from Black's perspective, kept so a game cut
                // off at the ply cap can be adjudicated rather than discarded.
                root_black_value: step.search.root_black_value(),
                selected_action: action_index(step.action, policy_config),
                game: game_index,
                ply: step.ply,
                seed: game_seed,
            });
        },
    )?;
    // A game that ran out of plies has no played-out result, but it is not
    // necessarily undecided: if both sides' own searches have agreed on who is
    // ahead over the closing plies, the position is settled in the only sense
    // the loop can observe, and discarding it throws away a real label.
    //
    // Agreement is required from *both* seats. The root value is Black-relative
    // and each ply is evaluated by the side to move, so alternating plies are
    // independent judgements; demanding the same sign from all of them means
    // the player who is behind concurs. A margin keeps near-even positions --
    // where the sides would disagree by noise alone -- discarded as before.
    let outcome = match playout.outcome {
        Some(outcome) => outcome,
        None => match adjudicate(&pending, ADJUDICATION_PLIES, ADJUDICATION_MARGIN) {
            Some(outcome) => outcome,
            None => {
                return Ok(GameSamples {
                    samples: Vec::new(),
                    completed: false,
                    calibration: Vec::new(),
                });
            }
        },
    };
    let black_value = outcome.black_utility() as f32;
    // Measured before `pending` is consumed below. Only exempt games can
    // calibrate: a game the rule already ended cannot say whether ending it was
    // right, and a resigned game's outcome was assigned by the rule under test.
    let calibration = if exempt_from_resignation && !playout.resigned {
        calibration_trials(&pending, config.resign_window, black_value > 0.0)
    } else {
        Vec::new()
    };
    let samples = pending
        .into_iter()
        .map(|sample| LabeledSample {
            position: sample.position,
            policy: sample.policy,
            policy_mask: sample.policy_mask,
            visits: sample.visits,
            beta: sample.beta,
            proposal_counts: sample.proposal_counts,
            value: if sample.to_move == Color::Black {
                black_value
            } else {
                -black_value
            },
            selected_action: sample.selected_action,
            game: sample.game,
            ply: sample.ply,
            seed: sample.seed,
        })
        .collect();
    Ok(GameSamples {
        samples,
        completed: true,
        calibration,
    })
}

#[derive(Default)]
struct AtomicGenerationMetrics {
    games_started: AtomicU64,
    games_finished: AtomicU64,
    completed_games: AtomicU64,
    discarded_games: AtomicU64,
    failed_games: AtomicU64,
    samples_generated: AtomicU64,
    active_games: AtomicUsize,
    peak_active_games: AtomicUsize,
    writer_backlog: AtomicUsize,
    peak_writer_backlog: AtomicUsize,
    summed_game_nanoseconds: AtomicU64,
}

struct GameEnvelope {
    result: Result<GameSamples, EvaluationError>,
    ready_at: Instant,
}

/// Owns every replay actor together with the receiving side of their bounded
/// result queue.
///
/// Native inference runtimes must never outlive `main` on detached threads:
/// TensorRT installs process-exit handlers which are unsafe to run concurrently
/// with an in-flight `Session::Run`. Closing the receiver before joining also
/// wakes an actor which is already blocked while sending into a full queue.
struct ActorPool {
    stopped: Arc<AtomicBool>,
    receiver: Option<mpsc::Receiver<GameEnvelope>>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl ActorPool {
    fn new(stopped: Arc<AtomicBool>, receiver: mpsc::Receiver<GameEnvelope>) -> Self {
        Self {
            stopped,
            receiver: Some(receiver),
            handles: Vec::new(),
        }
    }

    fn push(&mut self, handle: thread::JoinHandle<()>) {
        self.handles.push(handle);
    }

    fn recv(&self) -> Result<GameEnvelope, mpsc::RecvError> {
        self.receiver
            .as_ref()
            .expect("actor receiver exists until shutdown")
            .recv()
    }

    fn shutdown(&mut self) -> std::io::Result<()> {
        self.stopped.store(true, Ordering::Release);
        // This must precede the joins: a producer may already be blocked in
        // SyncSender::send after the collector reached its exact sample count.
        self.receiver.take();

        let mut actor_panicked = false;
        for handle in std::mem::take(&mut self.handles) {
            actor_panicked |= handle.join().is_err();
        }
        if actor_panicked {
            Err(std::io::Error::other(
                "one or more replay actors panicked during shutdown",
            ))
        } else {
            Ok(())
        }
    }
}

impl Drop for ActorPool {
    fn drop(&mut self) {
        // Every return path, including replay I/O and evaluator failures, must
        // quiesce actors before the shared native inference session is dropped.
        let _ = self.shutdown();
    }
}

struct GenerationReport {
    replay: PublishedReplay,
    completed_games: usize,
    discarded_games: usize,
    /// Per-threshold resignation counterfactuals over this shard's exempt
    /// games: (threshold, games measured, would have fired, would have been
    /// wrong, plies saved). The pipeline reads these to pick the next shard's
    /// threshold, so it tracks the value head as it changes rather than being
    /// fixed once.
    calibration: Vec<(f64, u32, u32, u32, u64)>,
    samples_generated_by_received_games: usize,
    serialization_truncated_samples: usize,
    games_started: u64,
    games_finished: u64,
    generated_completed_games: u64,
    generated_discarded_games: u64,
    failed_games: u64,
    generated_samples: u64,
    peak_active_games: usize,
    peak_writer_backlog: usize,
    tail_games_in_flight: usize,
    tail_writer_backlog: usize,
    tail_completed_samples: u64,
    writer_backpressure: Duration,
    summed_game_time: Duration,
    wall_time: Duration,
}

fn atomic_duration(nanoseconds: &AtomicU64) -> Duration {
    Duration::from_nanos(nanoseconds.load(Ordering::Relaxed))
}

fn duration_nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn generate_to_dataset(
    config: &Config,
    evaluator: Arc<dyn Evaluator>,
    dataset_path: &Path,
) -> std::io::Result<GenerationReport> {
    let started = Instant::now();
    let maximum_games = config
        .maximum_games
        .unwrap_or_else(|| (config.samples as u64).saturating_mul(8));
    let raster = RasterConfig::square_of(config.resolution, config.raster_kind);
    let policy_size = config.policy_resolution * config.policy_resolution + 1;
    let mut replay = ReplayStream::create(
        dataset_path,
        config.samples,
        raster,
        policy_size,
        config.examples,
    )?;
    let next_game = Arc::new(AtomicU64::new(0));
    let stopped = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(AtomicGenerationMetrics::default());
    let (sender, receiver) = mpsc::sync_channel(config.writer_queue_games);
    let mut actors = ActorPool::new(Arc::clone(&stopped), receiver);
    actors.handles.reserve(config.actors);
    for actor in 0..config.actors {
        let config = config.clone();
        let evaluator = Arc::clone(&evaluator);
        let next_game = Arc::clone(&next_game);
        let stopped = Arc::clone(&stopped);
        let metrics = Arc::clone(&metrics);
        let sender = sender.clone();
        actors.push(
            thread::Builder::new()
                .name(format!("vgo-replay-actor-{actor:03}"))
                .spawn(move || {
                    while !stopped.load(Ordering::Acquire) {
                        let index = next_game.fetch_add(1, Ordering::Relaxed);
                        if index >= maximum_games {
                            break;
                        }
                        metrics.games_started.fetch_add(1, Ordering::Relaxed);
                        let active = metrics.active_games.fetch_add(1, Ordering::Relaxed) + 1;
                        metrics
                            .peak_active_games
                            .fetch_max(active, Ordering::Relaxed);
                        let game_started = Instant::now();
                        let result = generate_game(&config, evaluator.as_ref(), index, &stopped);
                        metrics.summed_game_nanoseconds.fetch_add(
                            duration_nanoseconds(game_started.elapsed()),
                            Ordering::Relaxed,
                        );
                        metrics.active_games.fetch_sub(1, Ordering::Relaxed);
                        metrics.games_finished.fetch_add(1, Ordering::Relaxed);
                        match &result {
                            Ok(game) if game.completed => {
                                metrics.completed_games.fetch_add(1, Ordering::Relaxed);
                                metrics
                                    .samples_generated
                                    .fetch_add(game.samples.len() as u64, Ordering::Relaxed);
                            }
                            Ok(_) => {
                                metrics.discarded_games.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(_) => {
                                metrics.failed_games.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // The collector has already reached the exact record
                        // boundary. Do not enqueue a just-finished tail game.
                        if stopped.load(Ordering::Acquire) {
                            break;
                        }
                        let backlog = metrics.writer_backlog.fetch_add(1, Ordering::Relaxed) + 1;
                        metrics
                            .peak_writer_backlog
                            .fetch_max(backlog, Ordering::Relaxed);
                        let sent = sender.send(GameEnvelope {
                            result,
                            ready_at: Instant::now(),
                        });
                        if sent.is_err() {
                            metrics.writer_backlog.fetch_sub(1, Ordering::Relaxed);
                            break;
                        }
                    }
                })?,
        );
    }
    drop(sender);

    let mut completed_games = 0_usize;
    let mut discarded_games = 0_usize;
    // threshold bits -> (games measured, would have fired, would have been
    // wrong, plies that would have been saved)
    let mut calibration_totals: std::collections::BTreeMap<u64, (u32, u32, u32, u64)> =
        std::collections::BTreeMap::new();
    let mut samples_generated_by_received_games = 0_usize;
    let mut serialization_truncated_samples = 0_usize;
    let mut writer_backpressure = Duration::ZERO;
    while !replay.is_full() {
        let envelope = match actors.recv() {
            Ok(envelope) => {
                metrics.writer_backlog.fetch_sub(1, Ordering::Relaxed);
                envelope
            }
            Err(_) => {
                actors.shutdown()?;
                return Err(std::io::Error::other(format!(
                    "replay exhausted {maximum_games} game attempts after {completed_games} completed and {discarded_games} discarded games"
                )));
            }
        };
        writer_backpressure += envelope.ready_at.elapsed();
        let game = envelope.result.map_err(|error| {
            stopped.store(true, Ordering::Release);
            std::io::Error::other(error)
        })?;
        for trial in &game.calibration {
            let entry = calibration_totals
                .entry(trial.threshold.to_bits())
                .or_insert((0_u32, 0_u32, 0_u32, 0_u64));
            entry.0 += 1;
            entry.1 += u32::from(trial.fired);
            entry.2 += u32::from(trial.wrong);
            entry.3 += u64::from(trial.plies_saved);
        }
        if game.completed {
            completed_games += 1;
            samples_generated_by_received_games =
                samples_generated_by_received_games.saturating_add(game.samples.len());
            let written = replay.write_game(game.samples)?;
            serialization_truncated_samples =
                serialization_truncated_samples.saturating_add(written.samples_truncated);
        } else {
            discarded_games += 1;
        }
    }

    // Preserve boundary telemetry before cancellation. Actors check the flag
    // before each ply, so shutdown waits for at most their current search—not a
    // maximum-length game.
    let tail_games_in_flight = metrics.active_games.load(Ordering::Relaxed);
    let tail_writer_backlog = metrics.writer_backlog.load(Ordering::Relaxed);
    let generated_samples_at_boundary = metrics.samples_generated.load(Ordering::Relaxed);
    let tail_completed_samples =
        generated_samples_at_boundary.saturating_sub(samples_generated_by_received_games as u64);
    let games_started = metrics.games_started.load(Ordering::Relaxed);
    let games_finished = metrics.games_finished.load(Ordering::Relaxed);
    let generated_completed_games = metrics.completed_games.load(Ordering::Relaxed);
    let generated_discarded_games = metrics.discarded_games.load(Ordering::Relaxed);
    let failed_games = metrics.failed_games.load(Ordering::Relaxed);
    let peak_active_games = metrics.peak_active_games.load(Ordering::Relaxed);
    let peak_writer_backlog = metrics.peak_writer_backlog.load(Ordering::Relaxed);
    let summed_game_time = atomic_duration(&metrics.summed_game_nanoseconds);
    actors.shutdown()?;
    let published = replay.publish()?;
    Ok(GenerationReport {
        replay: published,
        completed_games,
        discarded_games,
        calibration: calibration_totals
            .into_iter()
            .map(|(bits, (measured, fired, wrong, saved))| {
                (f64::from_bits(bits), measured, fired, wrong, saved)
            })
            .collect(),
        samples_generated_by_received_games,
        serialization_truncated_samples,
        games_started,
        games_finished,
        generated_completed_games,
        generated_discarded_games,
        failed_games,
        generated_samples: generated_samples_at_boundary,
        peak_active_games,
        peak_writer_backlog,
        tail_games_in_flight,
        tail_writer_backlog,
        tail_completed_samples,
        writer_backpressure,
        summed_game_time,
        wall_time: started.elapsed(),
    })
}

fn write_manifest(
    path: &Path,
    config: &Config,
    report: &GenerationReport,
    model_sha256: Option<&str>,
    broker: BrokerMetrics,
    inference_lanes: &[BrokerMetrics],
) -> std::io::Result<()> {
    let temporary = path.with_extension("json.tmp");
    if path.exists() || temporary.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("replay manifest already exists: {}", path.display()),
        ));
    }
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema\": \"vgo.replay-shard.v1\",")?;
    writeln!(writer, "  \"replay_version\": {REPLAY_VERSION},")?;
    writeln!(writer, "  \"dataset\": \"dataset.vgo\",")?;
    writeln!(
        writer,
        "  \"dataset_sha256\": \"{}\",",
        report.replay.sha256
    )?;
    writeln!(
        writer,
        "  \"shard_id\": \"sha256:{}\",",
        report.replay.sha256
    )?;
    writeln!(writer, "  \"dataset_bytes\": {},", report.replay.bytes)?;
    writeln!(writer, "  \"samples\": {},", report.replay.samples)?;
    writeln!(writer, "  \"completed_games\": {},", report.completed_games)?;
    writeln!(writer, "  \"discarded_games\": {},", report.discarded_games)?;
    writeln!(writer, "  \"resign_calibration\": [")?;
    for (index, (threshold, measured, fired, wrong, saved)) in
        report.calibration.iter().enumerate()
    {
        let comma = if index + 1 == report.calibration.len() { "" } else { "," };
        writeln!(
            writer,
            "    {{\"threshold\": {threshold}, \"games\": {measured}, \"fired\": {fired}, \"wrong\": {wrong}, \"plies_saved\": {saved}}}{comma}"
        )?;
    }
    writeln!(writer, "  ],")?;

    match report.replay.first_game_id {
        Some(game) => writeln!(writer, "  \"first_serialized_game_id\": {game},")?,
        None => writeln!(writer, "  \"first_serialized_game_id\": null,")?,
    }
    match report.replay.last_game_id {
        Some(game) => writeln!(writer, "  \"last_serialized_game_id\": {game},")?,
        None => writeln!(writer, "  \"last_serialized_game_id\": null,")?,
    }
    writeln!(writer, "  \"channels\": {},", config.raster_kind.channels())?;
    writeln!(writer, "  \"height\": {},", config.resolution)?;
    writeln!(writer, "  \"width\": {},", config.resolution)?;
    writeln!(
        writer,
        "  \"policy_size\": {},",
        config.policy_resolution * config.policy_resolution + 1
    )?;
    writeln!(writer, "  \"simulations\": {},", config.simulations)?;
    writeln!(writer, "  \"coarse_pool\": {},", config.coarse_pool)?;
    writeln!(writer, "  \"temperature\": {},", config.temperature)?;
    writeln!(
        writer,
        "  \"temperature_plies\": {},",
        config.temperature_plies
    )?;
    writeln!(writer, "  \"radius\": {},", config.radius)?;
    writeln!(writer, "  \"seed\": {},", config.seed)?;
    writeln!(writer, "  \"maximum_plies\": {},", config.maximum_plies)?;
    writeln!(writer, "  \"actors\": {},", config.actors)?;
    writeln!(writer, "  \"leaf_batch\": {},", config.leaf_batch)?;
    writeln!(
        writer,
        "  \"writer_queue_games\": {},",
        config.writer_queue_games
    )?;
    writeln!(writer, "  \"evaluator\": \"{}\",", config.runtime.as_str())?;
    writeln!(writer, "  \"provider\": \"{}\",", config.provider.as_str())?;
    writeln!(writer, "  \"device_id\": {},", config.device_id)?;
    writeln!(writer, "  \"fp16\": {},", config.fp16)?;
    writeln!(writer, "  \"maximum_batch\": {},", config.maximum_batch)?;
    writeln!(
        writer,
        "  \"configured_inference_slots\": {},",
        config.inference_slots
    )?;
    writeln!(
        writer,
        "  \"active_inference_slots\": {},",
        inference_lanes.len()
    )?;
    writeln!(writer, "  \"delay_ms\": {},", config.delay_ms)?;
    match model_sha256 {
        Some(digest) => writeln!(writer, "  \"model_sha256\": \"{digest}\",")?,
        None => writeln!(writer, "  \"model_sha256\": null,")?,
    }
    match model_sha256 {
        Some(digest) => writeln!(writer, "  \"behavior_model_sha256\": \"{digest}\",")?,
        None => writeln!(writer, "  \"behavior_model_sha256\": null,")?,
    }
    writeln!(
        writer,
        "  \"game_identity\": \"record game id plus record seed; behavior model is pinned for the shard\","
    )?;
    writeln!(writer, "  \"generation_metrics\": {{")?;
    writeln!(
        writer,
        "    \"samples_generated_by_received_games\": {},",
        report.samples_generated_by_received_games
    )?;
    writeln!(
        writer,
        "    \"serialization_truncated_samples\": {},",
        report.serialization_truncated_samples
    )?;
    writeln!(writer, "    \"games_started\": {},", report.games_started)?;
    writeln!(writer, "    \"games_finished\": {},", report.games_finished)?;
    writeln!(
        writer,
        "    \"generated_completed_games\": {},",
        report.generated_completed_games
    )?;
    writeln!(
        writer,
        "    \"generated_discarded_games\": {},",
        report.generated_discarded_games
    )?;
    writeln!(writer, "    \"failed_games\": {},", report.failed_games)?;
    writeln!(
        writer,
        "    \"generated_samples_at_boundary\": {},",
        report.generated_samples
    )?;
    writeln!(
        writer,
        "    \"peak_active_games\": {},",
        report.peak_active_games
    )?;
    writeln!(
        writer,
        "    \"peak_writer_backlog\": {},",
        report.peak_writer_backlog
    )?;
    writeln!(
        writer,
        "    \"tail_games_in_flight\": {},",
        report.tail_games_in_flight
    )?;
    writeln!(
        writer,
        "    \"tail_writer_backlog\": {},",
        report.tail_writer_backlog
    )?;
    writeln!(
        writer,
        "    \"tail_completed_samples\": {},",
        report.tail_completed_samples
    )?;
    writeln!(
        writer,
        "    \"wall_seconds\": {:.6},",
        report.wall_time.as_secs_f64()
    )?;
    writeln!(
        writer,
        "    \"summed_game_seconds\": {:.6},",
        report.summed_game_time.as_secs_f64()
    )?;
    writeln!(
        writer,
        "    \"writer_seconds\": {:.6},",
        report.replay.write_time.as_secs_f64()
    )?;
    writeln!(
        writer,
        "    \"writer_backpressure_seconds\": {:.6},",
        report.writer_backpressure.as_secs_f64()
    )?;
    writeln!(
        writer,
        "    \"publish_sync_seconds\": {:.6}",
        report.replay.sync_time.as_secs_f64()
    )?;
    writeln!(writer, "  }},")?;
    writeln!(writer, "  \"broker_metrics\": {{")?;
    writeln!(writer, "    \"requests\": {},", broker.requests)?;
    writeln!(writer, "    \"batches\": {},", broker.batches)?;
    writeln!(writer, "    \"positions\": {},", broker.positions)?;
    writeln!(
        writer,
        "    \"maximum_observed_batch\": {},",
        broker.maximum_batch
    )?;
    writeln!(writer, "    \"failures\": {},", broker.failures)?;
    writeln!(
        writer,
        "    \"encoding_seconds\": {:.6},",
        broker.encoding_nanoseconds as f64 / 1e9
    )?;
    writeln!(
        writer,
        "    \"queue_seconds\": {:.6},",
        broker.queue_nanoseconds as f64 / 1e9
    )?;
    writeln!(
        writer,
        "    \"inference_seconds\": {:.6}",
        broker.inference_nanoseconds as f64 / 1e9
    )?;
    writeln!(writer, "  }},")?;
    writeln!(writer, "  \"inference_lane_metrics\": [")?;
    for (slot, lane) in inference_lanes.iter().enumerate() {
        let comma = if slot + 1 == inference_lanes.len() {
            ""
        } else {
            ","
        };
        writeln!(
            writer,
            "    {{\"slot\": {slot}, \"requests\": {}, \"batches\": {}, \"positions\": {}, \"maximum_observed_batch\": {}, \"failures\": {}, \"encoding_seconds\": {:.6}, \"queue_seconds\": {:.6}, \"inference_seconds\": {:.6}}}{comma}",
            lane.requests,
            lane.batches,
            lane.positions,
            lane.maximum_batch,
            lane.failures,
            lane.encoding_nanoseconds as f64 / 1e9,
            lane.queue_nanoseconds as f64 / 1e9,
            lane.inference_nanoseconds as f64 / 1e9,
        )?;
    }
    writeln!(writer, "  ],")?;
    writeln!(
        writer,
        "  \"orientation\": \"row 0 samples y near 0; column 0 samples x near 0\","
    )?;
    writeln!(writer, "  \"perspective\": \"current player\",")?;
    writeln!(
        writer,
        "  \"policy_target\": \"MCTS root visits aggregated by pixel; pass is last\","
    )?;
    writeln!(
        writer,
        "  \"policy_mask\": \"sampled candidate pixels and pass; training derives the full legal denominator from legal_clearance\","
    )?;
    writeln!(
        writer,
        "  \"raw_visits\": \"f32 MCTS root visits aggregated by policy cell\","
    )?;
    writeln!(
        writer,
        "  \"sampling_beta\": \"f32 exact per-draw coarse-to-fine proposal probability for sampled placements; zero for pass and legacy candidates\","
    )?;
    writeln!(
        writer,
        "  \"proposal_counts\": \"u32 raw coarse-to-fine proposal multiplicity aggregated by policy cell; zero for pass and legacy candidates\","
    )?;
    writeln!(
        writer,
        "  \"value_target\": \"terminal utility in [-1, 1] for current player\","
    )?;
    writeln!(writer, "  \"channel_names\": [")?;
    let names: Vec<&str> = match config.raster_kind {
        RasterKind::Semantic => CHANNELS.iter().map(|channel| channel.name).collect(),
        RasterKind::Rgb => vec!["red", "green", "blue"],
    };
    for (index, name) in names.iter().enumerate() {
        let comma = if index + 1 == names.len() { "" } else { "," };
        writeln!(writer, "    \"{name}\"{comma}")?;
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;
    writer.flush()?;
    writer.into_inner()?.sync_all()?;
    fs::rename(temporary, path)?;
    sync_parent_directory(path)
}

fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_bmp(
    path: &Path,
    width: usize,
    height: usize,
    rgb: &[u8],
    scale: usize,
) -> std::io::Result<()> {
    let output_width = width * scale;
    let output_height = height * scale;
    let row_bytes = output_width * 3;
    let row_stride = (row_bytes + 3) & !3;
    let pixel_bytes = row_stride * output_height;
    let file_bytes = 54 + pixel_bytes;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"BM")?;
    writer.write_all(&(file_bytes as u32).to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(&54_u32.to_le_bytes())?;
    writer.write_all(&40_u32.to_le_bytes())?;
    writer.write_all(&(output_width as i32).to_le_bytes())?;
    writer.write_all(&(output_height as i32).to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&24_u16.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&(pixel_bytes as u32).to_le_bytes())?;
    writer.write_all(&2_835_i32.to_le_bytes())?;
    writer.write_all(&2_835_i32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    let padding = vec![0_u8; row_stride - row_bytes];
    for row in (0..height).rev() {
        for _ in 0..scale {
            for column in 0..width {
                let start = (row * width + column) * 3;
                for _ in 0..scale {
                    writer.write_all(&[rgb[start + 2], rgb[start + 1], rgb[start]])?;
                }
            }
            writer.write_all(&padding)?;
        }
    }
    writer.flush()
}

fn write_examples(
    directory: &Path,
    rasters: &[SemanticRaster],
    count: usize,
) -> std::io::Result<()> {
    fs::create_dir_all(directory)?;
    for (sample_index, raster) in rasters.iter().take(count).enumerate() {
        let config = raster.config();
        write_bmp(
            &directory.join(format!("sample-{sample_index:03}-overview.bmp")),
            config.width,
            config.height,
            &raster.overview_rgb(),
            6,
        )?;
        for (channel_index, channel) in CHANNELS.iter().enumerate() {
            write_bmp(
                &directory.join(format!(
                    "sample-{sample_index:03}-{channel_index:02}-{}.bmp",
                    channel.name
                )),
                config.width,
                config.height,
                &raster.channel_rgb(channel_index),
                6,
            )?;
        }
    }
    Ok(())
}

fn write_json_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    writer.write_all(b"\"")?;
    for character in value.chars() {
        match character {
            '"' => writer.write_all(b"\\\"")?,
            '\\' => writer.write_all(b"\\\\")?,
            '\u{08}' => writer.write_all(b"\\b")?,
            '\u{0c}' => writer.write_all(b"\\f")?,
            '\n' => writer.write_all(b"\\n")?,
            '\r' => writer.write_all(b"\\r")?,
            '\t' => writer.write_all(b"\\t")?,
            control if control <= '\u{1f}' => write!(writer, "\\u{:04x}", control as u32)?,
            ordinary => write!(writer, "{ordinary}")?,
        }
    }
    writer.write_all(b"\"")
}

fn main() -> std::io::Result<()> {
    let config = Config::parse();
    validate_config(&config)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    fs::create_dir_all(&config.output)?;

    // The behaviour model's input layout is fixed at export and is independent
    // of what the shard records. Search only needs the model to guide MCTS; the
    // targets it produces -- visit counts, beta, proposal multiplicities, and
    // the terminal value -- are functions of board state, not of how the board
    // was drawn for the network. So a semantic model can generate a shard in
    // any layout, which is what makes two --raster-kind runs paired data over
    // identical positions rather than two different games.
    let raster = RasterConfig::square(config.resolution);
    let model_path = config.model.as_deref();
    let (evaluator, broker): (Arc<dyn Evaluator>, Option<BatchedEvaluatorPool>) =
        match config.runtime {
            GenerationRuntime::Naive => (Arc::new(NaiveEvaluator), None),
            GenerationRuntime::Onnx => {
                let model = model_path.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "--model is required for ONNX generation",
                    )
                })?;
                let mut lanes = Vec::with_capacity(config.inference_slots);
                // Load native sessions sequentially. TensorRT can reuse the
                // engine cache without racing two simultaneous cache builders.
                for _slot in 0..config.inference_slots {
                    let service = OnnxBatchService::load(&OnnxServiceConfig {
                        policy: Some(RasterConfig::square(config.policy_resolution)),
                        model: model.to_path_buf(),
                        raster,
                        maximum_batch: config.maximum_batch,
                        provider: config.provider,
                        device_id: config.device_id,
                        fp16: config.fp16,
                        cache_directory: config.cache_directory.clone(),
                    })
                    .map_err(std::io::Error::other)?;
                    lanes.push(
                        BatchedEvaluator::spawn(
                            BrokerConfig {
                                maximum_delay: Duration::from_millis(config.delay_ms),
                                queue_capacity: (config.actors * 4).max(config.maximum_batch * 2),
                            },
                            service,
                        )
                        .map_err(std::io::Error::other)?,
                    );
                }
                let pool = BatchedEvaluatorPool::new(lanes).map_err(std::io::Error::other)?;
                (Arc::new(pool.clone()), Some(pool))
            }
        };
    let model_sha256 = if config.runtime == GenerationRuntime::Onnx {
        model_path.map(file_sha256).transpose()?
    } else {
        None
    };
    let dataset_path = config.output.join("dataset.vgo");
    let report = generate_to_dataset(&config, evaluator, &dataset_path)?;
    let broker_metrics = broker
        .as_ref()
        .map(BatchedEvaluatorPool::metrics)
        .unwrap_or_default();
    let inference_lane_metrics = broker
        .as_ref()
        .map(BatchedEvaluatorPool::lane_metrics)
        .unwrap_or_default();
    // All actors have joined and the moved evaluator was dropped when
    // generate_to_dataset returned. Destroy the final broker owner—and therefore
    // the ORT/TensorRT session—while main is still in ordinary Rust control
    // flow, before libc begins running native process-exit handlers.
    drop(broker);
    write_examples(
        &config.output.join("images"),
        &report.replay.examples,
        config.examples,
    )?;
    write_manifest(
        &config.output.join("manifest.json"),
        &config,
        &report,
        model_sha256.as_deref(),
        broker_metrics,
        &inference_lane_metrics,
    )?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    write!(output, "{{\n  \"dataset\": ")?;
    write_json_string(&mut output, &dataset_path.to_string_lossy())?;
    writeln!(output, ",\n  \"samples\": {},", report.replay.samples)?;
    writeln!(output, "  \"completed_games\": {},", report.completed_games)?;
    writeln!(output, "  \"discarded_games\": {},", report.discarded_games)?;
    writeln!(
        output,
        "  \"dataset_sha256\": \"{}\",",
        report.replay.sha256
    )?;
    writeln!(
        output,
        "  \"shard_id\": \"sha256:{}\",",
        report.replay.sha256
    )?;
    writeln!(output, "  \"evaluator\": \"{}\",", config.runtime.as_str())?;
    writeln!(output, "  \"actors\": {},", config.actors)?;
    writeln!(
        output,
        "  \"configured_inference_slots\": {},",
        config.inference_slots
    )?;
    writeln!(
        output,
        "  \"active_inference_slots\": {},",
        inference_lane_metrics.len()
    )?;
    writeln!(
        output,
        "  \"writer_queue_games\": {},",
        config.writer_queue_games
    )?;
    writeln!(output, "  \"coarse_pool\": {},", config.coarse_pool)?;
    writeln!(output, "  \"temperature\": {},", config.temperature)?;
    writeln!(
        output,
        "  \"temperature_plies\": {},",
        config.temperature_plies
    )?;
    writeln!(output, "  \"channels\": {},", config.raster_kind.channels())?;
    writeln!(output, "  \"resolution\": {},", config.resolution)?;
    writeln!(
        output,
        "  \"policy_size\": {},",
        config.policy_resolution * config.policy_resolution + 1
    )?;
    writeln!(output, "  \"examples\": {},", report.replay.examples.len())?;
    writeln!(output, "  \"generation_metrics\": {{")?;
    writeln!(
        output,
        "    \"samples_generated_by_received_games\": {},",
        report.samples_generated_by_received_games
    )?;
    writeln!(
        output,
        "    \"serialization_truncated_samples\": {},",
        report.serialization_truncated_samples
    )?;
    writeln!(output, "    \"games_started\": {},", report.games_started)?;
    writeln!(output, "    \"games_finished\": {},", report.games_finished)?;
    writeln!(
        output,
        "    \"generated_completed_games\": {},",
        report.generated_completed_games
    )?;
    writeln!(
        output,
        "    \"generated_discarded_games\": {},",
        report.generated_discarded_games
    )?;
    writeln!(output, "    \"failed_games\": {},", report.failed_games)?;
    writeln!(
        output,
        "    \"generated_samples_at_boundary\": {},",
        report.generated_samples
    )?;
    writeln!(
        output,
        "    \"peak_active_games\": {},",
        report.peak_active_games
    )?;
    writeln!(
        output,
        "    \"peak_writer_backlog\": {},",
        report.peak_writer_backlog
    )?;
    writeln!(
        output,
        "    \"tail_games_in_flight\": {},",
        report.tail_games_in_flight
    )?;
    writeln!(
        output,
        "    \"tail_writer_backlog\": {},",
        report.tail_writer_backlog
    )?;
    writeln!(
        output,
        "    \"tail_completed_samples\": {},",
        report.tail_completed_samples
    )?;
    writeln!(
        output,
        "    \"wall_seconds\": {:.6},",
        report.wall_time.as_secs_f64()
    )?;
    writeln!(
        output,
        "    \"summed_game_seconds\": {:.6},",
        report.summed_game_time.as_secs_f64()
    )?;
    writeln!(
        output,
        "    \"writer_seconds\": {:.6},",
        report.replay.write_time.as_secs_f64()
    )?;
    writeln!(
        output,
        "    \"writer_backpressure_seconds\": {:.6},",
        report.writer_backpressure.as_secs_f64()
    )?;
    writeln!(
        output,
        "    \"publish_sync_seconds\": {:.6}",
        report.replay.sync_time.as_secs_f64()
    )?;
    writeln!(output, "  }},")?;
    writeln!(output, "  \"broker_metrics\": {{")?;
    writeln!(output, "    \"requests\": {},", broker_metrics.requests)?;
    writeln!(output, "    \"batches\": {},", broker_metrics.batches)?;
    writeln!(output, "    \"positions\": {},", broker_metrics.positions)?;
    writeln!(
        output,
        "    \"maximum_observed_batch\": {},",
        broker_metrics.maximum_batch
    )?;
    writeln!(
        output,
        "    \"encoding_seconds\": {:.6},",
        broker_metrics.encoding_nanoseconds as f64 / 1e9
    )?;
    writeln!(
        output,
        "    \"queue_seconds\": {:.6},",
        broker_metrics.queue_nanoseconds as f64 / 1e9
    )?;
    writeln!(
        output,
        "    \"inference_seconds\": {:.6},",
        broker_metrics.inference_nanoseconds as f64 / 1e9
    )?;
    writeln!(
        output,
        "    \"failures\": {}\n  }},",
        broker_metrics.failures
    )?;
    writeln!(output, "  \"inference_lane_metrics\": [")?;
    for (slot, lane) in inference_lane_metrics.iter().enumerate() {
        let comma = if slot + 1 == inference_lane_metrics.len() {
            ""
        } else {
            ","
        };
        writeln!(
            output,
            "    {{\"slot\": {slot}, \"requests\": {}, \"batches\": {}, \"positions\": {}, \"maximum_observed_batch\": {}, \"failures\": {}, \"encoding_seconds\": {:.6}, \"queue_seconds\": {:.6}, \"inference_seconds\": {:.6}}}{comma}",
            lane.requests,
            lane.batches,
            lane.positions,
            lane.maximum_batch,
            lane.failures,
            lane.encoding_nanoseconds as f64 / 1e9,
            lane.queue_nanoseconds as f64 / 1e9,
            lane.inference_nanoseconds as f64 / 1e9,
        )?;
    }
    writeln!(output, "  ]\n}}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    use clap::Parser;
    use vgo_core::Point;
    use vgo_raster::RasterConfig;
    use vgo_search::{Action, CandidateSource, ChildSummary, SearchResult, SearchStats};

    use super::{
        ActorPool, Config, GameEnvelope, GameSamples, PendingSample, adjudicate,
        calibration_trials, policy_target, resign_exempt, search_config, validate_config,
    };
    use vgo_core::{Color, Position};

    #[test]
    fn actor_pool_drop_cancels_and_joins_workers() {
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = Arc::clone(&stopped);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let (_sender, receiver) = mpsc::sync_channel(1);
        let mut actors = ActorPool::new(stopped, receiver);
        actors.push(thread::spawn(move || {
            while !worker_stopped.load(Ordering::Acquire) {
                thread::yield_now();
            }
            worker_finished.store(true, Ordering::Release);
        }));

        drop(actors);

        assert!(
            finished.load(Ordering::Acquire),
            "dropping the pool must not detach a live actor"
        );
    }

    #[test]
    fn actor_pool_closes_a_full_queue_before_joining() {
        let stopped = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(GameEnvelope {
                result: Ok(GameSamples {
                    samples: Vec::new(),
                    completed: false,
                    calibration: Vec::new(),
                }),
                ready_at: Instant::now(),
            })
            .expect("fill the bounded queue");
        let (sending, sending_receiver) = mpsc::channel();
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let handle = thread::spawn(move || {
            sending.send(()).expect("announce the blocked send");
            let _ = sender.send(GameEnvelope {
                result: Ok(GameSamples {
                    samples: Vec::new(),
                    completed: false,
                    calibration: Vec::new(),
                }),
                ready_at: Instant::now(),
            });
            worker_finished.store(true, Ordering::Release);
        });
        sending_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("worker reached its send");

        let mut actors = ActorPool::new(stopped, receiver);
        actors.push(handle);
        actors.shutdown().expect("actor shutdown succeeds");

        assert!(
            finished.load(Ordering::Acquire),
            "closing the receiver must wake a producer blocked on the full queue"
        );
    }

    #[test]
    fn malformed_cli_values_are_rejected() {
        assert!(Config::try_parse_from(["vgo-generate-demo", "--resolution", "large"]).is_err());
    }

    #[test]
    fn coarse_pool_cli_defaults_to_legacy_and_accepts_an_override() {
        let default = Config::try_parse_from(["vgo-generate-demo"]).expect("default CLI parses");
        assert_eq!(default.coarse_pool, 0);

        let configured = Config::try_parse_from(["vgo-generate-demo", "--coarse-pool", "8"])
            .expect("coarse sampling options parse");
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn coarse_sampling_is_applied_to_search_config() {
        let configured = search_config(37, 8, 1.0, 30, 1);
        assert_eq!(configured.simulations, 37);
        assert_eq!(configured.coarse_pool, 8);
        assert_eq!(configured.temperature, 1.0);
        assert_eq!(configured.temperature_plies, 30);
    }

    /// Generation defaults to sampled opening moves. A zero default here would
    /// silently reproduce the deterministic self-play this change exists to fix.
    #[test]
    fn generation_defaults_to_a_positive_opening_temperature() {
        let default = Config::try_parse_from(["vgo-generate-demo"]).expect("default CLI parses");
        assert!(default.temperature > 0.0);
        assert!(default.temperature_plies > 0);
    }

    #[test]
    fn generation_defaults_to_two_inference_lanes() {
        let default = Config::try_parse_from(["vgo-generate-demo"]).expect("default CLI parses");
        assert_eq!(default.inference_slots, 2);

        let configured = Config::try_parse_from(["vgo-generate-demo", "--inference-slots", "4"])
            .expect("inference lane override parses");
        assert_eq!(configured.inference_slots, 4);
    }

    #[test]
    fn inference_lane_count_must_be_positive() {
        let configured = Config::try_parse_from(["vgo-generate-demo", "--inference-slots", "0"])
            .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&configured),
            Err("--inference-slots must be positive")
        );
    }

    #[test]
    fn negative_temperature_is_rejected() {
        let configured =
            Config::try_parse_from(["vgo-generate-demo", "--temperature=-1"]).expect("CLI parses");
        assert!(validate_config(&configured).is_err());
    }

    #[test]
    fn invalid_coarse_sampling_config_is_rejected_before_generation() {
        let oversized_pool = Config::try_parse_from([
            "vgo-generate-demo",
            "--policy-resolution",
            "16",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&oversized_pool),
            Err("--coarse-pool must not exceed --policy-resolution")
        );

        // A policy grid may be decoupled and coarser, but a finer policy cannot
        // be derived from the rendered state and is rejected by the loader.
        let finer_policy = Config::try_parse_from([
            "vgo-generate-demo",
            "--resolution",
            "16",
            "--policy-resolution",
            "32",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&finer_policy),
            Err("--policy-resolution must not exceed --resolution")
        );
    }

    #[test]
    fn zero_maximum_plies_is_rejected_before_generation() {
        let config = Config::try_parse_from(["vgo-generate-demo", "--max-plies", "0"])
            .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&config),
            Err("generation counts, simulations, and dimensions must be positive")
        );
    }

    #[test]
    fn completed_game_queue_must_be_bounded_and_positive() {
        let config = Config::try_parse_from(["vgo-generate-demo", "--writer-queue-games", "0"])
            .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&config),
            Err("generation counts, simulations, and dimensions must be positive")
        );
    }

    #[test]
    fn invalid_radius_is_rejected_before_generation() {
        for radius in ["0", "0.5", "NaN"] {
            let config = Config::try_parse_from(["vgo-generate-demo", "--radius", radius])
                .expect("CLI syntax parses");
            assert_eq!(
                validate_config(&config),
                Err("--radius must be finite and between zero and one half")
            );
        }
    }

    #[test]
    fn policy_target_aggregates_proposal_multiplicity_by_pixel() {
        let sampled = |point, visits, proposal_count| ChildSummary {
            action: Action::Place(point),
            source: CandidateSource::AreaSequence,
            prior: 0.25,
            visits,
            black_value: 0.0,
            beta: Some(0.125),
            proposal_count,
        };
        let result = SearchResult::from_children(
            Action::Pass,
            vec![
                sampled(Point::new(0.1, 0.1), 2, 2),
                sampled(Point::new(0.2, 0.2), 1, 3),
                ChildSummary {
                    action: Action::Pass,
                    source: CandidateSource::Pass,
                    prior: 0.5,
                    visits: 1,
                    black_value: 0.0,
                    beta: None,
                    proposal_count: 0,
                },
            ],
            SearchStats::default(),
            vgo_core::Color::Black,
        );

        let target = policy_target(&result, RasterConfig::square(2));

        assert_eq!(target.proposal_counts, vec![5, 0, 0, 0, 0]);
        assert_eq!(target.proposal_counts[4], 0, "pass is not proposed");
    }

    fn adjudication_sample(to_move: Color, root_black_value: f64) -> PendingSample {
        PendingSample {
            position: Position::new(1.0 / 18.0, Vec::new(), to_move),
            root_black_value,
            policy: Vec::new(),
            policy_mask: Vec::new(),
            visits: Vec::new(),
            beta: Vec::new(),
            proposal_counts: Vec::new(),
            to_move,
            selected_action: 0,
            game: 0,
            ply: 0,
            seed: 0,
        }
    }

    fn closing(values: &[f64]) -> Vec<PendingSample> {
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let to_move = if index % 2 == 0 { Color::Black } else { Color::White };
                adjudication_sample(to_move, value)
            })
            .collect()
    }

    #[test]
    fn adjudication_awards_a_game_both_sides_agree_on() {
        // Every closing ply well past the margin and on the same side: the
        // player who is behind is evaluating too, so this is agreement rather
        // than one seat's opinion.
        let pending = closing(&[0.9; 12]);
        let outcome = adjudicate(&pending, 12, 0.5).expect("agreed games are awarded");
        assert_eq!(outcome.winner, Some(Color::Black));
    }

    #[test]
    fn adjudication_declines_a_disputed_game() {
        // Sign flips within the window: the sides disagree, so the game stays
        // discarded rather than being awarded on one player's view.
        let mut values = [0.9; 12];
        values[7] = -0.9;
        assert!(adjudicate(&closing(&values), 12, 0.5).is_none());
    }

    #[test]
    fn adjudication_declines_a_close_game() {
        // Consistent leader but inside the margin. The value head is wrong
        // often enough that a small edge is not evidence, so this is the case
        // the margin exists to reject.
        assert!(adjudicate(&closing(&[0.2; 12]), 12, 0.5).is_none());
    }

    #[test]
    fn adjudication_needs_both_seats_in_the_window() {
        // A window covering only one player's turns is that player agreeing
        // with itself; the opponent never weighed in.
        let one_sided: Vec<PendingSample> = (0..12)
            .map(|_| adjudication_sample(Color::Black, 0.9))
            .collect();
        assert!(adjudicate(&one_sided, 12, 0.5).is_none());
    }

    #[test]
    fn adjudication_declines_a_game_shorter_than_the_window() {
        assert!(adjudicate(&closing(&[0.9; 4]), 12, 0.5).is_none());
    }

    #[test]
    fn resign_exemption_is_stable_and_close_to_its_fraction() {
        // The exempt set is the calibration sample, so it must be a fair draw
        // and identical across reruns.
        let seeds: Vec<u64> = (0..20_000).map(|i| 990_001_u64.wrapping_add(i)).collect();
        let exempt = seeds.iter().filter(|&&s| resign_exempt(s, 0.1)).count();
        let rate = exempt as f64 / seeds.len() as f64;
        assert!((rate - 0.1).abs() < 0.01, "exemption rate {rate} is off target");
        assert!(seeds.iter().all(|&s| resign_exempt(s, 0.1) == resign_exempt(s, 0.1)));
        assert!(seeds.iter().all(|&s| !resign_exempt(s, 0.0)));
        assert!(seeds.iter().all(|&s| resign_exempt(s, 1.0)));
    }

    #[test]
    fn calibration_marks_a_concession_by_the_eventual_winner_as_wrong() {
        // Black's search says Black is losing badly for the whole game, but
        // Black in fact wins. Resignation would have conceded for the winner,
        // which is exactly the error the calibration exists to count -- and the
        // reason a threshold cannot be chosen without measuring.
        // The root value is Black-relative and flips for the mover, so a
        // constant value can never produce a streak: whichever side is behind
        // sees -v while the other sees +v, and the run resets every ply. A real
        // losing streak is consecutive plies *by the same player*, which is what
        // the window counts. Using single-seat plies here isolates that.
        let pending: Vec<PendingSample> =
            (0..20).map(|_| adjudication_sample(Color::Black, -0.99)).collect();
        let trials = calibration_trials(&pending, 5, true);
        let strict = trials.iter().find(|t| t.threshold == 0.90).unwrap();
        assert!(strict.fired, "a sustained despairing evaluation should fire");
        assert!(strict.wrong, "conceding for the eventual winner is a false positive");
        assert!(strict.plies_saved > 0);
    }

    #[test]
    fn calibration_marks_a_correct_concession_as_right() {
        // Same evaluations, but Black really does lose: the rule would have
        // been correct, and the plies after it are pure waste.
        let pending: Vec<PendingSample> =
            (0..20).map(|_| adjudication_sample(Color::Black, -0.99)).collect();
        let trials = calibration_trials(&pending, 5, false);
        let strict = trials.iter().find(|t| t.threshold == 0.90).unwrap();
        assert!(strict.fired);
        assert!(!strict.wrong);
    }

    #[test]
    fn a_higher_threshold_never_fires_more_often() {
        // Monotonicity is what lets the pipeline pick the lowest acceptable
        // threshold: raising it can only make the rule more cautious.
        let pending: Vec<PendingSample> =
            (0..30).map(|_| adjudication_sample(Color::Black, -0.88)).collect();
        let trials = calibration_trials(&pending, 5, false);
        for pair in trials.windows(2) {
            assert!(
                !(pair[1].fired && !pair[0].fired),
                "threshold {} fired while the looser {} did not",
                pair[1].threshold,
                pair[0].threshold
            );
        }
    }
}
