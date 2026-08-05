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
use vgo_core::{Analysis, Color, Position};
use vgo_inference::{
    BatchedEvaluator, BatchedEvaluatorPool, BrokerConfig, BrokerMetrics, OnnxBatchService,
    OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{
    CHANNELS, COMPACT_CHANNELS, ChannelSpec, RasterConfig, RasterKind, SemanticRaster,
    action_pixel,
};
use vgo_search::{
    Action, EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, SearchResult, search_at_ply,
};
use vgo_selfplay::{
    ResignRule, award_by_area,
    play_game_with_resignation as run_playout_with_resignation,
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

/// Buckets spanning the configured komi range. Eight over [-0.1, 0.2] gives
/// 0.0375-wide buckets and ~170 games each per sixteen-shard window.
const KOMI_BUCKETS: usize = 8;

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

/// One game's outcome, written to a sidecar rather than into the replay.
///
/// The replay record is a training tensor: position, policy, value, and nothing
/// that identifies which game it came from. Game-level facts -- the komi it was
/// played at, how it ended, by how much -- have no place there, and replicating
/// them across every position of a game would pay 72x for one number.
///
/// `first_sample` and `sample_count` are the join back: they name the record
/// range this game produced, which is what lets a later question condition
/// position-level data on a game-level outcome. They cannot be reconstructed
/// after the fact, so they are written even though nothing reads them yet.
#[derive(Clone, Debug)]
struct GameRecord {
    game: u64,
    komi: f64,
    plies: u32,
    /// Passes and no-op self-captures, which are how a stalled game passes the
    /// time. A capped game with many of either is stalling, not playing long.
    passes: u64,
    self_captures: u64,
    /// Black-relative: +1 Black, -1 White, 0 tie.
    black_utility: f32,
    /// Area margin, always non-negative.
    margin: f64,
    reached_ply_cap: bool,
    resigned: bool,
    /// Ply a soft concession fired at, if one did. The game played on from
    /// there at reduced search, so `resigned` stays false and the outcome is a
    /// real one -- this is what lets the rule be scored after the fact.
    soft_resign_ply: Option<u32>,
    first_sample: usize,
    sample_count: usize,
    /// The board as the game ended, for the ownership target.
    ///
    /// Stored as stones rather than as a rendered map: ownership at a point is
    /// the colour of the nearest stone, which is what the rasterizer already
    /// computes for the voronoi channels, so re-rendering from the position
    /// cannot drift from the inputs the net reads. It is also 26x smaller --
    /// ~600 bytes against a 16 KB int8 map at 128x128.
    final_stones: Vec<(f64, f64, u8)>,
}

struct GameSamples {
    samples: Vec<LabeledSample>,
    /// This game's outcome, absent only when it produced no samples.
    record: Option<GameRecord>,
    completed: bool,
    /// The game ran out of plies rather than ending on its own.
    ///
    /// Worth tracking on its own: a hundred plies is far past where the board
    /// settles, so a high share means the sides are stalling rather than that
    /// the game is genuinely long. It was 88% among games containing a run of
    /// no-op self-captures and 0% among games without one.
    reached_ply_cap: bool,
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
    /// Layout the behaviour model reads, which is fixed at its export and is
    /// independent of `--raster-kind`: the shard records one layout while the
    /// model may have been trained on another. Defaults to the shard's, which
    /// is right whenever a run generates with the model it is training.
    #[arg(long)]
    model_raster_kind: Option<RasterKind>,
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
    /// Finish in-flight games when the sample target is reached rather than
    /// cancelling them.
    ///
    /// Every actor has a game running when a shard fills, so cutting there
    /// discards roughly one partial game per actor. That is ~29% of the work at
    /// 16888 samples and 60-70% at 4000, because the tail is a fixed cost that
    /// does not shrink with the shard. Draining writes those games instead, so
    /// the shard overshoots its target by whatever the tail carried.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    drain_tail: bool,
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
    /// Earliest ply a game may be conceded at, regardless of the window.
    ///
    /// The window counts a seat's own turns and a seat moves every other ply,
    /// so a window of five permits resignation at ply 8 -- five stones each on
    /// a board that holds thirty-five. Measured on ddrnet-vs update 59 that is
    /// what happened: 17 of 28 games ended at ply 8, White conceded 26 of 28,
    /// and the same seed with resignation off ran every game to a hundred
    /// plies. A floor keeps the confidence test at five turns and forbids
    /// acting on it while the board is still an opening.
    #[arg(long, default_value_t = 0)]
    resign_minimum_ply: u32,
    /// Simulations for the remainder of a conceded game, or 0 to stop at the
    /// concession.
    ///
    /// A hard resignation asserts the winner and a false positive writes the
    /// wrong label. Playing on cheaply lets the game reach a real terminal
    /// state, so a mistaken concession corrects itself. Measured on this run's
    /// recent shards, a 0.7 threshold fires on 66% of games at a 4.8% error
    /// rate -- errors worth recovering rather than tolerating.
    #[arg(long, default_value_t = 0)]
    resign_soft_simulations: u32,
    /// Fraction of games played to a real finish regardless of the threshold.
    /// These are the only games that can measure how often resignation would
    /// have been wrong, so a run that resigns should always keep some.
    #[arg(long, default_value_t = 0.1)]
    resign_disable_fraction: f64,
    /// Lowest komi a game may be given. Komi is a fraction of the board, and
    /// positive favours White: scoring is `black - white - komi > 0`.
    #[arg(long, default_value_t = 0.0)]
    komi_low: f64,
    /// Highest komi a game may be given; each game draws uniformly from the
    /// range. A single value teaches one balance point, a range teaches the
    /// relationship, which is what lets one model play at any komi.
    #[arg(long, default_value_t = 0.0)]
    komi_high: f64,
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
/// Thresholds the counterfactual sweeps.
///
/// Extends past 0.98 because the head saturates: measured on ddrnet-komi3
/// update 34, 51% of positions read |v| > 0.95 and 30% read |v| > 0.99, so a
/// sweep stopping at 0.98 cannot tell a confident call from a certain one. The
/// tail matters -- 0.70 to 0.98 moved the false-positive rate only 9.0% to
/// 6.2% while still firing on 504 of ~660 games.
const CALIBRATION_THRESHOLDS: [f64; 9] =
    [0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999];

/// Windows swept alongside the thresholds.
///
/// The window is the other half of the rule and the untested half: raising the
/// threshold from 0.95 to 0.98 barely moved the false-positive rate, which says
/// the head is confidently wrong rather than marginally over the line. Demanding
/// more consecutive plies of agreement is a different lever -- it asks the
/// losing seat to keep conceding across more of its own turns, which noise
/// clears far less easily than one confident evaluation.
const CALIBRATION_WINDOWS: [u32; 4] = [5, 8, 12, 16];

/// What resignation would have done to one exempt game at one threshold.
#[derive(Clone, Copy, Debug)]
struct ResignTrial {
    threshold: f64,
    /// Consecutive own-turn agreements required before conceding.
    window: u32,
    /// The rule would have conceded this game.
    fired: bool,
    /// It would have conceded for the side that actually won: a false positive,
    /// and the label the loop would have learned is the wrong one.
    wrong: bool,
    /// Plies that would have been skipped had it fired.
    plies_saved: u32,
    /// The *least* confident evaluation in the window that triggered the
    /// concession, as a magnitude in [threshold, 1].
    ///
    /// This is the quantity the rule actually tests, and the question it
    /// answers is whether false positives are less extreme than true ones. If
    /// they are, confidence separates the two and a higher bar is worth
    /// raising. If the distributions overlap, no threshold can filter them and
    /// only a different signal -- a longer window, or not resigning -- will.
    /// The root values are discarded after a playout, so this cannot be
    /// recovered later; it has to be measured here.
    fired_confidence: f64,
}

/// Replays the resign rule over a finished game at each candidate threshold.
///
/// Only meaningful for games played to a real result, which is why this runs on
/// the exempt set: the rule's error rate is how often it would have conceded
/// for the eventual winner, and that is unknowable for a game the rule already
/// ended.
fn calibration_trials(
    pending: &[PendingSample],
    _window: u32,
    black_won: bool,
) -> Vec<ResignTrial> {
    CALIBRATION_THRESHOLDS
        .iter()
        .flat_map(|&threshold| {
            CALIBRATION_WINDOWS.iter().map(move |&window| (threshold, window))
        })
        .map(|(threshold, window)| {
            // Per seat, matching the live rule: a shared counter is reset by the
            // winning side's ply and can never reach the window.
            let mut streak = [0_u32; 2];
            // The weakest evaluation in the current streak, per seat: the rule
            // requires every ply in the window to clear the bar, so the streak
            // is only as confident as its least confident member.
            let mut weakest = [1.0_f64; 2];
            let mut fired_at = None;
            for (index, sample) in pending.iter().enumerate() {
                let mover_value = if sample.to_move == Color::Black {
                    sample.root_black_value
                } else {
                    -sample.root_black_value
                };
                let seat = usize::from(sample.to_move == Color::White);
                if mover_value <= -threshold {
                    streak[seat] += 1;
                    weakest[seat] = weakest[seat].min(mover_value.abs());
                } else {
                    streak[seat] = 0;
                    weakest[seat] = 1.0;
                }
                if streak[seat] >= window {
                    fired_at = Some((index, sample.to_move, weakest[seat]));
                    break;
                }
            }
            match fired_at {
                None => ResignTrial {
                    threshold,
                    window,
                    fired: false,
                    wrong: false,
                    plies_saved: 0,
                    fired_confidence: 0.0,
                },
                Some((index, conceding, confidence)) => {
                    let conceding_won =
                        (conceding == Color::Black) == black_won;
                    ResignTrial {
                        threshold,
                        window,
                        fired: true,
                        wrong: conceding_won,
                        plies_saved: (pending.len() - index - 1) as u32,
                        fired_confidence: confidence,
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

/// A uniform draw in `[0, 1)` from a game seed.
///
/// Shares `resign_exempt`'s SplitMix64 finalizer but with a different constant,
/// so a game's komi and its exemption are independent -- reusing the stream
/// would correlate the two and make every exempt game share a komi.
fn seeded_unit(game_seed: u64) -> f64 {
    let mut value = game_seed ^ 0x2545_f491_4f6c_dd1d;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / (1_u64 << 53) as f64
}

/// Komi for one game, drawn uniformly from `[low, high]`.
///
/// Varying it across games is what lets one model play at any komi: a net shown
/// a single value learns that value's balance rather than the relationship. It
/// also spreads the training distribution over positions that are close under
/// *some* komi, which a fixed value cannot do.
/// Komi for one game: a normal centred on the range, truncated to it.
///
/// Uniform spends as much mass at the ends of the range as at the middle, and
/// the ends are where komi decides the game rather than shading it. Measured on
/// ddrnet-resign64 with the range at [-0.037, 0.363]: Black took 77% of the
/// bottom bucket and 3.2% of the top, so a game drawn near either end is
/// settled before a stone is placed and teaches the value head only which end
/// it came from.
///
/// A normal concentrates games where the position is genuinely contested while
/// still reaching the ends often enough to teach the relationship. Sigma is a
/// quarter of the width, so the range spans +/-2 sigma and about 95% of draws
/// land inside it before truncation.
fn sampled_komi(game_seed: u64, low: f64, high: f64) -> f64 {
    if !(low.is_finite() && high.is_finite()) || high <= low {
        return low.max(0.0).min(high.max(0.0));
    }
    let centre = 0.5 * (low + high);
    let sigma = 0.25 * (high - low);
    // Box-Muller from two independent uniforms off the same seed. The second
    // stream uses a different constant so the pair is not degenerate.
    let u1 = seeded_unit(game_seed).max(f64::MIN_POSITIVE);
    let u2 = seeded_unit(game_seed ^ 0x51ed_2701_a3f5_9c7b);
    let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    (centre + sigma * normal).clamp(low, high)
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
            minimum_ply: config.resign_minimum_ply,
            soft_simulations: config.resign_soft_simulations,
        }
    } else {
        ResignRule::disabled()
    };
    // Set by the playout the ply a soft concession fires at, read by the search
    // closure below so the rest of the game runs cheaply. A Cell rather than a
    // parameter because the playout owns the loop and the closure is its
    // argument -- neither can see the other's locals.
    let soft_from = std::cell::Cell::new(u32::MAX);
    let playout = run_playout_with_resignation(
        Position::new(config.radius, Vec::new(), Color::Black)
            .with_komi(sampled_komi(game_seed, config.komi_low, config.komi_high)),
        config.maximum_plies,
        resign,
        |position, ply| {
            if stopped.load(Ordering::Acquire) {
                return Err(EvaluationError::new("replay generation cancelled"));
            }
            // Past a soft concession the game is only being played out to reach
            // a real terminal state, so it does not need full search.
            let mut config_for_ply = search_config;
            if ply >= soft_from.get() {
                config_for_ply.simulations = config.resign_soft_simulations;
            }
            search_at_ply(position, config_for_ply, game_seed, evaluator, ply)
        },
        |step| {
            if let Some(ply) = step.soft_resign_ply {
                soft_from.set(ply);
            }
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
        None => {
            let closing: Vec<_> = pending
                .iter()
                .map(|sample| sample.position.clone())
                .collect();
            // A game at the ply cap is decided on its final position rather
            // than discarded. A hundred plies is far past where the board
            // settles, and the margin was rejecting close games -- which are
            // the ones a balanced komi produces, so refusing them threw away
            // exactly the positions the run is trying to learn from.
            match closing.last() {
                Some(position) => award_by_area(position),
                None => {
                    return Ok(GameSamples {
                        samples: Vec::new(),
                        record: None,
                        completed: false,
                        reached_ply_cap: true,
                        calibration: Vec::new(),
                    });
                }
            }
        }
    };
    let black_value = outcome.black_utility() as f32;
    // Measured before `pending` is consumed below. Only a game the rule did
    // not end can calibrate it: a hard-resigned game's outcome was assigned by
    // the rule under test, so asking whether resigning was right is circular.
    //
    // A *soft* resignation is not circular, and reports `resigned: false` for
    // exactly that reason -- the game played on to a real terminal state, so
    // the outcome is independent of the rule that fired. Under soft resign
    // every game therefore calibrates, and `--resign-disable-fraction` is no
    // longer needed to hold out a clean sample: it exists to keep the rule from
    // deciding the label, which soft resign already prevents. That is a ten-fold
    // larger calibration sample at no cost, since the counterfactual only
    // replays root values every game already stores.
    //
    // The exemption still matters with hard resignation, where a fired game's
    // label really is the rule's own assertion.
    let calibration = if !playout.resigned {
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
    let samples: Vec<LabeledSample> = samples;
    Ok(GameSamples {
        record: Some(GameRecord {
            game: game_index,
            komi: playout.final_position.komi(),
            plies: playout.stats.plies,
            passes: playout.stats.passes,
            self_captures: playout.stats.self_captures,
            black_utility: black_value,
            // The board margin, not `outcome.margin`: a resignation reports a
            // real winner but leaves the margin at zero, since no area was
            // scored. Recomputing it from the final position is what makes a
            // resigned game reviewable -- it says how far behind the conceding
            // side actually was, which is the only way to judge the rule.
            margin: {
                let analysis = Analysis::new(&playout.final_position);
                (analysis.score.black
                    - analysis.score.white
                    - playout.final_position.komi())
                .abs()
            },
            reached_ply_cap: playout.outcome.is_none(),
            resigned: playout.resigned,
            soft_resign_ply: playout.stats.soft_resign_ply,
            // Filled by the writer, which owns the shard-relative offset.
            first_sample: 0,
            sample_count: samples.len(),
            final_stones: playout
                .final_position
                .stones()
                .iter()
                .map(|stone| {
                    (stone.x, stone.y, u8::from(stone.color == Color::White))
                })
                .collect(),
        }),
        samples,
        completed: true,
        reached_ply_cap: playout.outcome.is_none(),
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
    /// Games that ran out of plies rather than ending on their own.
    capped_games: usize,
    /// One row per game that reached the dataset, written to a sidecar.
    game_records: Vec<GameRecord>,
    replay: PublishedReplay,
    completed_games: usize,
    discarded_games: usize,
    /// Per-threshold resignation counterfactuals over this shard's exempt
    /// games: (threshold, games measured, would have fired, would have been
    /// wrong, plies saved). The pipeline reads these to pick the next shard's
    /// threshold, so it tracks the value head as it changes rather than being
    /// fixed once.
    calibration: Vec<(f64, u32, u32, u32, u32, u64, f64, f64)>,
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
    // Set when the shard reaches its target: actors stop taking new games but
    // finish the one they are playing. See the collector loop below.
    let draining = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(AtomicGenerationMetrics::default());
    let (sender, receiver) = mpsc::sync_channel(config.writer_queue_games);
    let mut actors = ActorPool::new(Arc::clone(&stopped), receiver);
    actors.handles.reserve(config.actors);
    for actor in 0..config.actors {
        let config = config.clone();
        let evaluator = Arc::clone(&evaluator);
        let next_game = Arc::clone(&next_game);
        let stopped = Arc::clone(&stopped);
        let draining = Arc::clone(&draining);
        let metrics = Arc::clone(&metrics);
        let sender = sender.clone();
        actors.push(
            thread::Builder::new()
                .name(format!("vgo-replay-actor-{actor:03}"))
                .spawn(move || {
                    while !stopped.load(Ordering::Acquire) {
                        // Two flags, deliberately. `draining` stops new games
                        // from starting; `stopped` cancels one already running.
                        // Checking draining here and stopped inside the playout
                        // is what lets a shard finish its tail instead of
                        // throwing away one partial game per actor.
                        if draining.load(Ordering::Acquire) {
                            break;
                        }
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
                        // A cancelled run drops its tail; a draining one keeps
                        // it, which is the whole point of draining.
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
    let mut capped_games = 0_usize;
    let mut game_records: Vec<GameRecord> = Vec::new();
    // (threshold bits, window) -> (games measured, would have fired, would have
    // been wrong, plies that would have been saved). Keyed on both, or the four
    // windows would sum into one another and report a single blended rate.
    // Trailing two sums split the triggering confidence by correctness, which
    // is what says whether a false positive is distinguishable from a true one.
    let mut calibration_totals: std::collections::BTreeMap<
        (u64, u32),
        (u32, u32, u32, u64, f64, f64),
    > = std::collections::BTreeMap::new();
    let mut samples_generated_by_received_games = 0_usize;
    let mut serialization_truncated_samples = 0_usize;
    let mut writer_backpressure = Duration::ZERO;
    // Phase one fills the shard; phase two drains whatever was already in
    // flight when it filled. `drain_tail` false keeps the old exact-count
    // behaviour for callers that need a fixed shard size.
    let mut draining_started = false;
    loop {
        if replay.is_full() && !draining_started {
            draining_started = true;
            if !config.drain_tail {
                break;
            }
            // Stop handing out new games, then keep collecting. Actors already
            // playing run to completion.
            draining.store(true, Ordering::Release);
            replay.allow_overshoot();
        }
        if draining_started && metrics.active_games.load(Ordering::Relaxed) == 0
            && metrics.writer_backlog.load(Ordering::Relaxed) == 0
        {
            break;
        }
        let envelope = match actors.recv() {
            Ok(envelope) => {
                metrics.writer_backlog.fetch_sub(1, Ordering::Relaxed);
                envelope
            }
            Err(_) if draining_started => break,
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
                .entry((trial.threshold.to_bits(), trial.window))
                .or_insert((0_u32, 0_u32, 0_u32, 0_u64, 0.0_f64, 0.0_f64));
            entry.0 += 1;
            entry.1 += u32::from(trial.fired);
            entry.2 += u32::from(trial.wrong);
            entry.3 += u64::from(trial.plies_saved);
            if trial.fired {
                if trial.wrong {
                    entry.5 += trial.fired_confidence;
                } else {
                    entry.4 += trial.fired_confidence;
                }
            }
        }
        if game.reached_ply_cap {
            capped_games += 1;
        }
        if game.completed {
            completed_games += 1;
            samples_generated_by_received_games =
                samples_generated_by_received_games.saturating_add(game.samples.len());
            let written = replay.write_game(game.samples)?;
            serialization_truncated_samples =
                serialization_truncated_samples.saturating_add(written.samples_truncated);
            // Only games that reached the dataset get a row, and the row
            // describes what was actually written: a game cut at the sample
            // boundary contributed fewer records than it played.
            if let Some(mut record) = game.record {
                record.first_sample = written.first_sample;
                record.sample_count = written.samples_written;
                game_records.push(record);
            }
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
        capped_games,
        game_records,
        calibration: calibration_totals
            .into_iter()
            .map(
                |((bits, window), (measured, fired, wrong, saved, right_sum, wrong_sum))| {
                    (
                        f64::from_bits(bits),
                        window,
                        measured,
                        fired,
                        wrong,
                        saved,
                        right_sum,
                        wrong_sum,
                    )
                },
            )
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

/// Writes one JSON object per game to `games.jsonl`.
///
/// JSONL rather than a field on the replay record: ~85 games against 6144
/// positions, so the file is a rounding error next to the dataset, and new
/// fields can be added later without a `replay_version` bump or a dtype change
/// on the training path.
fn write_game_records(path: &Path, records: &[GameRecord]) -> std::io::Result<()> {
    let temporary = path.with_extension("jsonl.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    for record in records {
        writeln!(
            writer,
            concat!(
                r#"{{"game":{},"komi":{:.6},"plies":{},"passes":{},"#,
                r#""self_captures":{},"black_utility":{},"margin":{:.6},"#,
                r#""reached_ply_cap":{},"resigned":{},"soft_resign_ply":{},"#,
                r#""first_sample":{},"sample_count":{},"final_stones":[{}]}}"#
            ),
            record.game,
            record.komi,
            record.plies,
            record.passes,
            record.self_captures,
            record.black_utility,
            record.margin,
            record.reached_ply_cap,
            record.resigned,
            record
                .soft_resign_ply
                .map_or_else(|| "null".to_owned(), |ply| ply.to_string()),
            record.first_sample,
            record.sample_count,
            record
                .final_stones
                .iter()
                .map(|(x, y, colour)| format!("[{x:.6},{y:.6},{colour}]"))
                .collect::<Vec<_>>()
                .join(","),
        )?;
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// Black's results bucketed over the configured komi range.
///
/// Fixed edges from the configuration rather than from the observed data, so
/// buckets line up across shards and can be summed over a replay window. That
/// pooling is not optional: a shard holds ~85 games, so a single shard's
/// buckets hold ~10 games each and the binomial interval on 5/10 spans roughly
/// [19%, 81%] -- noise. A window of sixteen shards is the smallest honest unit.
fn komi_buckets(records: &[GameRecord], low: f64, high: f64, count: usize) -> Vec<String> {
    let width = (high - low) / count as f64;
    (0..count)
        .map(|index| {
            let start = low + width * index as f64;
            let end = if index + 1 == count { high } else { start + width };
            let mut games = 0_usize;
            let mut black = 0_usize;
            let mut ties = 0_usize;
            let mut margins: Vec<f64> = Vec::new();
            for record in records {
                // Half-open, with the top bucket closed so `high` itself lands.
                let inside = record.komi >= start
                    && (record.komi < end || (index + 1 == count && record.komi <= end));
                if !inside {
                    continue;
                }
                games += 1;
                if record.black_utility > 0.0 {
                    black += 1;
                } else if record.black_utility == 0.0 {
                    ties += 1;
                }
                // Signed toward Black, so the median crosses zero at the same
                // komi the winrate crosses 50% -- a gap between those two
                // crossings is the model's seat asymmetry, not the game's.
                margins.push(if record.black_utility >= 0.0 {
                    record.margin
                } else {
                    -record.margin
                });
            }
            margins.sort_by(|a, b| a.partial_cmp(b).expect("finite margins"));
            let median = if margins.is_empty() {
                0.0
            } else if margins.len() % 2 == 1 {
                margins[margins.len() / 2]
            } else {
                (margins[margins.len() / 2 - 1] + margins[margins.len() / 2]) / 2.0
            };
            format!(
                concat!(
                    r#"{{"low": {:.4}, "high": {:.4}, "games": {}, "#,
                    r#""black_wins": {}, "ties": {}, "black_margin_median": {:.6}}}"#
                ),
                start, end, games, black, ties, median
            )
        })
        .collect()
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
    writeln!(writer, "  \"capped_games\": {},", report.capped_games)?;
    writeln!(writer, "  \"games\": \"games.jsonl\",")?;
    let buckets = komi_buckets(
        &report.game_records,
        config.komi_low,
        config.komi_high,
        KOMI_BUCKETS,
    );
    writeln!(writer, "  \"komi_calibration\": [")?;
    for (index, bucket) in buckets.iter().enumerate() {
        let comma = if index + 1 == buckets.len() { "" } else { "," };
        writeln!(writer, "    {bucket}{comma}")?;
    }
    writeln!(writer, "  ],")?;
    writeln!(writer, "  \"resign_calibration\": [")?;
    for (index, (threshold, window, measured, fired, wrong, saved, right_sum, wrong_sum)) in
        report.calibration.iter().enumerate()
    {
        let comma = if index + 1 == report.calibration.len() { "" } else { "," };
        // Mean confidence at the moment of firing, split by whether the
        // concession turned out right. Emitted as means rather than sums so a
        // reader does not have to divide, and null when nothing fired.
        let right_mean = if fired > wrong {
            format!("{:.4}", right_sum / f64::from(fired - wrong))
        } else {
            "null".to_owned()
        };
        let wrong_mean = if *wrong > 0 {
            format!("{:.4}", wrong_sum / f64::from(*wrong))
        } else {
            "null".to_owned()
        };
        writeln!(
            writer,
            "    {{\"threshold\": {threshold}, \"window\": {window}, \"games\": {measured}, \"fired\": {fired}, \"wrong\": {wrong}, \"plies_saved\": {saved}, \"confidence_right\": {right_mean}, \"confidence_wrong\": {wrong_mean}}}{comma}"
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
        RasterKind::Compact => vgo_raster::COMPACT_CHANNELS
            .iter()
            .map(|&channel| CHANNELS[channel].name)
            .collect(),
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
        // Previews are a diagnostic, and the channel table describes the
        // semantic layout. A shard written in another layout carries fewer
        // channels under different meanings, so only the channels this raster
        // actually holds can be dumped -- and the overview, which reads fixed
        // semantic indices, only applies to the layout it was written for.
        let channels = config.channels();
        if channels == CHANNELS.len() {
            write_bmp(
                &directory.join(format!("sample-{sample_index:03}-overview.bmp")),
                config.width,
                config.height,
                &raster.overview_rgb(),
                6,
            )?;
        }
        // A compact plane is not the semantic plane at the same index -- plane
        // 2 is voronoi_ridge, which the table lists at 6 -- so the name has to
        // come through the layout's own index map or the file would lie about
        // what it shows.
        let named: Vec<(usize, &ChannelSpec)> = if channels == CHANNELS.len() {
            CHANNELS.iter().enumerate().collect()
        } else {
            COMPACT_CHANNELS
                .iter()
                .take(channels)
                .enumerate()
                .map(|(plane, &semantic)| (plane, &CHANNELS[semantic]))
                .collect()
        };
        for (channel_index, channel) in named {
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
    let raster = RasterConfig::square_of(
        config.resolution,
        config.model_raster_kind.unwrap_or(config.raster_kind),
    );
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
    write_game_records(&config.output.join("games.jsonl"), &report.game_records)?;
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
    writeln!(output, "  \"capped_games\": {},", report.capped_games)?;
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
        ActorPool, Config, GameEnvelope, GameRecord, GameSamples, PendingSample,
        calibration_trials, komi_buckets, policy_target, resign_exempt, sampled_komi,
        search_config,
        validate_config,
    };
    use vgo_core::{Color, Position, Stone};

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

    fn game_at(komi: f64, black_wins: bool, margin: f64) -> GameRecord {
        GameRecord {
            game: 0,
            komi,
            plies: 60,
            passes: 0,
            self_captures: 0,
            black_utility: if black_wins { 1.0 } else { -1.0 },
            margin,
            reached_ply_cap: false,
            resigned: false,
            first_sample: 0,
            sample_count: 1,
            soft_resign_ply: None,
            final_stones: Vec::new(),
        }
    }

    /// Every game lands in exactly one bucket, including the range endpoints.
    ///
    /// The top bucket is closed and the rest are half-open, so `high` itself
    /// has a home; without that a game drawn at exactly the range maximum would
    /// vanish from the calibration while still being counted as played.
    /// Komi concentrates near the middle of its range and stays inside it.
    ///
    /// Uniform spent as much mass on the ends as the middle, and the ends are
    /// where komi decides the game outright: measured at range
    /// [-0.037, 0.363], Black took 77% of the bottom bucket and 3.2% of the
    /// top. Those games teach the value head which end they came from and
    /// little else.
    #[test]
    fn komi_is_drawn_from_a_truncated_normal() {
        let (low, high) = (-0.166, 0.234);
        let centre = 0.5 * (low + high);
        let draws: Vec<f64> = (0..20_000)
            .map(|index| sampled_komi(index as u64 * 2_654_435_761, low, high))
            .collect();

        assert!(
            draws.iter().all(|&k| (low..=high).contains(&k)),
            "every draw must stay inside the range"
        );
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!(
            (mean - centre).abs() < 0.01,
            "should centre on {centre}, got {mean}"
        );

        // Concentrated, not uniform: the middle fifth should hold well over
        // the 20% a uniform draw would put there.
        let width = high - low;
        let middle = draws
            .iter()
            .filter(|&&k| (k - centre).abs() < width * 0.1)
            .count();
        let share = middle as f64 / draws.len() as f64;
        assert!(
            share > 0.30,
            "middle fifth should hold far more than uniform's 20%, got {:.1}%",
            100.0 * share
        );
    }

    #[test]
    fn komi_buckets_place_every_game_including_the_endpoints() {
        let records = vec![
            game_at(-0.1, true, 0.5),  // exactly `low`
            game_at(0.2, false, 0.5),  // exactly `high`
            game_at(0.05, true, 0.5),  // interior
        ];
        let buckets = komi_buckets(&records, -0.1, 0.2, 4);
        let counted: usize = buckets
            .iter()
            .map(|bucket| {
                let marker = "\"games\": ";
                let start = bucket.find(marker).expect("games field") + marker.len();
                let rest = &bucket[start..];
                let end = rest.find(',').expect("field terminator");
                rest[..end].parse::<usize>().expect("integer count")
            })
            .sum();
        assert_eq!(counted, records.len(), "every game lands in exactly one bucket");
    }

    /// The median margin is signed toward Black.
    ///
    /// It has to be, for the margin curve to cross zero where the winrate
    /// crosses 50%. Taking the median of unsigned margins would report a
    /// positive lead for a bucket White dominates.
    #[test]
    fn komi_bucket_margin_is_signed_toward_black() {
        // One bucket, White winning both games by a clear margin.
        let records = vec![game_at(0.0, false, 0.30), game_at(0.05, false, 0.40)];
        let buckets = komi_buckets(&records, 0.0, 0.1, 1);
        let marker = "\"black_margin_median\": ";
        let start = buckets[0].find(marker).expect("median field") + marker.len();
        let rest = &buckets[0][start..];
        let end = rest.find('}').expect("object terminator");
        let median: f64 = rest[..end].trim().parse().expect("finite median");
        assert!(
            median < 0.0,
            "White winning both games must give Black a negative median, got {median}"
        );
        assert!((median + 0.35).abs() < 1.0e-9, "median of -0.30 and -0.40, got {median}");
    }

    #[test]
    fn actor_pool_closes_a_full_queue_before_joining() {
        let stopped = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(GameEnvelope {
                result: Ok(GameSamples {
                    samples: Vec::new(),
                    record: None,
                    completed: false,
                    reached_ply_cap: false,
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
                    record: None,
                    completed: false,
                    reached_ply_cap: false,
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

    /// `adjudicate_positions` over the samples these tests build.
    fn adjudicate_window(
        pending: &[PendingSample],
        plies: usize,
        margin: f64,
    ) -> Option<vgo_core::Outcome> {
        let positions: Vec<_> =
            pending.iter().map(|sample| sample.position.clone()).collect();
        vgo_selfplay::adjudicate_positions(&positions, plies, margin)
    }

    fn adjudication_sample(position: Position, root_black_value: f64) -> PendingSample {
        let to_move = position.to_move();
        PendingSample {
            position,
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

    /// A board where `black` stones sit on the left and `white` on the right, so
    /// the Voronoi split follows the stone counts.
    fn lopsided(black: usize, white: usize) -> Position {
        let radius = 1.0 / 18.0;
        let mut stones = Vec::new();
        for i in 0..black {
            let row = i / 3;
            let col = i % 3;
            stones.push(Stone::new(
                0.10 + col as f64 * 0.12,
                0.10 + row as f64 * 0.12,
                Color::Black,
            ));
        }
        for i in 0..white {
            let row = i / 2;
            let col = i % 2;
            stones.push(Stone::new(
                0.76 + col as f64 * 0.12,
                0.10 + row as f64 * 0.12,
                Color::White,
            ));
        }
        Position::new(radius, stones, Color::Black)
    }

    fn window_of(position: Position, plies: usize) -> Vec<PendingSample> {
        (0..plies)
            .map(|_| adjudication_sample(position.clone(), 0.0))
            .collect()
    }

    #[test]
    fn adjudication_awards_the_board_to_whoever_holds_it() {
        // Black occupies most of the board, so the Voronoi area is decisively
        // Black's regardless of what the value head thinks -- the samples carry
        // a root value of zero here precisely to show it is not consulted.
        let pending = window_of(lopsided(9, 1), 8);
        let outcome = adjudicate_window(&pending, 8, 0.10)
            .expect("a settled board is awarded");
        assert_eq!(outcome.winner, Some(Color::Black));
    }

    #[test]
    fn adjudication_declines_a_close_board() {
        // Balanced stones split the area near evenly, which is the case the
        // margin exists to reject: the game was cut off genuinely undecided.
        let pending = window_of(lopsided(4, 4), 8);
        assert!(adjudicate_window(&pending, 8, 0.10).is_none());
    }

    #[test]
    fn adjudication_declines_when_the_lead_changes_hands() {
        // The area leader flips inside the window, so the position is not
        // settled even though each individual ply is lopsided.
        let mut pending = window_of(lopsided(9, 1), 8);
        pending[5] = adjudication_sample(lopsided(1, 9), 0.0);
        assert!(adjudicate_window(&pending, 8, 0.10).is_none());
    }

    #[test]
    fn adjudication_declines_a_game_shorter_than_the_window() {
        assert!(adjudicate_window(&window_of(lopsided(9, 1), 4), 8, 0.10).is_none());
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
            (0..20)
                .map(|_| adjudication_sample(lopsided(3, 3), -0.99))
                .collect();
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
            (0..20)
                .map(|_| adjudication_sample(lopsided(3, 3), -0.99))
                .collect();
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
            (0..30)
                .map(|_| adjudication_sample(lopsided(3, 3), -0.88))
                .collect();
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
