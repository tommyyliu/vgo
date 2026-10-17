//! Machinery shared by every generator: how a game is set up, sampled and
//! labelled, independent of how results are batched into files.
//!
//! This lived inside `generate_demo` while there was one generator. A second one
//! reimplementing any of it -- which board a game is played on, what komi
//! balances it, how a search becomes a policy target -- would be a second
//! definition of the training data the moment either changed. The rules already
//! carry that problem across `vgo-core` and the JS reference, with mutual "must
//! match" comments and a bug that had to be found twice.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use vgo_core::{Analysis, Color, Position, Ruleset};
use vgo_search::{EvaluationError, Evaluator, search_at_ply};

use crate::replay_stream::LabeledSample;
use crate::{ResignRule, award_by_area, play_game_with_resignation};
use vgo_raster::{RasterConfig, action_pixel};

/// Everything a single game needs, independent of how games are batched.
///
/// `generate_game` used to take the generator's whole clap `Config`, which tied
/// it to one binary's flags. A generator that batches games differently wants
/// the same game -- same boards, same komi law, same targets -- and differs only
/// in what it does with the results.
#[derive(Clone, Debug)]
pub struct GameSettings {
    pub policy_resolution: usize,
    pub simulations: u32,
    /// Probability of a full search; first ply is always full to retain a sample.
    pub full_search_fraction: f64,
    pub fast_simulations: u32,
    pub coarse_pool: usize,
    pub temperature: f64,
    pub temperature_plies: u32,
    pub leaf_batch: usize,
    pub maximum_candidates: usize,
    pub root_exploration_noise: f64,
    pub widening_coefficient: f64,
    pub seed: u64,
    pub radius: f64,
    pub board_mix: Vec<BoardBand>,
    pub komi_low: f64,
    pub komi_high: f64,
    pub komi_area_coefficient: f64,
    pub maximum_plies: u32,
    pub ruleset: Ruleset,
    pub resign_threshold: f64,
    pub resign_window: u32,
    pub resign_minimum_ply: u32,
    pub resign_soft_simulations: u32,
    pub resign_disable_fraction: f64,
}

impl GameSettings {
    /// The board this game is played on, and what follows from it.
    ///
    /// Komi and the ply cap are not free parameters once a run mixes board
    /// sizes: komi scales with the board's area per stone, and the cap with how
    /// many stones the board holds. Derived together so no caller can set one
    /// without the others.
    #[must_use]
    pub(crate) fn board_for_game(&self, game_seed: u64) -> (f64, f64, u32) {
        let radius = sampled_radius(game_seed, &self.board_mix, self.radius);
        let komi = if self.board_mix.is_empty() {
            sampled_komi(game_seed, self.komi_low, self.komi_high)
        } else {
            let centre = komi_centre_for_radius(radius, self.komi_area_coefficient);
            let width = (self.komi_high - self.komi_low).max(0.0);
            let relative = if self.komi_low + self.komi_high > 0.0 {
                width / (0.5 * (self.komi_low + self.komi_high)).max(f64::MIN_POSITIVE)
            } else {
                0.5
            };
            let half = 0.5 * relative * centre;
            sampled_komi(game_seed, centre - half, centre + half)
        };
        let plies = maximum_plies_for_radius(radius, self.maximum_plies, self.radius);
        (radius, komi, plies)
    }
}

use vgo_search::{Action, SearchConfig, SearchResult};

pub(crate) struct PolicyTarget {
    /// Normalized visit distribution over cells (the legacy target).
    pub policy: Vec<f32>,
    /// 1.0 for any cell that received a candidate, else 0.0.
    pub mask: Vec<f32>,
    /// Raw visit counts per cell (unnormalized), for off-policy reweighting.
    pub visits: Vec<f32>,
    /// Coarse->fine sampling probability beta per cell; 0.0 for legacy/pass
    /// candidates (which have no factored sampling probability).
    pub beta: Vec<f32>,
    /// Number of raw coarse->fine proposal draws landing in each cell. Legacy
    /// candidates and pass have zero multiplicity.
    pub proposal_counts: Vec<u32>,
}

/// A uniform draw in `[0, 1)` from a game seed.
///
/// Shares `resign_exempt`'s SplitMix64 finalizer but with a different constant,
/// so a game's komi and its exemption are independent -- reusing the stream
/// would correlate the two and make every exempt game share a komi.
pub(crate) fn seeded_unit(game_seed: u64) -> f64 {
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
/// One band of the board mix: a weight and a range of board widths in units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoardBand {
    pub weight: f64,
    pub low_units: f64,
    pub high_units: f64,
}

/// Smallest board this will play, in stone diameters across.
///
/// Below here the game changes character rather than merely getting smaller.
pub(crate) const MINIMUM_BOARD_UNITS: f64 = 18.0;

/// Parse `WEIGHT:UNITS` or `WEIGHT:LOW-HIGH`.
pub fn parse_board_mix(specs: &[String]) -> Result<Vec<BoardBand>, String> {
    let mut bands = Vec::new();
    for spec in specs {
        let (weight, extent) = spec
            .split_once(':')
            .ok_or_else(|| format!("board mix {spec:?} is not WEIGHT:UNITS"))?;
        let weight: f64 = weight
            .trim()
            .parse()
            .map_err(|_| format!("board mix {spec:?} has a non-numeric weight"))?;
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!("board mix {spec:?} needs a positive weight"));
        }
        let (low, high) = match extent.trim().split_once('-') {
            Some((low, high)) => (low, high),
            None => (extent.trim(), extent.trim()),
        };
        let low: f64 = low
            .trim()
            .parse()
            .map_err(|_| format!("board mix {spec:?} has non-numeric units"))?;
        let high: f64 = high
            .trim()
            .parse()
            .map_err(|_| format!("board mix {spec:?} has non-numeric units"))?;
        if !(low.is_finite() && high.is_finite()) || low > high {
            return Err(format!("board mix {spec:?} has a malformed range"));
        }
        if low < MINIMUM_BOARD_UNITS {
            return Err(format!(
                "board mix {spec:?} goes below {MINIMUM_BOARD_UNITS} units; \
                 smaller boards are a different game"
            ));
        }
        bands.push(BoardBand {
            weight,
            low_units: low,
            high_units: high,
        });
    }
    Ok(bands)
}

/// This game's radius, drawn from the mix.
///
/// Uniform in *units* within a band, so a wide band spreads across board sizes
/// evenly rather than concentrating in its smallest boards.
pub(crate) fn sampled_radius(game_seed: u64, bands: &[BoardBand], fallback: f64) -> f64 {
    if bands.is_empty() {
        return fallback;
    }
    let total: f64 = bands.iter().map(|band| band.weight).sum();
    let mut choice = seeded_unit(game_seed ^ 0x9e37_79b9_7f4a_7c15) * total;
    for band in bands {
        if choice < band.weight || std::ptr::eq(band, &bands[bands.len() - 1]) {
            let units = if band.high_units > band.low_units {
                let position = seeded_unit(game_seed ^ 0xc2b2_ae3d_27d4_eb4f);
                band.low_units + position * (band.high_units - band.low_units)
            } else {
                band.low_units
            };
            return 1.0 / units.max(MINIMUM_BOARD_UNITS);
        }
        choice -= band.weight;
    }
    fallback
}

/// Balanced komi at this radius, before the per-game jitter.
///
/// Komi compensates about one stone's worth of area and a board holds about
/// `1/r^2` stones, so komi as a fraction of the board goes as `r^2`. Three
/// things agree on the coefficient: our own measurement of 0.104 at `r = 1/18`,
/// which fixes it at 33.7; Go's 9x9 komi of 8.6% of the board against our
/// 10.4% on a board of nearly the same stone capacity; and Go's 9x9-to-19x19
/// exponent of 1.89, near enough to 2.
///
/// It is a prior, not a law. `fit_komi_power_law` re-estimates both the
/// coefficient and the exponent from real games once a run has enough of them,
/// and the exponent is known to drift on small boards -- which is one reason
/// this refuses to play them.
pub const KOMI_AREA_COEFFICIENT: f64 = 0.104 * 18.0 * 18.0;

#[must_use]
pub(crate) fn komi_centre_for_radius(radius: f64, coefficient: f64) -> f64 {
    coefficient * radius * radius
}

/// Plies to allow a game on this board.
///
/// The cap exists to stop a game running forever, so it has to scale with what
/// the board can hold -- about `1/r^2` stones. Left at the mini board's value a
/// standard game is cut off around a fifth of the way in, and every value
/// target from it is a truncation artifact rather than a result.
#[must_use]
pub(crate) fn maximum_plies_for_radius(
    radius: f64,
    plies_at_reference: u32,
    reference: f64,
) -> u32 {
    if !(radius > 0.0 && reference > 0.0) {
        return plies_at_reference;
    }
    let scale = (reference / radius).powi(2);
    ((plies_at_reference as f64) * scale).ceil().min(4096.0) as u32
}

pub(crate) fn sampled_komi(game_seed: u64, low: f64, high: f64) -> f64 {
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

pub(crate) fn policy_target(result: &SearchResult, config: RasterConfig) -> PolicyTarget {
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

pub(crate) fn action_index(action: Action, config: RasterConfig) -> u32 {
    match action {
        Action::Pass => config.pixels() as u32,
        Action::Place(point) => action_pixel(point.x, point.y, config) as u32,
    }
}

pub fn search_config(
    simulations: u32,
    coarse_pool: usize,
    temperature: f64,
    temperature_plies: u32,
    leaf_batch: usize,
    maximum_candidates: usize,
    root_exploration_noise: f64,
    widening_coefficient: f64,
) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config.temperature = temperature;
    config.temperature_plies = temperature_plies;
    config.leaf_batch = leaf_batch.max(1);
    config.maximum_candidates = maximum_candidates.max(config.initial_candidates);
    config.root_exploration_noise = root_exploration_noise;
    config.widening_coefficient = widening_coefficient;
    config
}

/// Policy slots a record needs for a given draw budget.
///
/// Provisioned for the worst case -- every draw landing on a distinct cell --
/// rather than projected from a collision model. An earlier version modelled it
/// from measured collisions (79.6 draws touching 68.6 cells, inverting to an
/// effective pool of ~259) and sized 321 draws at 224 slots. That dropped
/// 109,111 cells in one shard: the model was fitted on runs with no exploration
/// noise, and mixing uniform mass into the proposal flattens it so collisions
/// become rare and distinct cells approach the draw count. A model of one
/// regime sized a record for a different one.
///
/// The cost of over-provisioning is zero-padding on disk. The cost of
/// under-provisioning is silently truncated policy targets, so this errs the
/// cheap way.
pub fn replay_capacity_for(maximum_candidates: usize, policy_size: usize) -> usize {
    // Every candidate plus pass.
    let wanted = maximum_candidates.saturating_add(1).max(64);
    let rounded = wanted.div_ceil(32) * 32;
    rounded.min(policy_size)
}

pub(crate) struct PendingSample {
    pub position: Position,
    pub policy: Vec<f32>,
    pub policy_mask: Vec<f32>,
    pub visits: Vec<f32>,
    pub beta: Vec<f32>,
    pub proposal_counts: Vec<u32>,
    pub to_move: Color,
    pub selected_action: u32,
    pub game: u64,
    pub ply: u32,
    pub seed: u64,
}

#[derive(Clone, Copy)]
struct ResignObservation {
    root_black_value: f64,
    to_move: Color,
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
pub struct GameRecord {
    pub game: u64,
    pub komi: f64,
    /// The board this game was played on. Constant across a single-radius run
    /// and the whole point of a mixed one -- and the komi controller cannot fit
    /// its balance law without it, since balanced komi is a function of radius.
    pub radius: f64,
    pub plies: u32,
    /// Passes and no-op self-captures, which are how a stalled game passes the
    /// time. A capped game with many of either is stalling, not playing long.
    pub passes: u64,
    pub self_captures: u64,
    /// Black-relative: +1 Black, -1 White, 0 tie.
    pub black_utility: f32,
    /// Area margin, always non-negative.
    pub margin: f64,
    /// Each side's Voronoi area on the final board, before komi.
    ///
    /// The score target: the win/loss label says who won, these say by how
    /// much, and keeping them before komi lets a reader score the same board at
    /// any komi. `black_area - white_area - komi` is Black's signed margin.
    pub black_area: f64,
    pub white_area: f64,
    pub reached_ply_cap: bool,
    pub resigned: bool,
    /// Ply a soft concession fired at, if one did. The game played on from
    /// there at reduced search, so `resigned` stays false and the outcome is a
    /// real one -- this is what lets the rule be scored after the fact.
    pub soft_resign_ply: Option<u32>,
    pub first_sample: usize,
    pub sample_count: usize,
    /// The board as the game ended, for the ownership target.
    ///
    /// Stored as stones rather than as a rendered map: ownership at a point is
    /// the colour of the nearest stone, which is what the rasterizer already
    /// computes for the voronoi channels, so re-rendering from the position
    /// cannot drift from the inputs the net reads. It is also 26x smaller --
    /// ~600 bytes against a 16 KB int8 map at 128x128.
    pub final_stones: Vec<(f64, f64, u8)>,
    /// Actual simulations and policy-target eligibility for every played ply.
    pub search_budgets: Vec<u32>,
    pub policy_eligible: Vec<bool>,
}

pub struct GameSamples {
    pub samples: Vec<LabeledSample>,
    /// This game's outcome, absent only when it produced no samples.
    pub record: Option<GameRecord>,
    pub completed: bool,
    /// The game ran out of plies rather than ending on its own.
    ///
    /// Worth tracking on its own: a hundred plies is far past where the board
    /// settles, so a high share means the sides are stalling rather than that
    /// the game is genuinely long. It was 88% among games containing a run of
    /// no-op self-captures and 0% among games without one.
    pub reached_ply_cap: bool,
    /// Counterfactual resignation outcomes for this game, one entry per
    /// candidate threshold. Produced for every game whose true result is known
    /// independently of the rule being measured: under hard resignation only
    /// the games exempted by `--resign-disable-fraction`, under soft
    /// resignation all of them, since a soft concession plays on to a real
    /// terminal state. See the note where this is populated.
    pub calibration: Vec<ResignTrial>,
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
/// Windows swept alongside the thresholds.
///
/// The window is the other half of the rule and the untested half: raising the
/// threshold from 0.95 to 0.98 barely moved the false-positive rate, which says
/// the head is confidently wrong rather than marginally over the line. Demanding
/// more consecutive plies of agreement is a different lever -- it asks the
/// losing seat to keep conceding across more of its own turns, which noise
/// clears far less easily than one confident evaluation.
pub(crate) const CALIBRATION_WINDOWS: [u32; 4] = [5, 8, 12, 16];

pub(crate) const CALIBRATION_THRESHOLDS: [f64; 9] =
    [0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999];

/// What resignation would have done to one calibrating game at one threshold.
#[derive(Clone, Copy, Debug)]
pub struct ResignTrial {
    pub threshold: f64,
    /// Consecutive own-turn agreements required before conceding.
    pub window: u32,
    pub minimum_ply: u32,
    /// The rule would have conceded this game.
    pub fired: bool,
    /// It would have conceded for the side that actually won: a false positive,
    /// and the label the loop would have learned is the wrong one.
    pub wrong: bool,
    /// Plies that would have been skipped had it fired.
    pub plies_saved: u32,
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
    pub fired_confidence: f64,
}

pub fn generate_game(
    settings: &GameSettings,
    evaluator: &dyn Evaluator,
    game_index: u64,
    stopped: &AtomicBool,
) -> Result<GameSamples, EvaluationError> {
    generate_game_observed(settings, evaluator, game_index, stopped, |_| {})
}

/// One completed search, including searches in games not yet published.
#[derive(Clone, Copy, Debug)]
pub struct GenerationProgress {
    pub radius: f64,
    pub ply: u32,
    pub maximum_plies: u32,
    pub soft_resigned: bool,
    pub policy_eligible: bool,
    pub simulations: u32,
    pub evaluations: u64,
    pub search_duration: Duration,
}

/// Generate the same game while reporting progress once per successful search.
/// The observer cannot change the search settings or selected move.
pub fn generate_game_observed(
    settings: &GameSettings,
    evaluator: &dyn Evaluator,
    game_index: u64,
    stopped: &AtomicBool,
    mut observe: impl FnMut(GenerationProgress),
) -> Result<GameSamples, EvaluationError> {
    if !settings.full_search_fraction.is_finite()
        || settings.full_search_fraction <= 0.0
        || settings.full_search_fraction > 1.0
        || (settings.full_search_fraction < 1.0
            && (settings.fast_simulations == 0
                || settings.fast_simulations >= settings.simulations))
    {
        return Err(EvaluationError::new(
            "full search fraction must be in (0,1]; fast budget must be positive and below the full budget when enabled",
        ));
    }
    // Policy targets, the recorded action index, and the replay policy vector all
    // live on the placement grid, which may be coarser than the render raster.
    let policy_config = RasterConfig::square(settings.policy_resolution);
    let search_config = search_config(
        settings.simulations,
        settings.coarse_pool,
        settings.temperature,
        settings.temperature_plies,
        settings.leaf_batch,
        settings.maximum_candidates,
        settings.root_exploration_noise,
        settings.widening_coefficient,
    );
    let game_seed = settings.seed.wrapping_add(game_index);
    let mut pending = Vec::new();
    let mut observations = Vec::new();
    let mut search_budgets = Vec::new();
    let mut eligibility = Vec::new();
    let keep_policy = std::cell::Cell::new(true);
    // Exempt a deterministic fraction of games from resignation. Deriving the
    // choice from the game seed rather than a counter keeps it reproducible and
    // independent of how games are distributed across actors, so a rerun
    // exempts exactly the same games.
    let exempt_from_resignation = resign_exempt(game_seed, settings.resign_disable_fraction);
    let resign = if settings.resign_threshold > 0.0 && !exempt_from_resignation {
        ResignRule {
            threshold: settings.resign_threshold,
            window: settings.resign_window,
            disable_fraction: settings.resign_disable_fraction,
            minimum_ply: settings.resign_minimum_ply,
            soft_simulations: settings.resign_soft_simulations,
        }
    } else {
        ResignRule::disabled()
    };
    // Set by the playout the ply a soft concession fires at, read by the search
    // closure below so the rest of the game runs cheaply. A Cell rather than a
    // parameter because the playout owns the loop and the closure is its
    // argument -- neither can see the other's locals.
    let soft_from = std::cell::Cell::new(u32::MAX);
    let final_position: std::cell::RefCell<Option<Position>> = std::cell::RefCell::new(None);
    // Board size, komi and the ply cap are all per game once a run mixes sizes,
    // and the last two follow from the first. Derived by `GameSettings` so a
    // second generator cannot arrive at a different answer.
    let (radius, komi, maximum_plies) = settings.board_for_game(game_seed);
    let playout = play_game_with_resignation(
        Position::new(radius, Vec::new(), Color::Black)
            .with_ruleset(settings.ruleset)
            .with_komi(komi),
        maximum_plies,
        resign,
        |position, ply| {
            if stopped.load(Ordering::Acquire) {
                return Err(EvaluationError::new("replay generation cancelled"));
            }
            // Past a soft concession, use the existing cheaper tail budget
            // while playing toward natural termination or the normal cap.
            let mut config_for_ply = search_config;
            let full = ply == 0
                || seeded_unit(
                    game_seed
                        ^ u64::from(ply).wrapping_mul(0x9e3779b97f4a7c15)
                        ^ 0xa0761d6478bd642f,
                ) < settings.full_search_fraction;
            keep_policy.set(full);
            if !full {
                config_for_ply.simulations = settings.fast_simulations;
                config_for_ply.root_exploration_noise = 0.0;
                config_for_ply.temperature = 0.0;
            }
            if ply >= soft_from.get() {
                config_for_ply.simulations = config_for_ply
                    .simulations
                    .min(settings.resign_soft_simulations);
            }
            let started = Instant::now();
            let result = search_at_ply(position, config_for_ply, game_seed, evaluator, ply)?;
            observe(GenerationProgress {
                radius,
                ply,
                maximum_plies,
                soft_resigned: ply >= soft_from.get(),
                policy_eligible: full,
                simulations: result.stats.simulations,
                evaluations: result.stats.evaluations,
                search_duration: started.elapsed(),
            });
            Ok(result)
        },
        |step| {
            if let Some(ply) = step.soft_resign_ply {
                soft_from.set(ply);
            }
            // The last position the game reached, recorded or not: the board
            // *after* this step's move. `step.position` is the board the move
            // was searched from, one ply short, and scoring a capped game on it
            // awarded the result on a board neither player finished on -- in a
            // close game, to the wrong side, and inconsistently with the final
            // areas recorded beside it.
            final_position.replace(Some(step.transition.position.clone()));
            observations.push(ResignObservation {
                root_black_value: step.search.root_black_value(),
                to_move: step.position.to_move(),
            });
            search_budgets.push(step.search.stats.simulations);
            eligibility.push(keep_policy.get());
            if !keep_policy.get() {
                return;
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
                selected_action: action_index(step.action, policy_config),
                game: game_index,
                ply: step.ply,
                seed: game_seed,
            });
        },
    )?;
    // Natural terminal games have a rules outcome. Capped games are awarded
    // by area on the board after the final move, matching the arena. Search
    // evaluations do not decide these labels; they calibrate resignation below.
    let outcome = match playout.outcome {
        Some(outcome) => outcome,
        None => {
            let closing = final_position.borrow().clone();
            // A game at the ply cap is decided on its final position rather
            // than discarded. A hundred plies is far past where the board
            // settles, and the margin was rejecting close games -- which are
            // the ones a balanced komi produces, so refusing them threw away
            // exactly the positions the run is trying to learn from.
            match closing.as_ref() {
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
        calibration_trials(
            &observations,
            settings.resign_minimum_ply,
            f64::from(black_value),
        )
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
    let final_score = Analysis::new(&playout.final_position).score;
    Ok(GameSamples {
        record: Some(GameRecord {
            game: game_index,
            komi: playout.final_position.komi(),
            radius: playout.final_position.radius(),
            plies: playout.stats.plies,
            passes: playout.stats.passes,
            self_captures: playout.stats.self_captures,
            black_utility: black_value,
            // The board margin, not `outcome.margin`: a resignation reports a
            // real winner but leaves the margin at zero, since no area was
            // scored. Recomputing it from the final position is what makes a
            // resigned game reviewable -- it says how far behind the conceding
            // side actually was, which is the only way to judge the rule.
            margin: (final_score.black - final_score.white - playout.final_position.komi()).abs(),
            black_area: final_score.black,
            white_area: final_score.white,
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
                .map(|stone| (stone.x, stone.y, u8::from(stone.color == Color::White)))
                .collect(),
            search_budgets,
            policy_eligible: eligibility,
        }),
        samples,
        completed: true,
        reached_ply_cap: playout.outcome.is_none(),
        calibration,
    })
}

/// Replays the resign rule over a finished game at each candidate threshold.
///
/// Only meaningful for games played to a real result: the rule's error rate is
/// how often it would have conceded for the eventual winner, and that is
/// unknowable for a game the rule already ended. That set is the exempt games
/// under hard resignation and every game under soft, which is the condition the
/// caller applies.
///
/// One caveat the counters cannot express: after a soft concession the rest of
/// the game is searched at `--resign-soft-simulations`, so both the root values
/// this replays and the outcome it scores against come from the cheaper search.
/// A low error rate there is partly the rule agreeing with a playout it shaped.
/// Games exempted by `--resign-disable-fraction` are the only ones measured at
/// full strength throughout.
fn calibration_trials(
    pending: &[ResignObservation],
    minimum_ply: u32,
    black_utility: f64,
) -> Vec<ResignTrial> {
    CALIBRATION_THRESHOLDS
        .iter()
        .flat_map(|&threshold| {
            CALIBRATION_WINDOWS
                .iter()
                .map(move |&window| (threshold, window))
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
                if streak[seat] >= window && index >= minimum_ply as usize {
                    fired_at = Some((index, sample.to_move, weakest[seat]));
                    break;
                }
            }
            match fired_at {
                None => ResignTrial {
                    threshold,
                    window,
                    minimum_ply,
                    fired: false,
                    wrong: false,
                    plies_saved: 0,
                    fired_confidence: 0.0,
                },
                Some((index, conceding, confidence)) => {
                    let conceding_won = black_utility == 0.0
                        || (conceding == Color::Black) == (black_utility > 0.0);
                    ResignTrial {
                        threshold,
                        window,
                        minimum_ply,
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

#[cfg(test)]
mod calibration_tests {
    use super::*;

    #[test]
    fn calibration_obeys_live_minimum_ply_and_counts_draw_concessions_as_wrong() {
        let observations: Vec<_> = (0..30)
            .map(|ply| ResignObservation {
                root_black_value: -1.0,
                to_move: if ply % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            })
            .collect();
        let trials = calibration_trials(&observations, 20, 1.0);
        let trial = trials
            .iter()
            .find(|trial| trial.threshold == 0.99 && trial.window == 5)
            .unwrap();
        assert!(trial.fired && trial.wrong);
        assert_eq!(trial.minimum_ply, 20);
        assert_eq!(trial.plies_saved, 9); // Fires at ply 20, not ply 8.
        assert!(
            calibration_trials(&observations, 30, 1.0)
                .iter()
                .all(|trial| !trial.fired)
        );
        let tied = calibration_trials(&observations, 20, 0.0);
        assert!(
            tied.iter()
                .filter(|trial| trial.fired)
                .all(|trial| trial.wrong)
        );
        let won_by_white = calibration_trials(&observations, 20, -1.0);
        assert!(won_by_white.iter().all(|trial| !trial.wrong));
    }
}

/// Whether a game is exempt from resignation, decided from its seed.
///
/// Hashing the seed rather than counting games keeps the exempt set stable
/// across reruns and independent of actor scheduling, which matters because
/// these games are the calibration sample: they have to be a fair draw, not
/// whichever games happened to land on a particular worker.
pub(crate) fn resign_exempt(game_seed: u64, fraction: f64) -> bool {
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
