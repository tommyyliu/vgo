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
use crate::tail::{TailSettings, TailTracker};
use crate::{PlayoutReport, PlayoutStep, award_by_area, play_game};
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
    pub tail: TailSettings,
    /// See [`crate::RandomStop`]. None plays every game to its end or the cap.
    pub random_stop: Option<crate::RandomStop>,
    /// A refitted komi centre, from `training/loop/komi-calibration.py`. None
    /// keeps the fixed `komi_area_coefficient` centre and relative range.
    pub komi_schedule: Option<KomiSchedule>,
}

/// Komi in points, recentred from recent results.
///
/// A board of radius `r` holds about `(1/2r)^2` stones, so `points` of komi is
/// `4 * points * r^2` of area -- the convention `komi_area_coefficient` uses,
/// where 28 is 7 points. The centre may move with board size, which runs 1/r =
/// 18 to 38 units across, and may go negative: in the first fresh run Black lost
/// most small-board games even at the lowest komi on offer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KomiSchedule {
    /// Centre in points at size 0, before `points_per_unit * size`.
    pub centre_points: f64,
    pub points_per_unit: f64,
    /// Half the sampled range in points. Komi is a normal clipped to the range.
    pub spread_points: f64,
}

impl KomiSchedule {
    /// Reads `{"centre_points": .., "points_per_unit": .., "spread_points": ..}`;
    /// other fields, such as the fit's diagnostics, are ignored.
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|error| error.to_string())?;
        let field = |name: &str| {
            value
                .get(name)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| format!("komi schedule needs a number for {name}"))
        };
        let schedule = Self {
            centre_points: field("centre_points")?,
            points_per_unit: field("points_per_unit")?,
            spread_points: field("spread_points")?,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.centre_points.is_finite() && self.points_per_unit.is_finite()) {
            return Err("komi schedule centre must be finite".to_owned());
        }
        if !(self.spread_points.is_finite() && self.spread_points >= 0.0) {
            return Err("komi schedule spread must be finite and nonnegative".to_owned());
        }
        Ok(())
    }

    /// The sampled range, in area units, for a board of this radius.
    #[must_use]
    pub fn range(&self, radius: f64) -> (f64, f64) {
        let centre = self.centre_points + self.points_per_unit / radius;
        let area = 4.0 * radius * radius;
        (
            (centre - self.spread_points) * area,
            (centre + self.spread_points) * area,
        )
    }
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
        let komi = if let Some(schedule) = self.komi_schedule {
            let (low, high) = schedule.range(radius);
            if high > low {
                sampled_komi(game_seed, low, high)
            } else {
                low
            }
        } else if self.board_mix.is_empty() {
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

/// A uniform draw in `[0, 1)` from a game seed, by SplitMix64's finalizer.
pub(crate) fn seeded_unit(game_seed: u64) -> f64 {
    let mut value = game_seed ^ 0x2545_f491_4f6c_dd1d;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / (1_u64 << 53) as f64
}

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

/// Komi for one game: a normal centred on the range, truncated to it.
///
/// Varying it across games is what lets one model play at any komi: a net shown
/// a single value learns that value's balance rather than the relationship. It
/// also spreads the training distribution over positions that are close under
/// *some* komi, which a fixed value cannot do.
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

#[allow(clippy::too_many_arguments)]
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
    /// The game ran out of plies rather than ending on its own.
    ///
    /// Worth tracking on its own: a hundred plies is far past where the board
    /// settles, so a high share means the sides are stalling rather than that
    /// the game is genuinely long. It was 88% among games containing a run of
    /// no-op self-captures and 0% among games without one.
    pub reached_ply_cap: bool,
    /// Ended by [`crate::RandomStop`] during a one-sided pass streak.
    pub random_stop: bool,
    /// The cap itself, which calibration reads game stage against.
    pub maximum_plies: u32,
    /// Ply ranges played in the tail, end exclusive.
    pub tail_spans: Vec<(u32, u32)>,
    /// How many of this game's samples came from those ranges.
    pub tail_samples: usize,
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
    /// Per ply, what each search found. These can't be recovered from a finished
    /// game, and any tail rule can be tested against them offline.
    pub searches: SearchLog,
}

/// Per-ply measurements of what search did, all from the side to move's turn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchLog {
    /// Root value, from Black's view.
    pub root_values: Vec<f32>,
    /// KL(visits || prior) over the root's children: how far search moved from
    /// what the net already believed.
    pub kl: Vec<f32>,
    /// How far ahead the most visited move is of the next, as root value from the
    /// mover's view. Zero when there is only one move.
    pub top_gap: Vec<f32>,
}

impl SearchLog {
    fn record(&mut self, search: &SearchResult, to_move: Color) {
        self.root_values.push(search.root_black_value() as f32);
        self.kl.push(search_kl(search) as f32);
        self.top_gap.push(top_gap(search, to_move) as f32);
    }
}

fn search_kl(search: &SearchResult) -> f64 {
    let visits: f64 = search.children.iter().map(|c| f64::from(c.visits)).sum();
    let prior: f64 = search.children.iter().map(|c| c.prior).sum();
    if visits <= 0.0 || prior <= 0.0 {
        return 0.0;
    }
    search
        .children
        .iter()
        .filter(|child| child.visits > 0 && child.prior > 0.0)
        .map(|child| {
            let share = f64::from(child.visits) / visits;
            share * (share * prior / child.prior).ln()
        })
        .sum()
}

fn top_gap(search: &SearchResult, to_move: Color) -> f64 {
    let mut children: Vec<_> = search.children.iter().collect();
    children.sort_by_key(|child| std::cmp::Reverse(child.visits));
    let [first, second, ..] = children[..] else {
        return 0.0;
    };
    let gap = first.black_value - second.black_value;
    if to_move == Color::Black { gap } else { -gap }
}

pub struct GameSamples {
    pub samples: Vec<LabeledSample>,
    pub record: GameRecord,
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
    pub in_tail: bool,
    pub policy_eligible: bool,
    pub simulations: u32,
    pub evaluations: u64,
    pub search_duration: Duration,
}

/// Reject budgets the per-ply choice below could not honour.
fn validate_budgets(settings: &GameSettings) -> Result<(), EvaluationError> {
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
    Ok(())
}

/// Whether this ply gets the full budget and may become a policy target.
fn is_full_search(settings: &GameSettings, game_seed: u64, ply: u32) -> bool {
    ply == 0
        || seeded_unit(
            game_seed ^ u64::from(ply).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xa0761d6478bd642f,
        ) < settings.full_search_fraction
}

/// The search for one ply. Fast turns and tail plies both play greedily, without
/// root noise: a cheap search that adds noise can throw away a position it would
/// otherwise have held, which says nothing about the position.
fn ply_search_config(
    settings: &GameSettings,
    base: SearchConfig,
    full: bool,
    in_tail: bool,
) -> SearchConfig {
    let mut config = base;
    if !full {
        config.simulations = settings.fast_simulations;
    }
    if in_tail {
        config.simulations = config.simulations.min(settings.tail.simulations);
    }
    if !full || in_tail {
        config.root_exploration_noise = 0.0;
        config.temperature = 0.0;
    }
    config
}

/// What a game's samples and record need from its plies, gathered as it plays.
#[derive(Default)]
struct GameLog {
    samples: Vec<LabeledSample>,
    searches: SearchLog,
    search_budgets: Vec<u32>,
    policy_eligible: Vec<bool>,
    tail_spans: Vec<(u32, u32)>,
    tail_samples: usize,
}

impl GameLog {
    /// Record one played ply. The sample's value stays zero until the game ends.
    fn record(
        &mut self,
        step: &PlayoutStep<'_>,
        full: bool,
        in_tail: bool,
        policy_config: RasterConfig,
        game: u64,
        seed: u64,
    ) {
        self.searches.record(step.search, step.position.to_move());
        self.search_budgets.push(step.search.stats.simulations);
        self.policy_eligible.push(full);
        if in_tail {
            extend_spans(&mut self.tail_spans, step.ply);
        }
        if !full {
            return;
        }
        self.tail_samples += usize::from(in_tail);
        let target = policy_target(step.search, policy_config);
        self.samples.push(LabeledSample {
            // Store the position; rendering is a training-time choice now,
            // which also takes the rasterizer off the self-play hot path.
            position: step.position.clone(),
            policy: target.policy,
            policy_mask: target.mask,
            visits: target.visits,
            beta: target.beta,
            proposal_counts: target.proposal_counts,
            value: 0.0,
            selected_action: action_index(step.action, policy_config),
            game,
            ply: step.ply,
            seed,
        });
    }
}

fn extend_spans(spans: &mut Vec<(u32, u32)>, ply: u32) {
    match spans.last_mut() {
        Some((_, end)) if *end == ply => *end += 1,
        _ => spans.push((ply, ply + 1)),
    }
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
    validate_budgets(settings)?;
    // Policy targets, the recorded action index, and the replay policy vector all
    // live on the placement grid, which may be coarser than the render raster.
    let policy_config = RasterConfig::square(settings.policy_resolution);
    let base_config = search_config(
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
    // Board size, komi and the ply cap are all per game once a run mixes sizes,
    // and the last two follow from the first. Derived by `GameSettings` so a
    // second generator cannot arrive at a different answer.
    let (radius, komi, maximum_plies) = settings.board_for_game(game_seed);
    let mut log = GameLog::default();
    let mut tail = TailTracker::new(&settings.tail, radius, maximum_plies);
    // Cells because the playout owns the loop and both closures are its
    // arguments: the observer learns what the search closure needs next.
    let in_tail = std::cell::Cell::new(false);
    let full_search = std::cell::Cell::new(true);
    let playout = play_game(
        Position::new(radius, Vec::new(), Color::Black)
            .with_ruleset(settings.ruleset)
            .with_komi(komi),
        maximum_plies,
        settings.random_stop.map(|rule| (rule, game_seed)),
        |position, ply| {
            if stopped.load(Ordering::Acquire) {
                return Err(EvaluationError::new("replay generation cancelled"));
            }
            let full = is_full_search(settings, game_seed, ply);
            full_search.set(full);
            let config = ply_search_config(settings, base_config, full, in_tail.get());
            let started = Instant::now();
            let result = search_at_ply(position, config, game_seed, evaluator, ply)?;
            observe(GenerationProgress {
                radius,
                ply,
                maximum_plies,
                in_tail: in_tail.get(),
                policy_eligible: full,
                simulations: result.stats.simulations,
                evaluations: result.stats.evaluations,
                search_duration: started.elapsed(),
            });
            Ok(result)
        },
        |step| {
            log.record(
                &step,
                full_search.get(),
                in_tail.get(),
                policy_config,
                game_index,
                game_seed,
            );
            let root_value = step.search.root_black_value();
            tail.observe(step.ply, step.position.to_move(), root_value);
            in_tail.set(tail.active());
        },
    )?;
    Ok(finish_game(playout, log, game_index, maximum_plies))
}

/// Label a played game and summarize it.
///
/// Natural terminal games have a rules outcome. Capped games are awarded by area
/// on the board after the final move, matching the arena, rather than discarded:
/// a hundred plies is far past where the board settles, and refusing them threw
/// away the close games a balanced komi produces, which are the positions the run
/// is trying to learn from. Search evaluations never decide these labels.
fn finish_game(
    playout: PlayoutReport,
    log: GameLog,
    game_index: u64,
    maximum_plies: u32,
) -> GameSamples {
    let GameLog {
        mut samples,
        searches,
        search_budgets,
        policy_eligible,
        tail_spans,
        tail_samples,
    } = log;
    let final_position = &playout.final_position;
    let outcome = playout
        .outcome
        .unwrap_or_else(|| award_by_area(final_position));
    let black_value = outcome.black_utility() as f32;
    for sample in &mut samples {
        sample.value = if sample.position.to_move() == Color::Black {
            black_value
        } else {
            -black_value
        };
    }
    let final_score = Analysis::new(final_position).score;
    let record = GameRecord {
        game: game_index,
        komi: final_position.komi(),
        radius: final_position.radius(),
        plies: playout.stats.plies,
        passes: playout.stats.passes,
        self_captures: playout.stats.self_captures,
        black_utility: black_value,
        margin: (final_score.black - final_score.white - final_position.komi()).abs(),
        black_area: final_score.black,
        white_area: final_score.white,
        reached_ply_cap: playout.outcome.is_none(),
        random_stop: playout.random_stop,
        maximum_plies,
        tail_spans,
        tail_samples,
        // Filled by the writer, which owns the shard-relative offset.
        first_sample: 0,
        sample_count: samples.len(),
        final_stones: final_position
            .stones()
            .iter()
            .map(|stone| (stone.x, stone.y, u8::from(stone.color == Color::White)))
            .collect(),
        search_budgets,
        policy_eligible,
        searches,
    };
    GameSamples { samples, record }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgo_search::{CandidateSource, ChildSummary, SearchStats};

    fn searched(children: &[(f64, u32, f64)], to_move: Color) -> SearchResult {
        let children = children
            .iter()
            .map(|&(prior, visits, black_value)| ChildSummary {
                action: Action::Pass,
                source: CandidateSource::AreaSequence,
                prior,
                visits,
                black_value,
                proposal_count: 0,
                beta: None,
            })
            .collect();
        SearchResult::from_children(Action::Pass, children, SearchStats::default(), to_move)
    }

    #[test]
    fn search_kl_is_zero_when_search_agrees_with_the_prior() {
        let agreed = searched(&[(0.75, 30, 0.0), (0.25, 10, 0.0)], Color::Black);
        assert!(search_kl(&agreed).abs() < 1e-12);
    }

    #[test]
    fn search_kl_grows_as_search_moves_away_from_the_prior() {
        let moved = searched(&[(0.25, 30, 0.0), (0.75, 10, 0.0)], Color::Black);
        // 0.75 ln 3 + 0.25 ln(1/3).
        let expected = 0.5 * 3.0_f64.ln();
        assert!((search_kl(&moved) - expected).abs() < 1e-12);
    }

    #[test]
    fn search_kl_renormalizes_a_partial_prior() {
        // Candidates hold only part of the prior, which shouldn't read as surprise.
        let partial = searched(&[(0.075, 30, 0.0), (0.025, 10, 0.0)], Color::Black);
        assert!(search_kl(&partial).abs() < 1e-12);
    }

    #[test]
    fn top_gap_reads_from_the_movers_view() {
        let children = [(0.5, 30, 0.4), (0.5, 10, 0.1)];
        let black = top_gap(&searched(&children, Color::Black), Color::Black);
        assert!((black - 0.3).abs() < 1e-12);
        let white = [(0.5, 30, -0.4), (0.5, 10, -0.1)];
        let white = top_gap(&searched(&white, Color::White), Color::White);
        assert!((white - 0.3).abs() < 1e-12);
    }

    #[test]
    fn top_gap_is_zero_with_one_move() {
        let only = searched(&[(1.0, 40, 0.9)], Color::Black);
        assert_eq!(top_gap(&only, Color::Black), 0.0);
    }

    #[test]
    fn tail_plies_play_cheaply_and_without_noise() {
        let base = search_config(100, 16, 1.0, 30, 4, 321, 0.1, 6.0);
        let settings = GameSettings {
            policy_resolution: 16,
            simulations: 100,
            full_search_fraction: 1.0,
            fast_simulations: 20,
            coarse_pool: 16,
            temperature: 1.0,
            temperature_plies: 30,
            leaf_batch: 4,
            maximum_candidates: 321,
            root_exploration_noise: 0.1,
            widening_coefficient: 6.0,
            seed: 1,
            radius: 1.0 / 18.0,
            board_mix: Vec::new(),
            komi_low: 0.017,
            komi_high: 0.137,
            komi_area_coefficient: KOMI_AREA_COEFFICIENT,
            maximum_plies: 70,
            ruleset: Ruleset::default(),
            tail: TailSettings {
                schedule: None,
                simulations: 40,
                window: 5,
                exit_value: 0.6,
                exit_window: 2,
            },
            random_stop: None,
            komi_schedule: None,
        };
        let normal = ply_search_config(&settings, base, true, false);
        assert_eq!(normal.simulations, 100);
        assert!(normal.root_exploration_noise > 0.0);
        let tail = ply_search_config(&settings, base, true, true);
        assert_eq!(tail.simulations, 40);
        assert_eq!(tail.root_exploration_noise, 0.0);
        assert_eq!(tail.temperature, 0.0);
        // The tail never raises a budget that was already cheaper.
        assert_eq!(
            ply_search_config(&settings, base, false, true).simulations,
            20
        );
    }

    #[test]
    fn consecutive_tail_plies_share_a_span() {
        let mut spans = Vec::new();
        for ply in [4, 5, 6, 9] {
            extend_spans(&mut spans, ply);
        }
        assert_eq!(spans, vec![(4, 7), (9, 10)]);
    }

    #[test]
    fn a_seven_point_schedule_is_the_fixed_coefficient_centre() {
        let schedule = KomiSchedule {
            centre_points: 7.0,
            points_per_unit: 0.0,
            spread_points: 0.0,
        };
        for radius in [1.0 / 18.0, 1.0 / 26.0, 1.0 / 38.0] {
            let (low, high) = schedule.range(radius);
            let fixed = komi_centre_for_radius(radius, 28.0);
            assert!((low - fixed).abs() < 1e-12 && (high - fixed).abs() < 1e-12);
        }
    }

    #[test]
    fn the_centre_moves_with_board_size_and_may_go_negative() {
        let schedule = KomiSchedule {
            centre_points: -2.0,
            points_per_unit: 0.25,
            spread_points: 1.0,
        };
        // 18 units across: -2 + 4.5 = 2.5 points, so 1.5 to 3.5.
        let r = 1.0 / 18.0;
        let (low, high) = schedule.range(r);
        assert!((low - 1.5 * 4.0 * r * r).abs() < 1e-12);
        assert!((high - 3.5 * 4.0 * r * r).abs() < 1e-12);
        let negative = KomiSchedule {
            centre_points: -3.0,
            points_per_unit: 0.0,
            spread_points: 0.5,
        };
        assert!(negative.range(r).1 < 0.0);
    }

    #[test]
    fn schedules_parse_and_reject_bad_input() {
        let parsed = KomiSchedule::parse(
            r#"{"centre_points": 6.5, "points_per_unit": 0.01, "spread_points": 5.5, "games": 900}"#,
        )
        .unwrap();
        assert_eq!(parsed.spread_points, 5.5);
        assert!(KomiSchedule::parse(r#"{"centre_points": 6.5, "points_per_unit": 0.0}"#).is_err());
        assert!(
            KomiSchedule::parse(
                r#"{"centre_points": 6.5, "points_per_unit": 0.0, "spread_points": -1}"#
            )
            .is_err()
        );
    }

    #[test]
    fn a_schedule_replaces_the_coefficient_in_each_game() {
        let mut settings = GameSettings {
            policy_resolution: 16,
            simulations: 100,
            full_search_fraction: 1.0,
            fast_simulations: 20,
            coarse_pool: 16,
            temperature: 1.0,
            temperature_plies: 30,
            leaf_batch: 4,
            maximum_candidates: 321,
            root_exploration_noise: 0.1,
            widening_coefficient: 6.0,
            seed: 1,
            radius: 1.0 / 18.0,
            board_mix: Vec::new(),
            komi_low: 0.017,
            komi_high: 0.137,
            komi_area_coefficient: KOMI_AREA_COEFFICIENT,
            maximum_plies: 70,
            ruleset: Ruleset::default(),
            tail: TailSettings {
                schedule: None,
                simulations: 40,
                window: 5,
                exit_value: 0.6,
                exit_window: 2,
            },
            random_stop: None,
            komi_schedule: Some(KomiSchedule {
                centre_points: -1.0,
                points_per_unit: 0.0,
                spread_points: 0.5,
            }),
        };
        let area = 4.0 / (18.0 * 18.0);
        for seed in 0..50 {
            let (_, komi, _) = settings.board_for_game(seed);
            assert!((-1.5 * area..=-0.5 * area).contains(&komi), "komi {komi}");
        }
        settings.komi_schedule = None;
        assert!(settings.board_for_game(3).1 > 0.0);
    }
}
