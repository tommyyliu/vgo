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

use vgo_core::{Analysis, Color, Position, Ruleset};
use vgo_search::{EvaluationError, Evaluator, search_at_ply};

use crate::{ResignRule, adjudicate_positions, award_by_area, play_game_with_resignation};
use crate::replay_stream::LabeledSample;
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
    pub ply_sample_rate: f64,
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
    pub fn board_for_game(&self, game_seed: u64) -> (f64, f64, u32) {
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

    /// Whether this ply is recorded, at the configured rate.
    ///
    /// Drawn per (game, ply) so the kept set is spread through the game rather
    /// than a prefix, and stable for a given seed.
    #[must_use]
    pub fn records_ply(&self, game_seed: u64, ply: u32) -> bool {
        let rate = self.ply_sample_rate.clamp(0.0, 1.0);
        if rate >= 1.0 {
            return true;
        }
        seeded_unit(
            game_seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(u64::from(ply)),
        ) < rate
    }
}

use vgo_search::{Action, SearchConfig, SearchResult};

pub struct PolicyTarget {
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
pub fn seeded_unit(game_seed: u64) -> f64 {
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
pub const MINIMUM_BOARD_UNITS: f64 = 18.0;

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
pub fn sampled_radius(game_seed: u64, bands: &[BoardBand], fallback: f64) -> f64 {
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
pub fn komi_centre_for_radius(radius: f64, coefficient: f64) -> f64 {
    coefficient * radius * radius
}

/// Plies to allow a game on this board.
///
/// The cap exists to stop a game running forever, so it has to scale with what
/// the board can hold -- about `1/r^2` stones. Left at the mini board's value a
/// standard game is cut off around a fifth of the way in, and every value
/// target from it is a truncation artifact rather than a result.
#[must_use]
pub fn maximum_plies_for_radius(radius: f64, plies_at_reference: u32, reference: f64) -> u32 {
    if !(radius > 0.0 && reference > 0.0) {
        return plies_at_reference;
    }
    let scale = (reference / radius).powi(2);
    ((plies_at_reference as f64) * scale).ceil().min(4096.0) as u32
}

pub fn sampled_komi(game_seed: u64, low: f64, high: f64) -> f64 {
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

pub fn policy_target(result: &SearchResult, config: RasterConfig) -> PolicyTarget {
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

pub fn action_index(action: Action, config: RasterConfig) -> u32 {
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

pub struct PendingSample {
    pub position: Position,
    pub root_black_value: f64,
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
pub const CALIBRATION_WINDOWS: [u32; 4] = [5, 8, 12, 16];

pub const CALIBRATION_THRESHOLDS: [f64; 9] = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999];

/// What resignation would have done to one calibrating game at one threshold.
#[derive(Clone, Copy, Debug)]
pub struct ResignTrial {
    pub threshold: f64,
    /// Consecutive own-turn agreements required before conceding.
    pub window: u32,
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
    let final_position: std::cell::RefCell<Option<Position>> =
        std::cell::RefCell::new(None);
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
            // Past a soft concession the game is only being played out to reach
            // a real terminal state, so it does not need full search.
            let mut config_for_ply = search_config;
            if ply >= soft_from.get() {
                config_for_ply.simulations = settings.resign_soft_simulations;
            }
            search_at_ply(position, config_for_ply, game_seed, evaluator, ply)
        },
        |step| {
            if let Some(ply) = step.soft_resign_ply {
                soft_from.set(ply);
            }
            // The last position the game reached, recorded or not. Adjudication
            // scores this rather than the last *sample*: under subsampling they
            // are different positions, and scoring a board several plies short
            // of the end awards the game on a state neither player played to.
            final_position.replace(Some(step.position.clone()));
            // Keep a fraction of plies. The draw is per (game, ply) so the kept
            // set is spread through the game rather than a prefix, and stable
            // for a given seed.
            if !settings.records_ply(game_seed, step.ply) {
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
        calibration_trials(&pending, settings.resign_window, black_value > 0.0)
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
            margin: {
                let analysis = Analysis::new(&playout.final_position);
                (analysis.score.black - analysis.score.white - playout.final_position.komi()).abs()
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
                .map(|stone| (stone.x, stone.y, u8::from(stone.color == Color::White)))
                .collect(),
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
pub fn calibration_trials(
    pending: &[PendingSample],
    _window: u32,
    black_won: bool,
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
                    let conceding_won = (conceding == Color::Black) == black_won;
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
pub fn resign_exempt(game_seed: u64, fraction: f64) -> bool {
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
