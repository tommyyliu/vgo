//! Machinery shared by every generator: how a game is set up, sampled and
//! labelled, independent of how results are batched into files.
//!
//! This lived inside `generate_demo` while there was one generator. A second one
//! reimplementing any of it -- which board a game is played on, what komi
//! balances it, how a search becomes a policy target -- would be a second
//! definition of the training data the moment either changed. The rules already
//! carry that problem across `vgo-core` and the JS reference, with mutual "must
//! match" comments and a bug that had to be found twice.

use vgo_raster::{RasterConfig, action_pixel};
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
