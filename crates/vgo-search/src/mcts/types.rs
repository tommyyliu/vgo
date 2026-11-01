use vgo_core::Color;

use super::ordering::preferred_child_indices;
use crate::{Action, CandidateSource};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    pub simulations: u32,
    pub initial_candidates: usize,
    pub maximum_candidates: usize,
    pub widening_coefficient: f64,
    pub widening_exponent: f64,
    pub exploration: f64,
    /// Uniform mass mixed into the root's candidate proposal, in [0, 1).
    ///
    /// Candidates are *sampled* from the policy rather than enumerated, so a
    /// placement the network rates near zero is never proposed and never
    /// searched -- and the training target is the root visit distribution, so
    /// the next policy learns the same blind spot. That is a fixed point a
    /// self-play loop can sit in indefinitely.
    ///
    /// Mixing uniform mass into the proposal puts a floor under the rate at
    /// which underrated moves get looked at, which is what Dirichlet noise does
    /// for AlphaZero. It is applied at the root only, for the same reason
    /// AlphaZero does: that is where the target comes from, and noising deeper
    /// nodes would weaken the search without diversifying anything learned.
    ///
    /// `beta` is recomputed from the mixture, so the importance correction the
    /// policy target relies on stays exact.
    ///
    /// Zero disables it, which is the default: this changes what gets explored
    /// and so belongs to a run's identity rather than arriving silently.
    pub root_exploration_noise: f64,
    pub maximum_depth: u32,
    /// Coarse pool factor for coarse->fine candidate sampling. When > 0 and the
    /// policy exposes a fine grid, candidates are drawn from the net's own map
    /// instead of the legacy quasi-random sequence. 0 keeps the legacy behaviour.
    pub coarse_pool: usize,
    /// Softmax temperature applied to root visit counts when choosing the played
    /// move: the move is drawn with probability proportional to
    /// `visits^(1 / temperature)`. Zero is deterministic argmax, which is what
    /// arenas and any promotion-grade measurement want. Self-play generation
    /// wants a positive value for the opening plies so the same board does not
    /// always produce the same game. See `temperature_plies`.
    pub temperature: f64,
    /// Number of opening plies over which `temperature` applies. From this ply
    /// onward selection is deterministic argmax regardless of `temperature`.
    /// Sampling late endgame moves throws away won positions for no diversity
    /// benefit, because by then the visit distribution is what we want to trust.
    pub temperature_plies: u32,
    /// Number of leaves to collect and evaluate together per simulation round.
    /// One keeps the sequential path: descend, evaluate, back up. Larger values
    /// let a single game keep that many evaluations in flight, which is what
    /// fills inference batches when few games run concurrently. Virtual loss
    /// steers the concurrent descents apart. Values above one change which nodes
    /// get explored, so arenas comparing two models should use the same setting.
    pub leaf_batch: usize,
}

impl SearchConfig {
    #[must_use]
    pub const fn canary(simulations: u32) -> Self {
        Self {
            simulations,
            initial_candidates: 4,
            maximum_candidates: 96,
            widening_coefficient: 2.0,
            widening_exponent: 0.5,
            exploration: 1.5,
            root_exploration_noise: 0.0,
            maximum_depth: 64,
            coarse_pool: 0,
            temperature: 0.0,
            temperature_plies: 0,
            leaf_batch: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub simulations: u32,
    pub evaluations: u64,
    pub expanded_nodes: u64,
    pub generated_candidates: u64,
    pub terminal_leaves: u64,
    pub depth_limited_leaves: u64,
    pub maximum_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChildSummary {
    pub action: Action,
    pub source: CandidateSource,
    pub prior: f64,
    pub visits: u32,
    pub black_value: f64,
    /// Number of IID coarse-to-fine proposal draws that selected this placement.
    /// Legacy candidates and the deterministically enumerated pass action use 0.
    pub proposal_count: u32,
    /// Coarse->fine sampling probability beta = P_coarse * P_fine for this
    /// candidate, or None for legacy (quasi-random) candidates and pass. Used by
    /// training for the Sampled-AlphaZero importance correction on the target.
    pub beta: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub action: Action,
    pub children: Vec<ChildSummary>,
    pub stats: SearchStats,
    /// Child indices in the order the caller should try them. Under deterministic
    /// selection this is visit-count order; under a positive temperature the head
    /// of the list is drawn from `visits^(1 / temperature)` instead. Held as an
    /// order rather than a single choice so a caller that cannot play the first
    /// action falls back through the same ranking the selection policy produced.
    pub(super) order: Vec<usize>,
}

impl SearchResult {
    /// Build a result whose selection order is deterministic visit order. For
    /// callers (mainly tests) that assemble a result directly rather than by
    /// searching; `search_at_ply` sets the order from the selection policy.
    #[must_use]
    pub fn from_children(
        action: Action,
        children: Vec<ChildSummary>,
        stats: SearchStats,
        to_move: Color,
    ) -> Self {
        let order = preferred_child_indices(&children, to_move);
        Self {
            action,
            children,
            stats,
            order,
        }
    }

    /// Root evaluation from Black's perspective: the visit-weighted mean of the
    /// children's values.
    ///
    /// Each child's `black_value` is the mean outcome backed up through it, so
    /// weighting by visits reconstructs what the root's own accumulator holds.
    /// Callers use this to judge whether a game is decided; it is deliberately
    /// not the raw network head, which has not been corrected by search.
    #[must_use]
    pub fn root_black_value(&self) -> f64 {
        let visits: u32 = self.children.iter().map(|child| child.visits).sum();
        if visits == 0 {
            return 0.0;
        }
        self.children
            .iter()
            .map(|child| child.black_value * f64::from(child.visits))
            .sum::<f64>()
            / f64::from(visits)
    }

    /// Actions in selection order. `to_move` is retained for API compatibility;
    /// the order is already resolved from the searching player's perspective.
    #[must_use]
    pub fn actions_by_preference(&self, _to_move: Color) -> Vec<Action> {
        self.order
            .iter()
            .map(|&index| self.children[index].action)
            .collect()
    }
}
