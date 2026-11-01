use vgo_core::{Analysis, Color, Phase, Position};

use super::{SearchConfig, SearchStats};
use crate::{
    Action, Candidate, CandidateSequence, CandidateSource, Evaluation, EvaluationError, Evaluator,
    FineGrid,
};

/// In [`Node::child_cells`], a child with no cell, which is what a pass is.
const NO_CELL: u32 = u32::MAX;

pub(super) struct Child {
    pub(super) candidate: Candidate,
    /// The rules refused this move, so it is not selectable.
    ///
    /// Marked rather than removed. `Descent` records a path as child *indices*
    /// and `back_up` replays it later, so with leaf batching several paths are
    /// outstanding at once -- removing a child shifts every index recorded after
    /// it and the replay walks off the end of a shorter vector. Marking keeps
    /// indices stable for the life of the node.
    pub(super) refused: bool,
    pub(super) policy_logit: f64,
    pub(super) prior: f64,
    /// Sampling probability beta = P_coarse * P_fine for coarse->fine candidates,
    /// or None for legacy (quasi-random) candidates. Recorded for the
    /// Sampled-AlphaZero importance correction on the policy target.
    pub(super) beta: Option<f64>,
    /// IID proposal multiplicity for coarse-to-fine placement candidates.
    /// Legacy candidates and pass use 0.
    pub(super) proposal_count: u32,
    pub(super) visits: u32,
    /// Pending descents that have selected this child but whose evaluation has
    /// not returned yet. Deliberately separate from `visits` rather than folded
    /// into it: the progressive-widening budget and the replay visit target both
    /// read `visits` and must see only completed simulations, while PUCT must see
    /// the in-flight ones so concurrent descents spread out instead of stacking
    /// on the same branch.
    pub(super) virtual_visits: u32,
    pub(super) black_value_sum: f64,
    pub(super) node: Option<Box<Node>>,
}

impl Child {
    pub(super) fn black_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.black_value_sum / f64::from(self.visits)
        }
    }
}

pub(crate) struct Node {
    pub(super) position: Position,
    pub(super) visits: u32,
    /// In-flight descents through this node; see `Child::virtual_visits`.
    pub(super) virtual_visits: u32,
    pub(super) children: Vec<Child>,
    /// Each child's fine-grid cell, or [`NO_CELL`], in `children` order. Kept as
    /// integers because a sampled move is checked against every child, on nodes
    /// with hundreds of them, and comparing points cost 10% of all generation.
    child_cells: Vec<u32>,
    /// The legacy sequence is comparatively expensive to construct because it
    /// enumerates legal vertices. Dense spatial policies never use it, so build
    /// it only if coarse-to-fine sampling is unavailable.
    pub(super) candidates: Option<CandidateSequence>,
    pub(super) candidates_exhausted: bool,
    pub(super) terminal_black_value: Option<f64>,
    pub(super) evaluation: Option<Evaluation>,
    pub(super) match_seed: u64,
    /// Counter-based RNG state for coarse-to-fine candidate sampling, seeded from
    /// the match seed and this node's position so sampling is deterministic per
    /// (match, position) yet advances across widening calls.
    pub(super) sample_rng: u64,
    /// Cumulative number of coarse-to-fine placement proposals drawn at this node.
    pub(super) proposal_draws: usize,
    /// Highest visit-count coarse proposal budget already serviced. `None` means
    /// the policy has not yet confirmed that it exposes a spatial fine grid.
    pub(super) coarse_budget: Option<usize>,
    /// Building a fine grid computes legality for every policy cell. Cache both
    /// success and absence so progressive widening never repeats that work.
    pub(super) fine_grid: FineGridCache,
    /// Whether this node is the search root. Only the root takes exploration
    /// noise, so the flag has to survive being moved into the tree.
    pub(super) is_root: bool,
}

pub(super) enum FineGridCache {
    Uninitialized,
    Unavailable,
    Ready(FineGrid),
}

impl Node {
    pub(crate) fn memory_usage(&self) -> crate::TreeMemory {
        fn visit(node: &Node, counter: &mut crate::memory::MemoryCounter) {
            counter.report.nodes += 1;
            counter.report.node_bytes += size_of::<Node>();
            counter.report.edge_capacity_bytes += node.children.capacity() * size_of::<Child>()
                + node.child_cells.capacity() * size_of::<u32>();
            counter.report.position_bytes += std::mem::size_of_val(node.position.stones());
            if let Some(candidates) = &node.candidates {
                counter.report.sampling_bytes +=
                    size_of::<CandidateSequence>() + candidates.payload_bytes();
            }
            if let Some(evaluation) = &node.evaluation {
                evaluation.account_memory(counter);
            }
            if let FineGridCache::Ready(grid) = &node.fine_grid {
                grid.account_memory(counter);
            }
            for child in &node.children {
                if let Some(node) = &child.node {
                    visit(node, counter);
                }
            }
        }
        let mut counter = crate::memory::MemoryCounter::default();
        visit(self, &mut counter);
        counter.report
    }
    /// Build a node whose evaluation was already computed elsewhere.
    ///
    /// The batched path evaluates a round of leaves together, so by the time a
    /// node is constructed its evaluation exists. Finished positions are built
    /// with `terminal` and never reach an evaluator.
    pub(crate) fn from_evaluation(
        position: Position,
        evaluation: Evaluation,
        match_seed: u64,
    ) -> Self {
        debug_assert_eq!(position.phase(), Phase::Playing);
        let sample_rng =
            crate::candidates::splitmix64(match_seed ^ crate::candidates::position_hash(&position));
        Self {
            position,
            visits: 0,
            virtual_visits: 0,
            children: Vec::new(),
            child_cells: Vec::new(),
            candidates: None,
            candidates_exhausted: false,
            evaluation: Some(evaluation),
            terminal_black_value: None,
            match_seed,
            sample_rng,
            proposal_draws: 0,
            coarse_budget: None,
            fine_grid: FineGridCache::Uninitialized,
            is_root: false,
        }
    }

    pub(super) fn terminal(position: Position, black_value: f64, match_seed: u64) -> Self {
        debug_assert_eq!(position.phase(), Phase::Finished);
        let sample_rng =
            crate::candidates::splitmix64(match_seed ^ crate::candidates::position_hash(&position));
        Self {
            position,
            visits: 0,
            virtual_visits: 0,
            children: Vec::new(),
            child_cells: Vec::new(),
            candidates: None,
            candidates_exhausted: true,
            terminal_black_value: Some(black_value),
            evaluation: None,
            match_seed,
            sample_rng,
            proposal_draws: 0,
            coarse_budget: None,
            fine_grid: FineGridCache::Unavailable,
            is_root: false,
        }
    }

    pub(super) fn new(
        position: Position,
        analysis: Option<&Analysis>,
        match_seed: u64,
        evaluator: &dyn Evaluator,
        stats: &mut SearchStats,
    ) -> Result<Self, EvaluationError> {
        let terminal_black_value = if position.phase() == Phase::Finished {
            Some(
                analysis
                    .map_or_else(|| Analysis::new(&position).outcome, |value| value.outcome)
                    .black_utility(),
            )
        } else {
            None
        };
        if let Some(black_value) = terminal_black_value {
            return Ok(Self::terminal(position, black_value, match_seed));
        }
        stats.evaluations += 1;
        let evaluation = evaluator.evaluate(&position)?;
        Ok(Self::from_evaluation(position, evaluation, match_seed))
    }

    /// Mark this node as the search root, so it takes exploration noise.
    ///
    /// Set by whoever builds the root rather than inferred, because a node has
    /// no view of the tree it will be placed in.
    pub(crate) fn mark_root(&mut self) {
        self.is_root = true;
    }

    pub(crate) fn visits(&self) -> u32 {
        self.visits
    }

    pub(crate) fn matches_position(&self, position: &Position) -> bool {
        self.position == *position
    }

    /// Drop siblings and keep only the child for the committed move. Exact
    /// Position equality includes captures, pass count, rules, komi and mover.
    pub(crate) fn take_child(&mut self, action: Action, position: &Position) -> Option<Box<Node>> {
        self.children
            .iter_mut()
            .find(|child| child.candidate.action == action)
            .and_then(|child| child.node.take())
            .filter(|node| node.matches_position(position))
    }

    pub(crate) fn black_evaluation(&self) -> f64 {
        let current_value = self
            .evaluation
            .as_ref()
            .expect("nonterminal nodes are evaluated")
            .current_value;
        if self.position.to_move() == Color::Black {
            current_value
        } else {
            -current_value
        }
    }

    pub(super) fn widen(&mut self, config: SearchConfig, stats: &mut SearchStats) {
        let progressive = (config.widening_coefficient
            * f64::from(self.visits + 1).powf(config.widening_exponent))
        .ceil() as usize;
        let desired = config
            .initial_candidates
            .max(progressive)
            .min(config.maximum_candidates);

        if config.coarse_pool > 0 {
            if self.coarse_budget.is_some_and(|budget| budget >= desired) {
                return;
            }
            if matches!(self.fine_grid, FineGridCache::Uninitialized) {
                self.fine_grid = self
                    .evaluation
                    .as_ref()
                    .expect("playable nodes are evaluated")
                    .fine_grid(&self.position, config.coarse_pool)
                    .map_or(FineGridCache::Unavailable, FineGridCache::Ready);
            }
            if matches!(self.fine_grid, FineGridCache::Ready(_)) {
                // Root only. The training target is the root visit
                // distribution, so that is where diversifying the proposal
                // changes what gets learned; noising deeper nodes would spend
                // simulations on worse moves without widening anything the
                // loop trains on.
                let noise = if self.is_root {
                    config.root_exploration_noise.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.widen_coarse_fine(desired, noise, stats);
                self.coarse_budget = Some(desired);
                normalize_priors(&mut self.children);
                return;
            }
            // Policy is not spatial (e.g. naive): fall through to the legacy path.
        }

        if self.candidates.is_none() {
            self.candidates = Some(CandidateSequence::new(&self.position, self.match_seed));
        }
        while self.children.len() < desired && !self.candidates_exhausted {
            if let Some(candidate) = self
                .candidates
                .as_mut()
                .expect("legacy candidate sequence initialized")
                .next_candidate()
            {
                let policy_logit = self
                    .evaluation
                    .as_ref()
                    .expect("playable nodes are evaluated")
                    .policy_logit(candidate.action);
                self.children.push(Child {
                    refused: false,
                    candidate,
                    policy_logit,
                    prior: 0.0,
                    beta: None,
                    proposal_count: 0,
                    visits: 0,
                    virtual_visits: 0,
                    black_value_sum: 0.0,
                    node: None,
                });
                stats.generated_candidates += 1;
            } else {
                self.candidates_exhausted = true;
            }
        }
        normalize_priors(&mut self.children);
    }

    /// Extend the visit-count progressive-widening budget with IID coarse-to-fine
    /// draws from the net's policy map. Repeated cells increment proposal
    /// multiplicity on the existing child rather than triggering a retry. Each
    /// child's exact sampling probability beta is recorded, and pass is always
    /// available.
    /// `noise` is uniform mass mixed into the proposal; see
    /// [`SearchConfig::root_exploration_noise`]. Passed in rather than read
    /// from the config here because only the root takes it.
    pub(super) fn widen_coarse_fine(
        &mut self,
        desired: usize,
        noise: f64,
        stats: &mut SearchStats,
    ) {
        let FineGridCache::Ready(grid) = &self.fine_grid else {
            unreachable!("coarse-to-fine widening requires a cached grid");
        };
        // Ensure Pass is a candidate exactly once (it is not part of the grid).
        if !self
            .children
            .iter()
            .any(|child| matches!(child.candidate.action, Action::Pass))
        {
            let policy_logit = self
                .evaluation
                .as_ref()
                .expect("playable nodes are evaluated")
                .policy_logit(Action::Pass);
            self.children.push(Child {
                refused: false,
                candidate: Candidate {
                    action: Action::Pass,
                    source: CandidateSource::Pass,
                },
                policy_logit,
                prior: 0.0,
                beta: None,
                proposal_count: 0,
                visits: 0,
                virtual_visits: 0,
                black_value_sum: 0.0,
                node: None,
            });
        }

        let draw_count = desired.saturating_sub(self.proposal_draws);
        if draw_count == 0 {
            return;
        }
        let mut rng = self.sample_rng;
        let mut next = || {
            rng = crate::candidates::splitmix64(rng);
            crate::candidates::unit_f64(rng)
        };
        let samples =
            crate::coarse_fine::sample_candidates_noisy(grid, draw_count, noise, &mut next);
        self.sample_rng = rng;
        let known = self.child_cells.len();
        self.child_cells
            .extend(
                self.children[known..]
                    .iter()
                    .map(|child| match child.candidate.action {
                        Action::Place(point) => grid.cell_index(point),
                        Action::Pass => NO_CELL,
                    }),
            );
        for sample in samples {
            self.proposal_draws += 1;
            let action = Action::Place(sample.point);
            let cell = grid.cell_index(sample.point);
            if let Some(index) = self.child_cells.iter().position(|&known| known == cell) {
                self.children[index].proposal_count += 1;
                continue;
            }
            self.child_cells.push(cell);
            let policy_logit = self
                .evaluation
                .as_ref()
                .expect("playable nodes are evaluated")
                .policy_logit(action);
            self.children.push(Child {
                refused: false,
                candidate: Candidate {
                    action,
                    source: CandidateSource::AreaSequence,
                },
                policy_logit,
                prior: 0.0,
                beta: Some(sample.beta),
                proposal_count: 1,
                visits: 0,
                virtual_visits: 0,
                black_value_sum: 0.0,
                node: None,
            });
            stats.generated_candidates += 1;
        }
    }

    pub(super) fn select_child(&self, config: SearchConfig) -> usize {
        let parent_visits = f64::from((self.visits + self.virtual_visits).max(1)).sqrt();
        let perspective = if self.position.to_move() == Color::Black {
            1.0
        } else {
            -1.0
        };
        self.children
            .iter()
            .enumerate()
            .filter(|(_, child)| !child.refused)
            .map(|(index, child)| {
                // Virtual loss counts an in-flight descent as a visit that scored
                // the worst possible value for the player to move. That both
                // shrinks the exploration bonus and drags the mean down, so a
                // second concurrent descent prefers a different branch instead of
                // piling onto the same one. It is removed when the real value
                // arrives, so it biases only the order of exploration, never the
                // statistics that survive the search.
                let pending = f64::from(child.virtual_visits);
                let total = f64::from(child.visits) + pending;
                let exploitation = if total == 0.0 {
                    0.0
                } else {
                    // Pending descents contribute -1 from the mover's perspective.
                    (perspective * child.black_value_sum - pending) / total
                };
                let exploration = config.exploration * child.prior * parent_visits / (1.0 + total);
                (index, exploitation + exploration)
            })
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .expect("every playable node has pass as a candidate")
    }
}

/// Select a child the rules will actually allow.
///
/// Under [`Ruleset::Official`](vgo_core::Ruleset::Official) a placement that
/// takes only the mover's own stones is illegal, and nothing short of resolving
/// the move can tell -- so candidate generation cannot filter these out and the
/// check has to happen here, when a child is first chosen.
///
/// It costs nothing. The search was going to resolve that move anyway to build
/// the child position, and that already happens before the network sees
/// anything, so a refused candidate never reaches inference. A child that is
/// already expanded skips the check, because the rules accepted it when it was
/// expanded. Under `Vgo` `try_apply` never refuses, so this is one branch.
///
/// A refused child is **marked, not removed** -- see [`Child::refused`]. It is
/// then never selected again, and never appears in the assembled result.
///
/// Terminates because the pass candidate is always present and `pass` cannot
/// self-capture, so at worst every placement is marked and the pass is chosen.
pub(super) fn select_playable_child(
    node: &mut Node,
    config: SearchConfig,
) -> Option<(usize, Option<vgo_core::MoveResult>)> {
    while node.children.iter().any(|child| !child.refused) {
        let index = node.select_child(config);
        // Already expanded, so the rules accepted it when it was expanded.
        if node.children[index].node.is_some() {
            return Some((index, None));
        }
        if let Some(transition) = node.children[index]
            .candidate
            .action
            .try_apply(&node.position)
        {
            return Some((index, Some(transition)));
        }
        node.children[index].refused = true;
    }
    None
}

pub(super) fn normalize_priors(children: &mut [Child]) {
    let maximum = children
        .iter()
        .map(|child| child.policy_logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for child in children.iter_mut() {
        child.prior = (child.policy_logit - maximum).exp();
        total += child.prior;
    }
    for child in children {
        child.prior /= total;
    }
}
