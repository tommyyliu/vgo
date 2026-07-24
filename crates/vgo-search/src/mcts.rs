use vgo_core::{Analysis, Color, Phase, Position};

use crate::{
    Action, Candidate, CandidateSequence, CandidateSource, Evaluation, EvaluationError, Evaluator,
    FineGrid, NaiveEvaluator,
};

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub simulations: u32,
    pub initial_candidates: usize,
    pub maximum_candidates: usize,
    pub widening_coefficient: f64,
    pub widening_exponent: f64,
    pub exploration: f64,
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
            maximum_depth: 64,
            coarse_pool: 0,
            temperature: 0.0,
            temperature_plies: 0,
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
    /// of the list is drawn from `visits^(1 / temperature)` instead. Held so that
    /// a caller rejecting the first action (repetition avoidance) falls back
    /// through the same order the selection policy produced.
    order: Vec<usize>,
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

struct Child {
    candidate: Candidate,
    policy_logit: f64,
    prior: f64,
    /// Sampling probability beta = P_coarse * P_fine for coarse->fine candidates,
    /// or None for legacy (quasi-random) candidates. Recorded for the
    /// Sampled-AlphaZero importance correction on the policy target.
    beta: Option<f64>,
    /// IID proposal multiplicity for coarse-to-fine placement candidates.
    /// Legacy candidates and pass use 0.
    proposal_count: u32,
    visits: u32,
    black_value_sum: f64,
    node: Option<Box<Node>>,
}

impl Child {
    fn black_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.black_value_sum / f64::from(self.visits)
        }
    }
}

struct Node {
    position: Position,
    visits: u32,
    children: Vec<Child>,
    candidates: CandidateSequence,
    candidates_exhausted: bool,
    terminal_black_value: Option<f64>,
    evaluation: Option<Evaluation>,
    /// Counter-based RNG state for coarse-to-fine candidate sampling, seeded from
    /// the match seed and this node's position so sampling is deterministic per
    /// (match, position) yet advances across widening calls.
    sample_rng: u64,
    /// Cumulative number of coarse-to-fine placement proposals drawn at this node.
    proposal_draws: usize,
    /// Highest visit-count coarse proposal budget already serviced. `None` means
    /// the policy has not yet confirmed that it exposes a spatial fine grid.
    coarse_budget: Option<usize>,
}

impl Node {
    fn new(
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
        let evaluation = if terminal_black_value.is_some() {
            None
        } else {
            stats.evaluations += 1;
            Some(evaluator.evaluate(&position)?)
        };
        let candidates = CandidateSequence::new(&position, match_seed);
        let sample_rng =
            crate::candidates::splitmix64(match_seed ^ crate::candidates::position_hash(&position));
        Ok(Self {
            position,
            visits: 0,
            children: Vec::new(),
            candidates,
            candidates_exhausted: false,
            terminal_black_value,
            evaluation,
            sample_rng,
            proposal_draws: 0,
            coarse_budget: None,
        })
    }

    fn black_evaluation(&self) -> f64 {
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

    fn widen(&mut self, config: SearchConfig, stats: &mut SearchStats) {
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
            if let Some(grid) = self
                .evaluation
                .as_ref()
                .expect("playable nodes are evaluated")
                .fine_grid(&self.position, config.coarse_pool)
            {
                self.widen_coarse_fine(&grid, desired, stats);
                self.coarse_budget = Some(desired);
                normalize_priors(&mut self.children);
                return;
            }
            // Policy is not spatial (e.g. naive): fall through to the legacy path.
        }

        while self.children.len() < desired && !self.candidates_exhausted {
            if let Some(candidate) = self.candidates.next_candidate() {
                let policy_logit = self
                    .evaluation
                    .as_ref()
                    .expect("playable nodes are evaluated")
                    .policy_logit(candidate.action);
                self.children.push(Child {
                    candidate,
                    policy_logit,
                    prior: 0.0,
                    beta: None,
                    proposal_count: 0,
                    visits: 0,
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
    fn widen_coarse_fine(&mut self, grid: &FineGrid, desired: usize, stats: &mut SearchStats) {
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
                candidate: Candidate {
                    action: Action::Pass,
                    source: CandidateSource::Pass,
                },
                policy_logit,
                prior: 0.0,
                beta: None,
                proposal_count: 0,
                visits: 0,
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
        let samples = crate::coarse_fine::sample_candidates(grid, draw_count, &mut next);
        self.sample_rng = rng;
        for sample in samples {
            self.proposal_draws += 1;
            let action = Action::Place(sample.point);
            if let Some(existing) =
                self.children
                    .iter_mut()
                    .find(|child| match child.candidate.action {
                        Action::Place(existing) => grid.same_cell(existing, sample.point),
                        Action::Pass => false,
                    })
            {
                existing.proposal_count += 1;
                continue;
            }
            let policy_logit = self
                .evaluation
                .as_ref()
                .expect("playable nodes are evaluated")
                .policy_logit(action);
            self.children.push(Child {
                candidate: Candidate {
                    action,
                    source: CandidateSource::AreaSequence,
                },
                policy_logit,
                prior: 0.0,
                beta: Some(sample.beta),
                proposal_count: 1,
                visits: 0,
                black_value_sum: 0.0,
                node: None,
            });
            stats.generated_candidates += 1;
        }
    }

    fn select_child(&self, config: SearchConfig) -> usize {
        let parent_visits = f64::from(self.visits.max(1)).sqrt();
        let perspective = if self.position.to_move() == Color::Black {
            1.0
        } else {
            -1.0
        };
        self.children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                let exploitation = perspective * child.black_value();
                let exploration = config.exploration * child.prior * parent_visits
                    / (1.0 + f64::from(child.visits));
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

fn normalize_priors(children: &mut [Child]) {
    let maximum = children
        .iter()
        .map(|child| child.policy_logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = children
        .iter()
        .map(|child| (child.policy_logit - maximum).exp())
        .sum();
    for child in children {
        child.prior = (child.policy_logit - maximum).exp() / total;
    }
}

fn preferred_child_indices(children: &[ChildSummary], to_move: Color) -> Vec<usize> {
    let perspective = if to_move == Color::Black { 1.0 } else { -1.0 };
    let mut indices = (0..children.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left = *left;
        let right = *right;
        children[right]
            .visits
            .cmp(&children[left].visits)
            .then_with(|| {
                (perspective * children[right].black_value)
                    .total_cmp(&(perspective * children[left].black_value))
            })
            .then_with(|| left.cmp(&right))
    });
    indices
}

/// Order children by sampling without replacement from `visits^(1 / temperature)`.
///
/// Only children with visits participate in the draw; unvisited children keep
/// their deterministic ordering at the tail, so a caller falling back through the
/// list for repetition avoidance still degrades to visit order rather than to
/// noise. `temperature <= 0` never reaches here.
fn sampled_child_indices(
    children: &[ChildSummary],
    to_move: Color,
    temperature: f64,
    rng_state: u64,
) -> Vec<usize> {
    let deterministic = preferred_child_indices(children, to_move);
    let exponent = 1.0 / temperature;
    // Scale by the maximum visit count before exponentiating: visits^(1/t) with a
    // small temperature overflows f64 for even modest visit counts (128^100), and
    // the distribution is unchanged by dividing through by the max first.
    let maximum = children
        .iter()
        .map(|child| child.visits)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut pool = deterministic
        .iter()
        .copied()
        .filter(|&index| children[index].visits > 0)
        .map(|index| {
            let share = f64::from(children[index].visits) / f64::from(maximum);
            (index, share.powf(exponent))
        })
        .collect::<Vec<_>>();

    let mut rng = rng_state;
    let mut next = || {
        rng = crate::candidates::splitmix64(rng);
        crate::candidates::unit_f64(rng)
    };

    let mut order = Vec::with_capacity(children.len());
    while !pool.is_empty() {
        let total: f64 = pool.iter().map(|&(_, weight)| weight).sum();
        if !(total > 0.0) || !total.is_finite() {
            // Degenerate weights (all zero, or non-finite): fall back to the
            // deterministic order for whatever remains.
            order.extend(pool.iter().map(|&(index, _)| index));
            break;
        }
        let mut target = next() * total;
        let mut chosen = pool.len() - 1;
        for (position, &(_, weight)) in pool.iter().enumerate() {
            target -= weight;
            if target < 0.0 {
                chosen = position;
                break;
            }
        }
        order.push(pool.remove(chosen).0);
    }
    // Unvisited children, in deterministic order, after every visited one.
    order.extend(
        deterministic
            .into_iter()
            .filter(|&index| children[index].visits == 0),
    );
    order
}

fn simulate(
    node: &mut Node,
    config: SearchConfig,
    match_seed: u64,
    evaluator: &dyn Evaluator,
    depth: u32,
    stats: &mut SearchStats,
) -> Result<f64, EvaluationError> {
    stats.maximum_depth = stats.maximum_depth.max(depth);
    if let Some(value) = node.terminal_black_value {
        stats.terminal_leaves += 1;
        node.visits += 1;
        return Ok(value);
    }
    if depth >= config.maximum_depth {
        stats.depth_limited_leaves += 1;
        node.visits += 1;
        return Ok(node.black_evaluation());
    }

    node.widen(config, stats);
    let child_index = node.select_child(config);
    let child = &mut node.children[child_index];
    let black_value = if let Some(child_node) = child.node.as_mut() {
        simulate(child_node, config, match_seed, evaluator, depth + 1, stats)?
    } else {
        let transition = child.candidate.action.apply(&node.position);
        let child_node = Node::new(
            transition.position,
            Some(&transition.analysis),
            match_seed,
            evaluator,
            stats,
        )?;
        let value = child_node
            .terminal_black_value
            .unwrap_or_else(|| child_node.black_evaluation());
        if child_node.terminal_black_value.is_some() {
            stats.terminal_leaves += 1;
        }
        stats.maximum_depth = stats.maximum_depth.max(depth + 1);
        stats.expanded_nodes += 1;
        child.node = Some(Box::new(child_node));
        value
    };
    child.visits += 1;
    child.black_value_sum += black_value;
    node.visits += 1;
    Ok(black_value)
}

#[must_use]
pub fn search(position: &Position, config: SearchConfig, match_seed: u64) -> SearchResult {
    search_with_evaluator(position, config, match_seed, &NaiveEvaluator)
        .expect("the in-process naive evaluator is infallible")
}

pub fn search_with_evaluator(
    position: &Position,
    config: SearchConfig,
    match_seed: u64,
    evaluator: &dyn Evaluator,
) -> Result<SearchResult, EvaluationError> {
    search_at_ply(position, config, match_seed, evaluator, 0)
}

/// Search a position that occurs at `ply` of the game.
///
/// Identical to [`search_with_evaluator`] except that the ply decides whether
/// `config.temperature` applies: temperature is used while `ply <
/// config.temperature_plies` and selection is deterministic afterwards. Callers
/// that do not track a ply (arenas, one-off analysis) get deterministic
/// selection, which is what a promotion-grade measurement requires.
pub fn search_at_ply(
    position: &Position,
    config: SearchConfig,
    match_seed: u64,
    evaluator: &dyn Evaluator,
    ply: u32,
) -> Result<SearchResult, EvaluationError> {
    assert_eq!(
        position.phase(),
        Phase::Playing,
        "cannot search a finished position"
    );
    let mut stats = SearchStats::default();
    let mut root = Node::new(position.clone(), None, match_seed, evaluator, &mut stats)?;
    for _ in 0..config.simulations {
        simulate(&mut root, config, match_seed, evaluator, 0, &mut stats)?;
        stats.simulations += 1;
    }
    let children = root
        .children
        .iter()
        .map(|child| ChildSummary {
            action: child.candidate.action,
            source: child.candidate.source,
            prior: child.prior,
            visits: child.visits,
            black_value: child.black_value(),
            proposal_count: child.proposal_count,
            beta: child.beta,
        })
        .collect::<Vec<_>>();
    let sampling = config.temperature > 0.0 && ply < config.temperature_plies;
    let order = if sampling {
        // Seed from the match, the position, and the ply so the draw is
        // reproducible for a given (match, position, ply) yet differs from the
        // candidate-sampling stream at the same node.
        let seed = crate::candidates::splitmix64(
            match_seed
                ^ crate::candidates::position_hash(position)
                ^ u64::from(ply).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        sampled_child_indices(&children, position.to_move(), config.temperature, seed)
    } else {
        preferred_child_indices(&children, position.to_move())
    };
    let best = order
        .first()
        .map(|index| children[*index].action)
        .expect("search produces at least the pass child");
    Ok(SearchResult {
        action: best,
        children,
        stats,
        order,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use vgo_core::{Color, Position};

    use crate::{Action, Evaluation, EvaluationError, Evaluator, Policy};

    use super::{
        ChildSummary, Node, SearchConfig, SearchStats, sampled_child_indices, search,
        search_at_ply, search_with_evaluator,
    };

    fn child(action: Action, visits: u32) -> ChildSummary {
        ChildSummary {
            action,
            source: crate::CandidateSource::AreaSequence,
            prior: 0.0,
            visits,
            black_value: 0.0,
            proposal_count: 0,
            beta: None,
        }
    }

    fn place(x: f64) -> Action {
        Action::Place(vgo_core::Point::new(x, 0.5))
    }

    /// Sampling must reach the low-visit move sometimes and the high-visit move
    /// usually. A distribution that always returns argmax is the bug we are
    /// fixing; one that ignores visits entirely is equally wrong.
    #[test]
    fn temperature_sampling_follows_visit_counts() {
        let children = vec![child(place(0.1), 90), child(place(0.2), 10)];
        let mut first_is_top = 0_u32;
        let trials = 400_u32;
        for seed in 0..u64::from(trials) {
            let order = sampled_child_indices(&children, Color::Black, 1.0, seed);
            assert_eq!(order.len(), 2, "every visited child must be ordered");
            if order[0] == 0 {
                first_is_top += 1;
            }
        }
        let share = f64::from(first_is_top) / f64::from(trials);
        assert!(
            (0.80..0.97).contains(&share),
            "90/10 visits should pick the leader ~90% of the time, got {share:.3}"
        );
    }

    /// A low temperature must concentrate on the visit leader without becoming
    /// literally deterministic, and must not overflow on large visit counts.
    #[test]
    fn low_temperature_concentrates_without_overflow() {
        let children = vec![child(place(0.1), 5_000), child(place(0.2), 2_500)];
        let mut leader = 0;
        for seed in 0..200 {
            let order = sampled_child_indices(&children, Color::Black, 0.1, seed);
            assert_eq!(order.len(), 2);
            if order[0] == 0 {
                leader += 1;
            }
        }
        assert!(leader >= 195, "2:1 visits at t=0.1 should be near-certain");
    }

    /// Unvisited children must stay behind every visited one so that repetition
    /// fallback degrades to visit order rather than to noise.
    #[test]
    fn unvisited_children_sort_last() {
        let children = vec![child(place(0.1), 0), child(place(0.2), 7), child(place(0.3), 3)];
        for seed in 0..50 {
            let order = sampled_child_indices(&children, Color::Black, 1.0, seed);
            assert_eq!(order.len(), 3);
            assert_eq!(order[2], 0, "the zero-visit child must be last");
        }
    }

    /// Same seed, same order: replay and debugging depend on it.
    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let children = vec![child(place(0.1), 40), child(place(0.2), 30), child(place(0.3), 20)];
        let first = sampled_child_indices(&children, Color::Black, 1.0, 12_345);
        let again = sampled_child_indices(&children, Color::Black, 1.0, 12_345);
        assert_eq!(first, again);
    }

    /// Temperature must not leak into arenas: past `temperature_plies`, and for
    /// the plain entry point, selection is deterministic.
    #[test]
    fn temperature_applies_only_within_the_opening() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let mut config = SearchConfig::canary(32);
        config.temperature = 1.5;
        config.temperature_plies = 4;

        let late = (0..6)
            .map(|_| {
                search_at_ply(&position, config, 7, &crate::NaiveEvaluator, 10)
                    .expect("naive search is infallible")
                    .action
            })
            .collect::<Vec<_>>();
        assert!(
            late.windows(2).all(|pair| pair[0] == pair[1]),
            "past temperature_plies selection must be deterministic"
        );

        let plain = (0..6)
            .map(|_| {
                search_with_evaluator(&position, config, 7, &crate::NaiveEvaluator)
                    .expect("naive search is infallible")
                    .action
            })
            .collect::<Vec<_>>();
        assert!(
            plain.windows(2).all(|pair| pair[0] == pair[1]),
            "search_with_evaluator must ignore temperature"
        );
    }

    struct FlatPolicy;

    impl Policy for FlatPolicy {
        fn logit(&self, _action: Action) -> f64 {
            0.0
        }
    }

    /// A spatial test policy backed by a uniform fine grid, so coarse->fine
    /// sampling has a map to draw from.
    struct GridPolicy {
        width: usize,
        height: usize,
    }

    impl Policy for GridPolicy {
        fn logit(&self, _action: Action) -> f64 {
            0.0
        }

        fn fine_grid(&self, position: &Position, coarse: usize) -> Option<crate::FineGrid> {
            Some(crate::FineGrid::build(
                position,
                self.width,
                self.height,
                coarse,
                |_, _| 0.0,
            ))
        }
    }

    struct GridEvaluator {
        width: usize,
        height: usize,
    }

    impl Evaluator for GridEvaluator {
        fn evaluate(&self, _position: &Position) -> Result<Evaluation, EvaluationError> {
            Ok(Evaluation::new(
                0.0,
                Box::new(GridPolicy {
                    width: self.width,
                    height: self.height,
                }),
            ))
        }
    }

    struct CountingGridPolicy {
        fine_grid_calls: Arc<AtomicUsize>,
    }

    impl Policy for CountingGridPolicy {
        fn logit(&self, _action: Action) -> f64 {
            0.0
        }

        fn fine_grid(&self, position: &Position, coarse: usize) -> Option<crate::FineGrid> {
            self.fine_grid_calls.fetch_add(1, Ordering::Relaxed);
            Some(crate::FineGrid::build(position, 8, 8, coarse, |_, _| 0.0))
        }
    }

    struct CountingGridEvaluator {
        fine_grid_calls: Arc<AtomicUsize>,
    }

    impl Evaluator for CountingGridEvaluator {
        fn evaluate(&self, _position: &Position) -> Result<Evaluation, EvaluationError> {
            Ok(Evaluation::new(
                0.0,
                Box::new(CountingGridPolicy {
                    fine_grid_calls: Arc::clone(&self.fine_grid_calls),
                }),
            ))
        }
    }

    struct ConstantEvaluator(f64);

    impl Evaluator for ConstantEvaluator {
        fn evaluate(&self, _position: &Position) -> Result<Evaluation, EvaluationError> {
            Ok(Evaluation::new(self.0, Box::new(FlatPolicy)))
        }
    }

    struct FailingEvaluator;

    impl Evaluator for FailingEvaluator {
        fn evaluate(&self, _position: &Position) -> Result<Evaluation, EvaluationError> {
            Err(EvaluationError::new("expected failure"))
        }
    }

    #[test]
    fn coarse_fine_duplicate_draws_increment_proposal_count() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let mut config = SearchConfig::canary(1);
        config.coarse_pool = 1;
        config.initial_candidates = 7;
        let evaluator = GridEvaluator {
            width: 1,
            height: 1,
        };

        let result = search_with_evaluator(&position, config, 7, &evaluator)
            .expect("coarse-fine search completes");
        let placements = result
            .children
            .iter()
            .filter(|child| matches!(child.action, Action::Place(_)))
            .collect::<Vec<_>>();

        assert_eq!(placements.len(), 1);
        assert_eq!(
            placements[0].proposal_count,
            config.initial_candidates as u32
        );
        assert_eq!(
            result
                .children
                .iter()
                .map(|child| child.proposal_count)
                .sum::<u32>(),
            config.initial_candidates as u32
        );
        assert!(
            result
                .children
                .iter()
                .filter(|child| matches!(child.action, Action::Pass))
                .all(|child| child.proposal_count == 0)
        );
    }

    #[test]
    fn coarse_fine_progressive_widening_can_add_later_candidates() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let evaluator = GridEvaluator {
            width: 32,
            height: 32,
        };
        let mut config = SearchConfig::canary(1);
        config.coarse_pool = 4;
        let mut stats = SearchStats::default();
        let mut node =
            Node::new(position, None, 11, &evaluator, &mut stats).expect("node evaluates");

        node.widen(config, &mut stats);
        let initial_actions = node
            .children
            .iter()
            .filter_map(|child| match child.candidate.action {
                Action::Place(point) => Some(point),
                Action::Pass => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(node.proposal_draws, 4);
        assert_eq!(
            node.children
                .iter()
                .map(|child| child.proposal_count)
                .sum::<u32>(),
            4
        );

        node.visits = 1_023;
        node.widen(config, &mut stats);
        let later_actions = node
            .children
            .iter()
            .filter_map(|child| match child.candidate.action {
                Action::Place(point) => Some(point),
                Action::Pass => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(node.proposal_draws, 64);
        assert_eq!(
            node.children
                .iter()
                .map(|child| child.proposal_count)
                .sum::<u32>(),
            64
        );
        assert!(later_actions.len() > initial_actions.len());
        assert!(
            initial_actions
                .iter()
                .all(|action| later_actions.contains(action))
        );
    }

    #[test]
    fn coarse_fine_grid_is_rebuilt_only_when_the_draw_budget_grows() {
        let fine_grid_calls = Arc::new(AtomicUsize::new(0));
        let evaluator = CountingGridEvaluator {
            fine_grid_calls: Arc::clone(&fine_grid_calls),
        };
        let mut config = SearchConfig::canary(1);
        config.coarse_pool = 2;
        let mut stats = SearchStats::default();
        let mut node = Node::new(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            None,
            13,
            &evaluator,
            &mut stats,
        )
        .expect("node evaluates");

        node.widen(config, &mut stats);
        assert_eq!(fine_grid_calls.load(Ordering::Relaxed), 1);
        assert_eq!(node.proposal_draws, 4);

        node.visits = 1;
        node.widen(config, &mut stats);
        assert_eq!(fine_grid_calls.load(Ordering::Relaxed), 1);
        assert_eq!(node.proposal_draws, 4);

        node.visits = 4;
        node.widen(config, &mut stats);
        assert_eq!(fine_grid_calls.load(Ordering::Relaxed), 2);
        assert_eq!(node.proposal_draws, 5);
    }

    #[test]
    fn coarse_fine_widening_draws_candidates_from_the_map() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let mut config = SearchConfig::canary(64);
        config.coarse_pool = 4; // enable coarse->fine sampling
        let evaluator = GridEvaluator {
            width: 32,
            height: 32,
        };
        let result = search_with_evaluator(&position, config, 7, &evaluator)
            .expect("coarse-fine search completes");
        assert_eq!(result.stats.simulations, 64);
        // It must produce real placement candidates plus pass, all legal.
        let placements = result
            .children
            .iter()
            .filter(|c| matches!(c.action, Action::Place(_)))
            .count();
        assert!(
            placements >= 2,
            "expected several placements, got {placements}"
        );
        assert!(
            result
                .children
                .iter()
                .any(|c| matches!(c.action, Action::Pass)),
            "pass must remain a candidate"
        );
        assert_eq!(
            result
                .children
                .iter()
                .map(|child| child.proposal_count)
                .sum::<u32>(),
            16
        );
        // The visit budget is honoured.
        let total: u32 = result.children.iter().map(|c| c.visits).sum();
        assert!(total > 0);
    }

    #[test]
    fn search_uses_the_requested_simulation_budget() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let result = search(&position, SearchConfig::canary(10), 3);
        assert_eq!(result.stats.simulations, 10);
        assert_eq!(
            result
                .children
                .iter()
                .map(|child| child.visits)
                .sum::<u32>(),
            10
        );
        assert_eq!(
            result.actions_by_preference(position.to_move())[0],
            result.action
        );
    }

    #[test]
    fn progressive_widening_adds_candidates_with_compute() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let small = search(&position, SearchConfig::canary(10), 11);
        let large = search(&position, SearchConfig::canary(1_000), 11);
        assert!(large.children.len() > small.children.len());
        let small_actions: Vec<_> = small.children.iter().map(|child| child.action).collect();
        let large_actions: Vec<_> = large.children.iter().map(|child| child.action).collect();
        assert_eq!(small_actions, large_actions[..small_actions.len()]);
    }

    #[test]
    fn leaf_values_are_converted_from_current_player_to_black() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let result = search_with_evaluator(
            &position,
            SearchConfig::canary(1),
            5,
            &ConstantEvaluator(0.75),
        )
        .expect("constant evaluator succeeds");
        assert_eq!(result.children[0].action, Action::Pass);
        assert_eq!(result.children[0].black_value, -0.75);
        assert_eq!(result.stats.evaluations, 2);
    }

    #[test]
    fn evaluator_errors_abort_search() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let error = search_with_evaluator(&position, SearchConfig::canary(1), 5, &FailingEvaluator)
            .expect_err("failing evaluator must abort search");
        assert_eq!(error, EvaluationError::new("expected failure"));
    }
}
