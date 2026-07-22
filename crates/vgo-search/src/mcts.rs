use vgo_core::{Analysis, Color, Phase, Position};

use crate::{
    Action, Candidate, CandidateSequence, CandidateSource, Evaluation, EvaluationError, Evaluator,
    NaiveEvaluator,
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
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub action: Action,
    pub children: Vec<ChildSummary>,
    pub stats: SearchStats,
}

impl SearchResult {
    #[must_use]
    pub fn actions_by_preference(&self, to_move: Color) -> Vec<Action> {
        preferred_child_indices(&self.children, to_move)
            .into_iter()
            .map(|index| self.children[index].action)
            .collect()
    }
}

struct Child {
    candidate: Candidate,
    policy_logit: f64,
    prior: f64,
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
        Ok(Self {
            position,
            visits: 0,
            children: Vec::new(),
            candidates,
            candidates_exhausted: false,
            terminal_black_value,
            evaluation,
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
        })
        .collect::<Vec<_>>();
    let best = preferred_child_indices(&children, position.to_move())
        .first()
        .map(|index| children[*index].action)
        .expect("search produces at least the pass child");
    Ok(SearchResult {
        action: best,
        children,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position};

    use crate::{Action, Evaluation, EvaluationError, Evaluator, Policy};

    use super::{SearchConfig, search, search_with_evaluator};

    struct FlatPolicy;

    impl Policy for FlatPolicy {
        fn logit(&self, _action: Action) -> f64 {
            0.0
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
