use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use vgo_core::{Color, Phase, Position};

use crate::{Action, Evaluation, EvaluationError, Evaluator, NaiveEvaluator, Policy};

use super::{
    ChildSummary, Descent, Node, SearchConfig, SearchResult, SearchStats, descend,
    preferred_child_indices, run_batched_simulations, sampled_child_indices, search, search_at_ply,
    search_with_evaluator, simulate,
};

#[test]
fn reroot_uses_the_committed_capture_position_and_rejects_a_mismatch() {
    use vgo_core::{Point, Stone};
    let before = Position::new(
        0.2,
        vec![
            Stone::new(0.799_999, 0.394_301_338_356_336_74, Color::Black),
            Stone::new(0.450_359_755_708_616_2, 0.200_001, Color::White),
            Stone::new(0.200_001, 0.590_089_667_194_192_1, Color::Black),
        ],
        Color::White,
    )
    .with_komi(0.12);
    let action = Action::Place(Point::new(0.604_780_272_776_142_7, 0.799_999));
    let transition = action.apply(&before);
    assert_eq!(transition.captured, 2);
    let after = transition.position;
    let mut root =
        Node::from_evaluation(before.clone(), NaiveEvaluator.evaluate(&before).unwrap(), 7);
    root.widen(SearchConfig::canary(1), &mut SearchStats::default());
    // Attach the evaluated capturing child, then perform the production
    // extraction. It must match captures, not an appended-stone shortcut.
    root.children[0].candidate.action = action;
    root.children[0].node = Some(Box::new(Node::from_evaluation(
        after.clone(),
        NaiveEvaluator.evaluate(&after).unwrap(),
        7,
    )));
    let mut retained = root.take_child(action, &after).unwrap();
    assert!(retained.matches_position(&after));
    assert!(!retained.matches_position(&after.clone().with_komi(0.13)));
    retained.mark_root();
    assert!(retained.is_root);
    assert!(root.take_child(action, &after).is_none());
}

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

/// Unvisited children must stay behind every visited one so that any
/// fallback through the order degrades to visit order rather than to noise.
#[test]
fn unvisited_children_sort_last() {
    let children = vec![
        child(place(0.1), 0),
        child(place(0.2), 7),
        child(place(0.3), 3),
    ];
    for seed in 0..50 {
        let order = sampled_child_indices(&children, Color::Black, 1.0, seed);
        assert_eq!(order.len(), 3);
        assert_eq!(order[2], 0, "the zero-visit child must be last");
    }
}

/// Same seed, same order: replay and debugging depend on it.
#[test]
fn sampling_is_reproducible_for_a_seed() {
    let children = vec![
        child(place(0.1), 40),
        child(place(0.2), 30),
        child(place(0.3), 20),
    ];
    let first = sampled_child_indices(&children, Color::Black, 1.0, 12_345);
    let again = sampled_child_indices(&children, Color::Black, 1.0, 12_345);
    assert_eq!(first, again);
}

/// `leaf_batch = 1` must reproduce the sequential path exactly. It routes
/// through descend/back_up rather than `simulate`, so this pins that the
/// restructuring did not change the search.
#[test]
fn unit_leaf_batch_matches_the_sequential_path() {
    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let sequential = SearchConfig::canary(64);
    let mut batched = sequential;
    batched.leaf_batch = 1;

    let left = search_with_evaluator(&position, sequential, 11, &NaiveEvaluator)
        .expect("naive search is infallible");
    let right = search_with_evaluator(&position, batched, 11, &NaiveEvaluator)
        .expect("naive search is infallible");
    assert_eq!(left.action, right.action);
    assert_eq!(left.stats.simulations, right.stats.simulations);
    let visits =
        |result: &SearchResult| result.children.iter().map(|c| c.visits).collect::<Vec<_>>();
    assert_eq!(visits(&left), visits(&right));
}

/// Virtual loss must leave no residue: after the search every in-flight
/// counter is back to zero and the visit total equals the simulation budget.
/// A leak here would silently distort PUCT for the rest of the game.
#[test]
fn batched_search_conserves_visits_and_clears_virtual_loss() {
    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    for leaf_batch in [1usize, 2, 4, 8] {
        let mut config = SearchConfig::canary(64);
        config.leaf_batch = leaf_batch;
        let mut stats = SearchStats::default();
        let mut root = Node::new(position.clone(), None, 7, &NaiveEvaluator, &mut stats)
            .expect("root evaluates");
        if leaf_batch > 1 {
            run_batched_simulations(&mut root, config, 7, &NaiveEvaluator, &mut stats)
                .expect("batched search runs");
        } else {
            for _ in 0..config.simulations {
                simulate(&mut root, config, 7, &NaiveEvaluator, 0, &mut stats)
                    .expect("sequential search runs");
                stats.simulations += 1;
            }
        }
        assert_eq!(
            root.virtual_visits, 0,
            "root virtual loss leaked at leaf_batch {leaf_batch}"
        );
        assert!(
            root.children.iter().all(|child| child.virtual_visits == 0),
            "child virtual loss leaked at leaf_batch {leaf_batch}"
        );
        assert_eq!(
            root.visits, config.simulations,
            "root visits must equal the simulation budget at leaf_batch {leaf_batch}"
        );
        let child_visits: u32 = root.children.iter().map(|child| child.visits).sum();
        assert_eq!(
            child_visits, config.simulations,
            "child visits must sum to the budget at leaf_batch {leaf_batch}"
        );
        assert_eq!(stats.simulations, config.simulations);
    }
}

/// Virtual loss exists to spread concurrent descents. Without it a round of
/// descents would all pick the same child, so a batched search would explore
/// far fewer distinct branches than a sequential one.
#[test]
fn virtual_loss_spreads_a_batch_across_children() {
    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let mut config = SearchConfig::canary(16);
    config.leaf_batch = 8;
    let mut stats = SearchStats::default();
    let mut root = Node::new(position.clone(), None, 3, &NaiveEvaluator, &mut stats).expect("root");
    root.widen(config, &mut stats);

    // One round of descents without backing any of them up: every descent
    // sees the previous descents' virtual loss.
    let mut chosen = std::collections::HashSet::new();
    for _ in 0..8 {
        let path = match descend(&mut root, config, &mut stats) {
            Descent::Pending { path, .. } | Descent::Resolved { path, .. } => path,
        };
        if let Some(&first) = path.first() {
            chosen.insert(first);
        }
    }
    assert!(
        chosen.len() > 1,
        "virtual loss should spread 8 descents over more than one child, got {}",
        chosen.len()
    );
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

#[derive(Default)]
struct BatchCountingEvaluator {
    single_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    batch_positions: AtomicUsize,
}

impl Evaluator for BatchCountingEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        assert_eq!(
            position.phase(),
            Phase::Playing,
            "terminal positions must not reach evaluate"
        );
        self.single_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Evaluation::new(0.0, Box::new(FlatPolicy)))
    }

    fn evaluate_batch(&self, positions: &[Position]) -> Result<Vec<Evaluation>, EvaluationError> {
        assert!(
            positions
                .iter()
                .all(|position| position.phase() == Phase::Playing),
            "terminal positions must not reach evaluate_batch"
        );
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.batch_positions
            .fetch_add(positions.len(), Ordering::Relaxed);
        Ok(positions
            .iter()
            .map(|_| Evaluation::new(0.0, Box::new(FlatPolicy)))
            .collect())
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
    let mut node = Node::new(position, None, 11, &evaluator, &mut stats).expect("node evaluates");

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
fn coarse_fine_grid_is_built_once_per_node() {
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
    assert_eq!(fine_grid_calls.load(Ordering::Relaxed), 1);
    assert_eq!(node.proposal_draws, 5);
}

#[test]
fn batched_search_coalesces_pending_edges_and_skips_terminal_children() {
    let evaluator = BatchCountingEvaluator::default();
    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let mut config = SearchConfig::canary(16);
    config.initial_candidates = 1;
    config.maximum_candidates = 1;
    config.leaf_batch = 8;

    let result = search_with_evaluator(&position, config, 19, &evaluator)
        .expect("counting evaluator succeeds");

    // Root evaluation remains the ordinary single-position call. The first
    // round descends eight times through the only pending edge, which must
    // become one model position. The next eight descents end the game with a
    // second pass and are resolved entirely from transition analysis.
    assert_eq!(evaluator.single_calls.load(Ordering::Relaxed), 1);
    assert_eq!(evaluator.batch_calls.load(Ordering::Relaxed), 1);
    assert_eq!(evaluator.batch_positions.load(Ordering::Relaxed), 1);
    assert_eq!(result.stats.evaluations, 2);
    assert_eq!(result.stats.terminal_leaves, 8);
    assert_eq!(result.stats.simulations, 16);
    assert_eq!(result.children[0].visits, 16);
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

/// Scores a stone by where it landed: the left half is good for whoever
/// placed it. That gives search a known right answer for either colour.
struct LeftHalfWins;

impl Evaluator for LeftHalfWins {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        let placed_well = position.stones().last().is_some_and(|stone| stone.x < 0.5);
        // The value is for the side to move, who is the placer's opponent.
        let value = if placed_well { -0.9 } else { 0.9 };
        Ok(Evaluation::new(value, Box::new(FlatPolicy)))
    }
}

#[test]
fn search_plays_the_move_the_value_favours_for_either_colour() {
    use vgo_core::Stone;
    for colour in [Color::Black, Color::White] {
        for seed in [1, 2, 3] {
            let opponent = Stone::new(0.9, 0.9, colour.other());
            let position = Position::new(0.05, vec![opponent], colour);
            let config = SearchConfig {
                widening_coefficient: 4.0,
                ..SearchConfig::canary(128)
            };
            let result = search_with_evaluator(&position, config, seed, &LeftHalfWins)
                .expect("scripted evaluator succeeds");
            let Action::Place(point) = result.action else {
                panic!("{colour:?} passed, seed {seed}");
            };
            assert!(
                point.x < 0.5,
                "{colour:?} chose x = {}, seed {seed}",
                point.x
            );
        }
    }
}

#[test]
fn equal_visits_break_toward_the_movers_value() {
    let summaries = [
        ChildSummary {
            black_value: -0.5,
            ..child(place(0.2), 5)
        },
        ChildSummary {
            black_value: 0.5,
            ..child(place(0.8), 5)
        },
    ];
    assert_eq!(
        preferred_child_indices(&summaries, Color::Black),
        vec![1, 0]
    );
    assert_eq!(
        preferred_child_indices(&summaries, Color::White),
        vec![0, 1]
    );
}

#[test]
fn more_visits_outrank_value() {
    let summaries = [
        ChildSummary {
            black_value: 0.9,
            ..child(place(0.2), 2)
        },
        ChildSummary {
            black_value: -0.9,
            ..child(place(0.8), 9)
        },
    ];
    assert_eq!(
        preferred_child_indices(&summaries, Color::Black),
        vec![1, 0]
    );
}

/// PUCT's exploration bonus scales with the square root of the parent's
/// visits. The numbers are chosen so `sqrt` keeps the proven child while a
/// linear bonus jumps to the unvisited one.
#[test]
fn exploration_grows_with_the_root_of_parent_visits() {
    let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
    let config = SearchConfig {
        exploration: 1.5,
        ..SearchConfig::canary(1)
    };
    let mut root = Node::from_evaluation(
        position.clone(),
        ConstantEvaluator(0.0).evaluate(&position).unwrap(),
        7,
    );
    root.widen(
        SearchConfig {
            widening_coefficient: 4.0,
            ..config
        },
        &mut SearchStats::default(),
    );
    assert!(root.children.len() >= 2);
    root.children
        .iter_mut()
        .skip(2)
        .for_each(|child| child.refused = true);
    root.visits = 100;
    root.children[0].prior = 0.5;
    root.children[0].visits = 100;
    root.children[0].black_value_sum = 50.0;
    root.children[1].prior = 0.02;
    root.children[1].visits = 0;
    root.children[1].black_value_sum = 0.0;
    assert_eq!(root.select_child(config), 0);
}

/// A search under the official rules never returns a move those rules
/// refuse, and still finds moves.
///
/// The check that matters is the second half. Dropping refused candidates
/// could in principle strip a node down to its pass and quietly turn the bot
/// into one that only passes, so this asserts real placements come back.
#[test]
fn an_official_search_never_plays_a_refused_move() {
    use vgo_core::{Ruleset, Stone};
    let radius = 1.0 / 18.0;
    for seed in [1u64, 5, 9] {
        let mut stones = Vec::new();
        let spacing = 2.0 * radius * 1.15;
        for index in 0..24 {
            let (row, column) = (index / 5, index % 5);
            let x = 0.12 + column as f64 * spacing;
            let y = 0.12 + row as f64 * spacing;
            if x > 0.94 || y > 0.94 {
                break;
            }
            let colour = if index % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            stones.push(Stone::new(x, y, colour));
        }
        let position = Position::new(radius, stones, Color::Black)
            .with_komi(0.104)
            .with_ruleset(Ruleset::Official);
        if !position.validate().is_playable() {
            continue;
        }

        // Leaf batches above one are the case that matters. `Descent` records
        // a path as child indices and `back_up` replays it after the batch
        // returns, so several paths are outstanding at once -- which is what
        // made an earlier version of this, that removed refused children
        // instead of marking them, walk off the end of a shortened vector
        // in production while every test here passed at leaf batch 1.
        for leaf_batch in [1usize, 4, 8] {
            let mut config = SearchConfig::canary(64);
            config.temperature = 0.0;
            config.leaf_batch = leaf_batch;
            let result = crate::search_at_ply(&position, config, seed, &crate::NaiveEvaluator, 0)
                .expect("search");

            // Whatever it chose, the rules must accept it.
            if let Action::Place(point) = result.action {
                assert!(
                    vgo_core::place(&position, point.x, point.y).is_ok(),
                    "seed {seed}: search returned a move the official rules refuse"
                );
            }
            // And every surviving child must be a move too.
            for child in &result.children {
                if let Action::Place(point) = child.action {
                    assert!(
                        vgo_core::place(&position, point.x, point.y).is_ok(),
                        "seed {seed}: a refused move survived as a candidate"
                    );
                }
            }
            assert!(
                result
                    .children
                    .iter()
                    .any(|c| matches!(c.action, Action::Place(_))),
                "seed {seed}: the search kept no placements at all"
            );
        }
    }
}
