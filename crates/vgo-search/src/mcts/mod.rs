//! Progressive-widening MCTS: configuration and results in `types`, the tree in
//! `node`, final move ordering in `ordering`, and the search driver here.

mod node;
mod ordering;
mod types;

use vgo_core::{Phase, Position};

use crate::{EvaluationError, Evaluator, NaiveEvaluator};

pub(crate) use node::Node;
use node::select_playable_child;
use ordering::{preferred_child_indices, sampled_child_indices};
pub use types::{ChildSummary, SearchConfig, SearchResult, SearchStats};

/// Run the simulation budget in rounds of up to `config.leaf_batch` descents,
/// evaluating each round's unique leaves as one batch.
///
/// MCTS is sequential within a game -- a simulation cannot start until the
/// previous one backs up -- so a single game keeps exactly one evaluation in
/// flight and the inference broker sees batches of one. Collecting several
/// leaves before evaluating raises that to `leaf_batch`, which is what lets one
/// game fill a batch. Virtual loss keeps the concurrent descents from all
/// choosing the same branch.
///
/// `Evaluator::evaluate_batch` is the concurrency boundary. Model-backed
/// evaluators can submit the round directly to a vectorized backend; simple
/// evaluators inherit the sequential default. Duplicate descents to the same
/// pending tree edge share one evaluation, then back up in original descent
/// order so the tree remains deterministic.
fn run_batched_simulations(
    root: &mut Node,
    config: SearchConfig,
    match_seed: u64,
    evaluator: &dyn Evaluator,
    stats: &mut SearchStats,
) -> Result<(), EvaluationError> {
    let mut remaining = config.simulations;
    while remaining > 0 {
        let round = config.leaf_batch.min(remaining as usize).max(1);
        let mut descents = Vec::with_capacity(round);
        for _ in 0..round {
            descents.push(descend(root, config, stats));
        }

        // Build one ordered batch and coalesce repeated descents to the same
        // unexpanded edge. Path identity is deliberately stricter than position
        // identity: transpositions need distinct nodes, while a repeated path can
        // safely attach one node and back up the remaining visits through it.
        let mut pending_paths = Vec::<Vec<usize>>::new();
        let mut pending_positions = Vec::<Position>::new();
        let mut pending_index = Vec::with_capacity(descents.len());
        for descent in &descents {
            let Descent::Pending { path, position, .. } = descent else {
                pending_index.push(None);
                continue;
            };
            let index = pending_paths
                .iter()
                .position(|existing| existing == path)
                .unwrap_or_else(|| {
                    let index = pending_paths.len();
                    pending_paths.push(path.clone());
                    pending_positions.push(position.clone());
                    index
                });
            pending_index.push(Some(index));
        }

        let evaluations = if pending_positions.is_empty() {
            Vec::new()
        } else {
            stats.evaluations += pending_positions.len() as u64;
            let evaluations = evaluator.evaluate_batch(&pending_positions)?;
            if evaluations.len() != pending_positions.len() {
                return Err(EvaluationError::new(format!(
                    "evaluator returned {} results for {} positions",
                    evaluations.len(),
                    pending_positions.len()
                )));
            }
            evaluations
        };
        let mut evaluated = pending_positions
            .into_iter()
            .zip(evaluations)
            .map(|(position, evaluation)| {
                let node = Node::from_evaluation(position, evaluation, match_seed);
                let value = node.black_evaluation();
                (value, Some(Box::new(node)))
            })
            .collect::<Vec<_>>();

        for (descent, pending_index) in descents.into_iter().zip(pending_index) {
            match descent {
                Descent::Resolved {
                    path,
                    value,
                    expansion,
                } => {
                    back_up(root, &path, value, expansion, stats);
                }
                Descent::Pending { path, depth, .. } => {
                    let index = pending_index.expect("pending descent has an evaluation index");
                    let (value, node) = &mut evaluated[index];
                    stats.maximum_depth = stats.maximum_depth.max(depth);
                    back_up(root, &path, *value, node.take(), stats);
                }
            }
            stats.simulations += 1;
        }
        remaining -= round as u32;
    }
    Ok(())
}

/// The outcome of descending to a leaf without evaluating it.
pub(crate) enum Descent {
    /// A position needing evaluation, reached by the recorded child path. The
    /// caller evaluates it with the rest of the leaf batch and returns the result
    /// through `back_up`.
    Pending {
        path: Vec<usize>,
        position: Position,
        depth: u32,
    },
    /// The descent ended at a terminal or depth-limited node, whose value is
    /// already known. A newly discovered terminal child carries its node so the
    /// tree records the expansion without ever sending it to inference.
    Resolved {
        path: Vec<usize>,
        value: f64,
        expansion: Option<Box<Node>>,
    },
}

/// Descend from the root to a leaf, applying virtual loss along the way.
///
/// Unlike `simulate` this performs no evaluation, so several descents can be run
/// back to back and their leaves evaluated as one batch. Each visited edge takes
/// a virtual visit, which `back_up` releases. Every `descend` must be paired with
/// exactly one `back_up` or the tree keeps a phantom in-flight visit forever.
pub(crate) fn descend(node: &mut Node, config: SearchConfig, stats: &mut SearchStats) -> Descent {
    let mut path = Vec::new();
    let mut current = &mut *node;
    let mut depth = 0_u32;

    loop {
        stats.maximum_depth = stats.maximum_depth.max(depth);
        if let Some(value) = current.terminal_black_value {
            stats.terminal_leaves += 1;
            current.virtual_visits += 1;
            return Descent::Resolved {
                path,
                value,
                expansion: None,
            };
        }
        if depth >= config.maximum_depth {
            stats.depth_limited_leaves += 1;
            let value = current.black_evaluation();
            current.virtual_visits += 1;
            return Descent::Resolved {
                path,
                value,
                expansion: None,
            };
        }

        current.widen(config, stats);
        let Some((index, transition)) = select_playable_child(current, config) else {
            // Every placement here self-captures and the rules refuse them all,
            // so only the pass remains -- and `select_playable_child` never
            // discards that. Reaching this means the node has no children at
            // all, which `widen` does not produce for a playable position.
            unreachable!("a playable node always keeps its pass candidate");
        };
        current.virtual_visits += 1;
        current.children[index].virtual_visits += 1;
        path.push(index);

        if current.children[index].node.is_none() {
            // Resolve terminal children from the transition analysis. They have
            // exact outcomes and must never consume a model slot.
            let transition = transition.expect("unexpanded selection returns its resolved move");
            if transition.position.phase() == Phase::Finished {
                let value = transition.analysis.outcome.black_utility();
                let expansion = Node::terminal(transition.position, value, current.match_seed);
                stats.maximum_depth = stats.maximum_depth.max(depth + 1);
                stats.terminal_leaves += 1;
                return Descent::Resolved {
                    path,
                    value,
                    expansion: Some(Box::new(expansion)),
                };
            }
            // A playable unexpanded leaf is evaluated by the caller and created
            // during back-up once its evaluation exists.
            return Descent::Pending {
                path,
                position: transition.position,
                depth: depth + 1,
            };
        }
        current = current.children[index]
            .node
            .as_mut()
            .expect("child node checked present");
        depth += 1;
    }
}

/// Release the virtual loss along `path` and record `black_value`.
///
/// `expansion` supplies the evaluated node for a `Descent::Pending` leaf; a
/// `Descent::Resolved` descent passes `None`.
pub(crate) fn back_up(
    node: &mut Node,
    path: &[usize],
    black_value: f64,
    mut expansion: Option<Box<Node>>,
    stats: &mut SearchStats,
) {
    fn recurse(
        node: &mut Node,
        path: &[usize],
        black_value: f64,
        expansion: &mut Option<Box<Node>>,
        stats: &mut SearchStats,
    ) {
        node.virtual_visits = node.virtual_visits.saturating_sub(1);
        node.visits += 1;
        let Some((&index, rest)) = path.split_first() else {
            return;
        };
        let child = &mut node.children[index];
        child.virtual_visits = child.virtual_visits.saturating_sub(1);
        child.visits += 1;
        child.black_value_sum += black_value;
        if let Some(next) = child.node.as_mut() {
            recurse(next, rest, black_value, expansion, stats);
        } else if let Some(created) = expansion.take() {
            debug_assert!(rest.is_empty(), "expansion must terminate the path");
            let mut created = created;
            // The freshly evaluated leaf counts one visit of its own.
            created.visits += 1;
            child.node = Some(created);
            stats.expanded_nodes += 1;
        }
    }
    recurse(node, path, black_value, &mut expansion, stats);
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
    let Some((child_index, transition)) = select_playable_child(node, config) else {
        unreachable!("a playable node always keeps its pass candidate");
    };
    let child = &mut node.children[child_index];
    let black_value = if let Some(child_node) = child.node.as_mut() {
        simulate(child_node, config, match_seed, evaluator, depth + 1, stats)?
    } else {
        let transition = transition.expect("unexpanded selection returns its resolved move");
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
    evaluator.begin_search(position);
    crate::coarse_fine::next_grid_is_search_root();
    let mut root = Node::new(position.clone(), None, match_seed, evaluator, &mut stats)?;
    root.mark_root();
    if config.leaf_batch > 1 {
        run_batched_simulations(&mut root, config, match_seed, evaluator, &mut stats)?;
    } else {
        for _ in 0..config.simulations {
            simulate(&mut root, config, match_seed, evaluator, 0, &mut stats)?;
            stats.simulations += 1;
        }
    }
    Ok(assemble_result(
        &root, position, config, match_seed, ply, stats,
    ))
}

/// Turn a searched tree into a [`SearchResult`].
///
/// Shared by the batched and stepped drivers so selection cannot drift between
/// them; the stepped path exists for browsers, where the caller owns the loop.
pub(crate) fn assemble_result(
    root: &Node,
    position: &Position,
    config: SearchConfig,
    match_seed: u64,
    ply: u32,
    stats: SearchStats,
) -> SearchResult {
    // Refused children never leave the tree, so they must not leave the result
    // either: a policy target listing a move the rules forbid would teach the
    // network to propose it, and a caller reading `children` as the legal move
    // list would offer one.
    let children = root
        .children
        .iter()
        .filter(|child| !child.refused)
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
    SearchResult {
        action: best,
        children,
        stats,
        order,
    }
}

#[cfg(test)]
mod tests;
