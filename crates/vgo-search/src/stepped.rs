//! Search with the loop turned inside out, so the caller owns evaluation.
//!
//! [`search_at_ply`](crate::search_at_ply) runs to completion and calls an
//! [`Evaluator`] from inside its own loop. That is the right shape for
//! self-play, where inference is a synchronous call into a local broker.
//!
//! It is the wrong shape in a browser. There, inference is `await
//! session.run(...)`, which cannot return a value synchronously — and the
//! thread that would have to block waiting for the promise is the same thread
//! that must run the event loop to resolve it. Blocking is not slow there, it
//! deadlocks.
//!
//! So this hands the loop to the caller:
//!
//! ```text
//! let mut search = SteppedSearch::new(position, config, seed, ply);
//! while !search.finished() {
//!     let batch = search.next_batch();       // positions needing evaluation
//!     let results = evaluate_somehow(batch); // caller may await here
//!     search.submit(results)?;
//! }
//! let result = search.finish();
//! ```
//!
//! Two properties this buys beyond avoiding the deadlock, both of which matter
//! more than they sound:
//!
//!   * **A time budget instead of a simulation count.** The caller can stop at
//!     a deadline rather than a fixed number of simulations, which is what a
//!     browser needs when the same code runs on a desktop and a four-year-old
//!     phone.
//!   * **A swappable backend.** WebGPU, a WASM CPU fallback, or a remote
//!     server are all just different implementations of "evaluate this batch",
//!     with no change here.
//!
//! The search itself is unchanged: this reuses `descend`, `back_up` and
//! `Node` directly, and produces bit-identical results to the batched path for
//! the same seed. `stepped_search_matches_the_batched_search` in the tests
//! below is what holds that true.

use vgo_core::{Analysis, Phase, Position};

use crate::evaluator::{Evaluation, EvaluationError};
use crate::mcts::{
    Descent, Node, SearchConfig, SearchResult, SearchStats, assemble_result, back_up, descend,
};

/// A search that pauses whenever it needs the network.
pub struct SteppedSearch {
    position: Position,
    config: SearchConfig,
    match_seed: u64,
    ply: u32,
    stats: SearchStats,
    stage: Stage,
    /// Positions handed to the caller and not yet answered. Kept so `submit`
    /// can check the count and so `next_batch` is idempotent.
    outstanding: Vec<Position>,
}

enum Stage {
    /// The root itself has not been evaluated.
    Root,
    /// A round of descents is in flight; its leaves are `outstanding`.
    Round {
        root: Box<Node>,
        remaining: u32,
        round: usize,
        descents: Vec<Descent>,
        pending_index: Vec<Option<usize>>,
    },
    /// Between rounds, with simulations still to run.
    Ready { root: Box<Node>, remaining: u32 },
    Finished { root: Box<Node> },
}

impl SteppedSearch {
    /// Begin a search. Nothing is evaluated until [`Self::next_batch`].
    ///
    /// # Panics
    /// If the position is already finished, as the batched search does.
    #[must_use]
    pub fn new(position: Position, config: SearchConfig, match_seed: u64, ply: u32) -> Self {
        assert_eq!(
            position.phase(),
            Phase::Playing,
            "cannot search a finished position"
        );
        Self {
            position,
            config,
            match_seed,
            ply,
            stats: SearchStats::default(),
            stage: Stage::Root,
            outstanding: Vec::new(),
        }
    }

    /// Whether the simulation budget is spent and [`Self::finish`] may be called.
    #[must_use]
    pub fn finished(&self) -> bool {
        matches!(self.stage, Stage::Finished { .. })
    }

    /// The next positions needing evaluation, in order.
    ///
    /// Empty exactly when the search is finished. Calling this repeatedly
    /// without an intervening [`Self::submit`] returns the same batch: rounds
    /// whose descents all resolved without needing the network are applied
    /// internally and the search moves on, so an empty return never means
    /// "nothing to do right now".
    pub fn next_batch(&mut self) -> &[Position] {
        if !self.outstanding.is_empty() {
            return &self.outstanding;
        }
        loop {
            match std::mem::replace(&mut self.stage, Stage::Root) {
                Stage::Root => {
                    // The root is evaluated like any other leaf, except that a
                    // finished root is rejected in `new`, so it always needs
                    // the network.
                    self.stats.evaluations += 1;
                    self.outstanding.push(self.position.clone());
                    self.stage = Stage::Root;
                    return &self.outstanding;
                }
                Stage::Ready { root, remaining } => {
                    if remaining == 0 {
                        self.stage = Stage::Finished { root };
                        return &self.outstanding;
                    }
                    let mut root = root;
                    let round = self.config.leaf_batch.min(remaining as usize).max(1);
                    let mut descents = Vec::with_capacity(round);
                    for _ in 0..round {
                        descents.push(descend(&mut root, self.config, &mut self.stats));
                    }
                    // Coalesce repeated descents to the same unexpanded edge,
                    // exactly as the batched path does. Path identity is
                    // stricter than position identity on purpose: transpositions
                    // need distinct nodes, while a repeated path can attach one
                    // node and back the remaining visits up through it.
                    let mut pending_paths = Vec::<Vec<usize>>::new();
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
                                self.outstanding.push(position.clone());
                                index
                            });
                        pending_index.push(Some(index));
                    }
                    self.stats.evaluations += self.outstanding.len() as u64;
                    self.stage = Stage::Round {
                        root,
                        remaining,
                        round,
                        descents,
                        pending_index,
                    };
                    if self.outstanding.is_empty() {
                        // Every descent resolved without the network. Apply the
                        // round here rather than handing back an empty batch,
                        // which the caller would read as "finished".
                        self.apply(Vec::new())
                            .expect("a round with no pending leaves cannot mismatch");
                        continue;
                    }
                    return &self.outstanding;
                }
                stage @ (Stage::Round { .. } | Stage::Finished { .. }) => {
                    self.stage = stage;
                    return &self.outstanding;
                }
            }
        }
    }

    /// Supply evaluations for the batch returned by [`Self::next_batch`].
    ///
    /// # Errors
    /// If the count does not match the outstanding batch.
    pub fn submit(&mut self, evaluations: Vec<Evaluation>) -> Result<(), EvaluationError> {
        if evaluations.len() != self.outstanding.len() {
            return Err(EvaluationError::new(format!(
                "evaluator returned {} results for {} positions",
                evaluations.len(),
                self.outstanding.len()
            )));
        }
        self.apply(evaluations)
    }

    fn apply(&mut self, evaluations: Vec<Evaluation>) -> Result<(), EvaluationError> {
        match std::mem::replace(&mut self.stage, Stage::Root) {
            Stage::Root => {
                let mut evaluations = evaluations;
                let evaluation = evaluations.pop().ok_or_else(|| {
                    EvaluationError::new("the root needs exactly one evaluation")
                })?;
                let root = Box::new(Node::from_evaluation(
                    self.position.clone(),
                    evaluation,
                    self.match_seed,
                ));
                self.outstanding.clear();
                self.stage = Stage::Ready {
                    root,
                    remaining: self.config.simulations,
                };
                Ok(())
            }
            Stage::Round {
                mut root,
                remaining,
                round,
                descents,
                pending_index,
            } => {
                let mut evaluated = self
                    .outstanding
                    .drain(..)
                    .zip(evaluations)
                    .map(|(position, evaluation)| {
                        let node = Node::from_evaluation(position, evaluation, self.match_seed);
                        let value = node.black_evaluation();
                        (value, Some(Box::new(node)))
                    })
                    .collect::<Vec<_>>();

                for (descent, index) in descents.into_iter().zip(pending_index) {
                    match descent {
                        Descent::Resolved {
                            path,
                            value,
                            expansion,
                        } => back_up(&mut root, &path, value, expansion, &mut self.stats),
                        Descent::Pending { path, depth, .. } => {
                            let index =
                                index.expect("a pending descent carries an evaluation index");
                            let (value, node) = &mut evaluated[index];
                            self.stats.maximum_depth = self.stats.maximum_depth.max(depth);
                            back_up(&mut root, &path, *value, node.take(), &mut self.stats);
                        }
                    }
                    self.stats.simulations += 1;
                }
                let remaining = remaining - round as u32;
                self.stage = if remaining == 0 {
                    Stage::Finished { root }
                } else {
                    Stage::Ready { root, remaining }
                };
                Ok(())
            }
            stage @ (Stage::Ready { .. } | Stage::Finished { .. }) => {
                self.stage = stage;
                Err(EvaluationError::new(
                    "submit called with no batch outstanding",
                ))
            }
        }
    }

    /// The search result. Call once [`Self::finished`] reports true.
    ///
    /// # Errors
    /// If the budget is not yet spent.
    pub fn finish(self) -> Result<SearchResult, EvaluationError> {
        let Stage::Finished { root } = self.stage else {
            return Err(EvaluationError::new("search is not finished"));
        };
        Ok(assemble_result(
            &root,
            &self.position,
            self.config,
            self.match_seed,
            self.ply,
            self.stats,
        ))
    }

    /// The move the search currently prefers, without consuming it.
    ///
    /// Callable before the budget is spent: stopping on a deadline is the
    /// expected use of this driver, not an error. Needs at least the root to
    /// have been evaluated.
    ///
    /// # Errors
    /// If no simulations have run yet.
    pub fn best_action(&self) -> Result<crate::Action, EvaluationError> {
        let root = match &self.stage {
            Stage::Ready { root, .. } | Stage::Finished { root } => root,
            Stage::Round { root, .. } => root,
            Stage::Root => {
                return Err(EvaluationError::new("the root has not been evaluated yet"));
            }
        };
        Ok(assemble_result(
            root,
            &self.position,
            self.config,
            self.match_seed,
            self.ply,
            self.stats,
        )
        .action)
    }

    /// Simulations completed so far, for a caller pacing against a deadline.
    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.stats.simulations
    }
}

/// Drive a [`SteppedSearch`] to completion with a synchronous evaluator.
///
/// The stepped and batched paths must not drift, and the cheapest way to keep
/// them honest is to run the same tests through both. Also useful on the host
/// where blocking is fine.
///
/// # Errors
/// Whatever the evaluator returns.
pub fn drive(
    search: &mut SteppedSearch,
    evaluator: &dyn crate::Evaluator,
) -> Result<(), EvaluationError> {
    while !search.finished() {
        let batch = search.next_batch().to_vec();
        if batch.is_empty() {
            break;
        }
        let evaluations = evaluator.evaluate_batch(&batch)?;
        search.submit(evaluations)?;
    }
    Ok(())
}

/// Terminal value of a position, for callers that want to avoid a round trip.
#[must_use]
pub fn terminal_black_value(position: &Position) -> Option<f64> {
    (position.phase() == Phase::Finished).then(|| Analysis::new(position).outcome.black_utility())
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Point, Position, Stone};

    use super::*;
    use crate::{Action, Evaluator, NaiveEvaluator, search_at_ply};

    fn fixture(stones: usize, radius: f64) -> Position {
        let spacing = 2.0 * radius * 1.1;
        let per_row = ((0.84_f64 / spacing).floor() as usize).max(1);
        let mut placed = Vec::new();
        for index in 0..stones {
            let (row, column) = (index / per_row, index % per_row);
            let x = 0.08 + (column as f64 + 0.5) * spacing;
            let y = 0.08 + (row as f64 + 0.5) * spacing;
            if x > 0.94 || y > 0.94 {
                break;
            }
            let colour = if index % 2 == 0 { Color::Black } else { Color::White };
            placed.push(Stone::new(x, y, colour));
        }
        Position::new(radius, placed, Color::Black).with_komi(0.104)
    }

    fn same_action(left: Action, right: Action) -> bool {
        match (left, right) {
            (Action::Pass, Action::Pass) => true,
            (Action::Place(a), Action::Place(b)) => a.x == b.x && a.y == b.y,
            _ => false,
        }
    }

    /// The whole reason the stepped driver is safe to adopt.
    ///
    /// Search is deterministic given a seed, so handing the loop to the caller
    /// must not change a single visit. Anything less than bit-identical means
    /// the browser bot and self-play would be playing subtly different games,
    /// and the divergence would be invisible until it mattered.
    #[test]
    fn stepped_search_matches_the_batched_search() {
        let radius = 0.055_714_285_714_285_716;
        for stones in [0usize, 1, 6, 20] {
            for &leaf_batch in &[1usize, 4, 8] {
                for seed in [1u64, 7, 12345] {
                    let position = fixture(stones, radius);
                    let mut config = crate::SearchConfig::canary(48);
                    config.leaf_batch = leaf_batch;
                    config.temperature = 0.0;

                    let batched = search_at_ply(&position, config, seed, &NaiveEvaluator, 3)
                        .expect("batched search");
                    let mut stepped = SteppedSearch::new(position.clone(), config, seed, 3);
                    drive(&mut stepped, &NaiveEvaluator).expect("stepped search");
                    let stepped = stepped.finish().expect("stepped result");

                    let context = format!("{stones} stones, leaf_batch {leaf_batch}, seed {seed}");
                    assert!(
                        same_action(batched.action, stepped.action),
                        "chosen action differs at {context}"
                    );
                    assert_eq!(
                        batched.children.len(),
                        stepped.children.len(),
                        "child count differs at {context}"
                    );
                    for (left, right) in batched.children.iter().zip(&stepped.children) {
                        assert!(
                            same_action(left.action, right.action),
                            "child action differs at {context}"
                        );
                        assert_eq!(left.visits, right.visits, "visits differ at {context}");
                        assert_eq!(
                            left.black_value.to_bits(),
                            right.black_value.to_bits(),
                            "child value differs at {context}"
                        );
                        assert_eq!(left.prior.to_bits(), right.prior.to_bits(),
                            "prior differs at {context}");
                    }
                    assert_eq!(
                        batched.stats.simulations, stepped.stats.simulations,
                        "simulation count differs at {context}"
                    );
                }
            }
        }
    }

    /// A search that never gets its batches answered must not appear finished,
    /// and must keep offering the same work.
    #[test]
    fn an_unanswered_batch_is_offered_again() {
        let position = fixture(4, 0.055_714_285_714_285_716);
        let mut search = SteppedSearch::new(position, crate::SearchConfig::canary(16), 5, 0);
        let first = search.next_batch().to_vec();
        let second = search.next_batch().to_vec();
        assert!(!first.is_empty());
        assert_eq!(first.len(), second.len());
        assert!(!search.finished());
        assert!(search.submit(Vec::new()).is_err(), "count mismatch must be rejected");
    }

    /// The caller may stop early; what it has is still a usable answer.
    #[test]
    fn a_deadline_can_stop_the_search_early() {
        let position = fixture(6, 0.055_714_285_714_285_716);
        let mut config = crate::SearchConfig::canary(256);
        config.temperature = 0.0;
        let mut search = SteppedSearch::new(position, config, 9, 0);
        // Answer only a few rounds, as a caller out of time would.
        for _ in 0..5 {
            let batch = search.next_batch().to_vec();
            if batch.is_empty() {
                break;
            }
            let evaluations = NaiveEvaluator
                .evaluate_batch(&batch)
                .expect("naive evaluation");
            search.submit(evaluations).expect("submit");
        }
        assert!(!search.finished(), "256 simulations cannot finish in five rounds");
        assert!(search.simulations() > 0, "some simulations must have run");
    }
}
