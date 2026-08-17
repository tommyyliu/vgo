use std::{error::Error, fmt};

use vgo_core::{GameEvent, Position, planar_length};

use crate::{Action, FineGrid};

/// Logit for a move the rules refuse outright, as `Ruleset::Official` refuses a
/// self-capture. Large enough to sit below any legal move, finite so a softmax
/// over an all-refused set still normalises -- which cannot happen anyway, since
/// pass is always a candidate and never self-captures.
const REFUSED_MOVE_LOGIT: f64 = 64.0;

const SELF_CAPTURE_LOGIT_PENALTY: f64 = 24.0;

pub trait Policy: Send + Sync {
    fn logit(&self, action: Action) -> f64;

    /// A dense fine grid of placement logits for coarse->fine candidate sampling,
    /// if this policy is spatial. `coarse` is the pool factor. Policies that are
    /// not backed by a spatial map (e.g. the naive heuristic) return `None`, and
    /// the search falls back to the legacy candidate sequence.
    fn fine_grid(&self, _position: &Position, _coarse: usize) -> Option<FineGrid> {
        None
    }
}

pub struct Evaluation {
    pub current_value: f64,
    policy: Box<dyn Policy>,
}

impl Evaluation {
    #[must_use]
    pub fn new(current_value: f64, policy: Box<dyn Policy>) -> Self {
        assert!(current_value.is_finite() && (-1.0..=1.0).contains(&current_value));
        Self {
            current_value,
            policy,
        }
    }

    #[must_use]
    pub fn policy_logit(&self, action: Action) -> f64 {
        let logit = self.policy.logit(action);
        assert!(logit.is_finite(), "policy logits must be finite");
        logit
    }

    /// The policy's fine grid for coarse->fine candidate sampling, if spatial.
    #[must_use]
    pub fn fine_grid(&self, position: &Position, coarse: usize) -> Option<FineGrid> {
        self.policy.fine_grid(position, coarse)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationError {
    message: String,
}

impl EvaluationError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for EvaluationError {}

pub trait Evaluator: Send + Sync {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError>;

    /// Evaluate several positions, returning results in the same order.
    ///
    /// The default keeps simple in-process evaluators source-compatible and is
    /// intentionally allocation-light apart from the returned vector. Evaluators
    /// backed by a vectorized model should override this method so a search can
    /// submit a whole leaf round without spawning one thread per position.
    fn evaluate_batch(&self, positions: &[Position]) -> Result<Vec<Evaluation>, EvaluationError> {
        positions
            .iter()
            .map(|position| self.evaluate(position))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveEvaluator;

impl Evaluator for NaiveEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        Ok(Evaluation::new(
            0.0,
            Box::new(NaivePolicy {
                position: position.clone(),
            }),
        ))
    }
}

struct NaivePolicy {
    position: Position,
}

impl Policy for NaivePolicy {
    fn logit(&self, action: Action) -> f64 {
        let Action::Place(point) = action else {
            return 0.0;
        };
        let clearance = if self.position.stones().is_empty() {
            point.x.min(1.0 - point.x).min(point.y).min(1.0 - point.y)
        } else {
            self.position
                .stones()
                .iter()
                .map(|stone| planar_length(point.x - stone.x, point.y - stone.y))
                .fold(f64::INFINITY, f64::min)
        };
        // Under the official rules this move may not exist at all. Scoring it
        // far below any legal move is the honest prior, and the search drops it
        // on selection regardless -- but it must not be resolved with `apply`,
        // which panics on a move the rules refuse.
        let Some(transition) = action.try_apply(&self.position) else {
            return -REFUSED_MOVE_LOGIT;
        };
        let self_captures = transition
            .events
            .iter()
            .filter_map(|event| match event {
                GameEvent::SelfCapture { count, .. } => Some(*count),
                _ => None,
            })
            .sum::<usize>();
        8.0 * (clearance - 2.0 * self.position.radius())
            - SELF_CAPTURE_LOGIT_PENALTY * self_captures as f64
    }
}
