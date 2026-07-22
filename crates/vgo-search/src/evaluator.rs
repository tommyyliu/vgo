use std::{error::Error, fmt};

use vgo_core::{GameEvent, Position};

use crate::Action;

const SELF_CAPTURE_LOGIT_PENALTY: f64 = 24.0;

pub trait Policy: Send + Sync {
    fn logit(&self, action: Action) -> f64;
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
                .map(|stone| (point.x - stone.x).hypot(point.y - stone.y))
                .fold(f64::INFINITY, f64::min)
        };
        let transition = action.apply(&self.position);
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
