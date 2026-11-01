//! Fixture positions and the fixed policy shared by `vgo-bench` and `vgo-verify`.

use std::path::Path;
use std::sync::Arc;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{DensePolicy, RasterConfig};
use vgo_search::{Evaluation, EvaluationError, Evaluator};

/// Policy grid and search settings the loop generates with.
pub const POLICY_RESOLUTION: usize = 128;
pub const COARSE_POOL: usize = 16;
pub const SEARCH_SIMULATIONS: u32 = 200;

#[must_use]
pub fn load(path: &Path) -> Vec<Position> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let number = |index: usize| -> f64 { fields[index].parse().expect("number") };
            let colour = |value: f64| {
                if value == 0.0 {
                    Color::Black
                } else {
                    Color::White
                }
            };
            let count = number(4) as usize;
            let stones = (0..count)
                .map(|stone| {
                    let base = 5 + 3 * stone;
                    Stone::new(number(base), number(base + 1), colour(number(base + 2)))
                })
                .collect();
            Position::new(number(0), stones, colour(number(2)))
                .with_komi(number(1))
                .with_passes(number(3) as u32)
        })
        .collect()
}

/// A smooth, fixed placement field, so search exercises the spatial proposal
/// path exactly as it does with the network without needing one.
pub struct FixedEvaluator {
    pub logits: Arc<Vec<f32>>,
}

impl FixedEvaluator {
    #[must_use]
    pub fn new() -> Self {
        let side = POLICY_RESOLUTION;
        let mut logits = Vec::with_capacity(side * side + 1);
        for row in 0..side {
            for column in 0..side {
                let (x, y) = (column as f32 / side as f32, row as f32 / side as f32);
                logits.push((7.0 * x).sin() * (5.0 * y).cos() + 0.5 * (13.0 * (x + y)).sin());
            }
        }
        logits.push(-2.0);
        Self {
            logits: Arc::new(logits),
        }
    }
}

impl Evaluator for FixedEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        // Varies with the position so backed-up values are not all equal.
        let value = ((position.stones().len() % 7) as f64 - 3.0) / 10.0;
        let policy = DensePolicy::new(
            RasterConfig::square(POLICY_RESOLUTION),
            self.logits.to_vec(),
        );
        Ok(Evaluation::new(value, Box::new(policy)))
    }
}
