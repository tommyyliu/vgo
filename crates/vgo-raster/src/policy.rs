//! A network policy laid out on a raster grid.
//!
//! Lives here rather than beside either consumer because it is defined *on* the
//! grid: turning an action into an index is [`action_pixel`], and duplicating
//! that mapping is how two implementations drift apart. `vgo-inference` and the
//! browser client both use this one.

use vgo_core::Position;
use vgo_search::{Action, FineGrid, Policy};

use crate::{RasterConfig, action_pixel};

/// Placement logits over a grid, with the trailing entry being the pass.
pub struct DensePolicy {
    /// The *policy* grid, which may be coarser than the rendered raster.
    config: RasterConfig,
    logits: Vec<f32>,
}

impl DensePolicy {
    /// # Panics
    /// If `logits` is not `config.pixels() + 1` long: the grid plus the pass.
    #[must_use]
    pub fn new(config: RasterConfig, logits: Vec<f32>) -> Self {
        assert_eq!(
            logits.len(),
            config.pixels() + 1,
            "a dense policy is one logit per cell plus the pass"
        );
        Self { config, logits }
    }
}

impl Policy for DensePolicy {
    fn logit(&self, action: Action) -> f64 {
        let index = match action {
            Action::Pass => self.config.pixels(),
            Action::Place(point) => action_pixel(point.x, point.y, self.config),
        };
        f64::from(self.logits[index])
    }

    fn fine_grid(&self, position: &Position, coarse: usize) -> Option<FineGrid> {
        let width = self.config.width;
        let height = self.config.height;
        Some(FineGrid::build(position, width, height, coarse, |row, col| {
            self.logits[row * width + col]
        }))
    }
}
