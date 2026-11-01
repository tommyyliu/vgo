//! The set of legal placement centres: membership, its vertices, exact queries
//! against it, and snapping a point onto it. The candidate enumeration rests on A6
//! in `spec/LEMMAS.md`.

mod buckets;
mod nearest;
mod queries;
mod vertices;

use crate::{Position, numeric};
use buckets::{StoneBuckets, clear_of_stones_pruned};

pub use nearest::{Nearest, nearest, nearest_indexed, nearest_indexed_within, nearest_with};
pub(crate) use queries::has_escape;
pub use queries::{
    distance, none_closer_than, none_closer_than_except_vertices, none_closer_than_indexed,
};
pub use vertices::{LegalSetIndex, vertices};

#[must_use]
pub fn in_inset(position: &Position, x: f64, y: f64) -> bool {
    let radius = position.radius();
    x >= radius - numeric::COORDINATE_EPSILON
        && x <= 1.0 - radius + numeric::COORDINATE_EPSILON
        && y >= radius - numeric::COORDINATE_EPSILON
        && y <= 1.0 - radius + numeric::COORDINATE_EPSILON
}

fn clear_of_stones(
    position: &Position,
    x: f64,
    y: f64,
    skip_a: Option<usize>,
    skip_b: Option<usize>,
) -> bool {
    let minimum = 2.0 * position.radius() - numeric::COORDINATE_EPSILON;
    let minimum_squared = minimum * minimum;
    position.stones().iter().enumerate().all(|(index, stone)| {
        if Some(index) == skip_a || Some(index) == skip_b {
            return true;
        }
        let dx = x - stone.x;
        let dy = y - stone.y;
        dx.mul_add(dx, dy * dy) >= minimum_squared
    })
}

#[must_use]
pub fn contains(position: &Position, x: f64, y: f64) -> bool {
    contains_with(position, None, x, y)
}

fn contains_with(position: &Position, buckets: Option<&StoneBuckets>, x: f64, y: f64) -> bool {
    x.is_finite()
        && y.is_finite()
        && in_inset(position, x, y)
        && clear_of_stones_pruned(position, buckets, x, y, None, None)
}

#[cfg(test)]
mod tests;
