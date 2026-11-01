use std::collections::HashSet;

use crate::{Point, Position, numeric};

use super::buckets::{BUCKET_MINIMUM_STONES, StoneBuckets, clear_of_stones_pruned};
use super::in_inset;

#[must_use]
pub fn vertices(position: &Position) -> Vec<Point> {
    let buckets =
        (position.stones().len() >= BUCKET_MINIMUM_STONES).then(|| StoneBuckets::build(position));
    vertices_with(position, buckets.as_ref())
}

/// The enumeration itself, with the pruning structure handed in so a test can
/// run it both ways and compare. `None` is the original quadratic form, and it
/// stays the definition of the answer.
pub(super) fn vertices_with(position: &Position, buckets: Option<&StoneBuckets>) -> Vec<Point> {
    let stones = position.stones();
    let radius = position.radius();
    let diameter = 2.0 * radius;
    let mut result = Vec::new();
    let mut seen = HashSet::new();

    let mut push = |x: f64, y: f64, skip_a: Option<usize>, skip_b: Option<usize>| {
        if !x.is_finite()
            || !y.is_finite()
            || !in_inset(position, x, y)
            || !clear_of_stones_pruned(position, buckets, x, y, skip_a, skip_b)
        {
            return;
        }
        let key = (
            (x / numeric::COORDINATE_EPSILON).round() as i64,
            (y / numeric::COORDINATE_EPSILON).round() as i64,
        );
        if seen.insert(key) {
            result.push(Point::new(x, y));
        }
    };

    for x in [radius, 1.0 - radius] {
        for y in [radius, 1.0 - radius] {
            push(x, y, None, None);
        }
    }
    for (index, stone) in stones.iter().enumerate() {
        for x in [radius, 1.0 - radius] {
            let discriminant = diameter.mul_add(diameter, -(x - stone.x).powi(2));
            if discriminant >= -numeric::COORDINATE_EPSILON {
                let offset = discriminant.max(0.0).sqrt();
                push(x, stone.y + offset, Some(index), None);
                push(x, stone.y - offset, Some(index), None);
            }
        }
        for y in [radius, 1.0 - radius] {
            let discriminant = diameter.mul_add(diameter, -(y - stone.y).powi(2));
            if discriminant >= -numeric::COORDINATE_EPSILON {
                let offset = discriminant.max(0.0).sqrt();
                push(stone.x + offset, y, Some(index), None);
                push(stone.x - offset, y, Some(index), None);
            }
        }
    }
    let mut partners: Vec<usize> = Vec::new();
    for first in 0..stones.len() {
        partners.clear();
        match buckets {
            Some(buckets) => {
                buckets.partners_into(stones[first].x, stones[first].y, first, &mut partners)
            }
            None => partners.extend(first + 1..stones.len()),
        }
        for &second in &partners {
            let dx = stones[second].x - stones[first].x;
            let dy = stones[second].y - stones[first].y;
            let separation = numeric::length(dx, dy);
            if separation < numeric::EDGE_EPSILON
                || separation > 2.0 * diameter + numeric::COORDINATE_EPSILON
            {
                continue;
            }
            let along = separation / 2.0;
            let height_squared = diameter.mul_add(diameter, -along * along);
            if height_squared < -numeric::COORDINATE_EPSILON {
                continue;
            }
            let height = height_squared.max(0.0).sqrt();
            let middle_x = (stones[first].x + stones[second].x) / 2.0;
            let middle_y = (stones[first].y + stones[second].y) / 2.0;
            let ux = dx / separation;
            let uy = dy / separation;
            push(
                middle_x - uy * height,
                middle_y + ux * height,
                Some(first),
                Some(second),
            );
            push(
                middle_x + uy * height,
                middle_y - ux * height,
                Some(first),
                Some(second),
            );
        }
    }
    result
}

/// The vertices and bucket grid for one position, built once and reused across
/// the many point queries a raster makes of it.
///
/// Every exact test in the settled classification screens its candidates with
/// `contains`, which is O(stones), inside a loop over the stones near the query
/// -- quadratic per test, and there are hundreds of tests per raster. Measured
/// at 240 stones that screen is 30% of an incremental append. The buckets
/// answer the same question against the nine cells that can hold a violator.
pub struct LegalSetIndex {
    pub(super) buckets: Option<StoneBuckets>,
    pub(super) vertices: Vec<Point>,
}

impl LegalSetIndex {
    #[must_use]
    pub fn build(position: &Position) -> Self {
        let buckets = (position.stones().len() >= BUCKET_MINIMUM_STONES)
            .then(|| StoneBuckets::build(position));
        let vertices = vertices_with(position, buckets.as_ref());
        Self { buckets, vertices }
    }

    #[must_use]
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }
}
