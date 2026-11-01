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

/// Legal-set vertices bucketed by location, so a query reads the few near it.
///
/// Liveness tests every cell vertex against the legal set, and a raster tests
/// hundreds of undecided pixels, so a scan of every vertex here is quadratic in
/// the board. Wider than the stone buckets because there are few vertices.
pub(super) struct VertexBuckets {
    cell: f64,
    side: usize,
    starts: Vec<u32>,
    points: Vec<Point>,
}

impl VertexBuckets {
    fn build(vertices: &[Point], cell: f64) -> Self {
        let side = ((1.0 / cell).ceil() as usize).clamp(1, 512);
        let bucket = |point: &Point| {
            let column = ((point.x / cell) as usize).min(side - 1);
            let row = ((point.y / cell) as usize).min(side - 1);
            row * side + column
        };
        let mut starts = vec![0u32; side * side + 1];
        for vertex in vertices {
            starts[bucket(vertex) + 1] += 1;
        }
        for index in 1..starts.len() {
            starts[index] += starts[index - 1];
        }
        let mut cursor = starts.clone();
        let mut points = vec![Point::new(0.0, 0.0); vertices.len()];
        for vertex in vertices {
            let slot = &mut cursor[bucket(vertex)];
            points[*slot as usize] = *vertex;
            *slot += 1;
        }
        Self {
            cell,
            side,
            starts,
            points,
        }
    }

    /// Visit every vertex within `reach` of `point`, plus possibly others, in no
    /// useful order, until `visit` accepts one. `reach` must be finite.
    pub(super) fn any_near(
        &self,
        point: Point,
        reach: f64,
        visit: &mut impl FnMut(Point) -> bool,
    ) -> bool {
        // Longer than asked, so rounding cannot leave a bucket unread. The caller
        // still compares each vertex, so this only limits what is read: [E10].
        let reach = reach * (1.0 + 1e-9) + 1e-12;
        let bucket =
            |value: f64| ((value / self.cell).floor().max(0.0) as usize).min(self.side - 1);
        for row in bucket(point.y - reach)..=bucket(point.y + reach) {
            let base = row * self.side;
            let first = self.starts[base + bucket(point.x - reach)] as usize;
            let last = self.starts[base + bucket(point.x + reach) + 1] as usize;
            if self.points[first..last].iter().any(|&vertex| visit(vertex)) {
                return true;
            }
        }
        false
    }
}

/// The vertices and bucket grids for one position, built once and reused across
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
    pub(super) vertex_buckets: VertexBuckets,
}

impl LegalSetIndex {
    #[must_use]
    pub fn build(position: &Position) -> Self {
        let buckets = (position.stones().len() >= BUCKET_MINIMUM_STONES)
            .then(|| StoneBuckets::build(position));
        let vertices = vertices_with(position, buckets.as_ref());
        let vertex_buckets = VertexBuckets::build(&vertices, 4.0 * position.radius());
        Self {
            buckets,
            vertices,
            vertex_buckets,
        }
    }

    #[must_use]
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    /// Whether a vertex lies strictly closer than `distance` to `point`, by the
    /// comparison the exact test makes. Proof: [E10].
    #[must_use]
    pub fn any_vertex_closer(&self, point: Point, distance: f64) -> bool {
        self.vertex_buckets
            .any_near(point, distance, &mut |vertex| {
                numeric::length_below(vertex.x - point.x, vertex.y - point.y, distance)
            })
    }
}
