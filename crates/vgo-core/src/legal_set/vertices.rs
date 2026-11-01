use std::collections::HashSet;

use crate::{Point, Position, Stone, numeric};

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
    enumerate(position, buckets).points
}

/// A vertex's place in the enumeration order: the corners, then every stone
/// against the walls, then one section per stone for the pairs it comes first in.
const CORNER_SECTION: u32 = 0;
const WALL_SECTION: u32 = 1;
const FIRST_PAIR_SECTION: u32 = 2;

/// The enumeration's result, and what extending it by one stone needs to know.
struct Enumeration {
    points: Vec<Point>,
    sections: Vec<u32>,
    keys: Vec<(i64, i64)>,
    /// A later candidate rounded onto an earlier one and was dropped. Whether it
    /// would survive the earlier one's removal cannot be told from `points`.
    dropped_duplicate: bool,
}

fn dedup_key(x: f64, y: f64) -> (i64, i64) {
    (
        (x / numeric::COORDINATE_EPSILON).round() as i64,
        (y / numeric::COORDINATE_EPSILON).round() as i64,
    )
}

/// Whether a candidate belongs to the legal set, the test every one passes.
fn admissible(
    position: &Position,
    buckets: Option<&StoneBuckets>,
    x: f64,
    y: f64,
    skip_a: Option<usize>,
    skip_b: Option<usize>,
) -> bool {
    x.is_finite()
        && y.is_finite()
        && in_inset(position, x, y)
        && clear_of_stones_pruned(position, buckets, x, y, skip_a, skip_b)
}

/// Where `stone`'s exclusion circle meets the walls, in enumeration order.
fn wall_candidates(radius: f64, stone: Stone, emit: &mut impl FnMut(f64, f64)) {
    let diameter = 2.0 * radius;
    for x in [radius, 1.0 - radius] {
        let discriminant = diameter.mul_add(diameter, -(x - stone.x).powi(2));
        if discriminant >= -numeric::COORDINATE_EPSILON {
            let offset = discriminant.max(0.0).sqrt();
            emit(x, stone.y + offset);
            emit(x, stone.y - offset);
        }
    }
    for y in [radius, 1.0 - radius] {
        let discriminant = diameter.mul_add(diameter, -(y - stone.y).powi(2));
        if discriminant >= -numeric::COORDINATE_EPSILON {
            let offset = discriminant.max(0.0).sqrt();
            emit(stone.x + offset, y);
            emit(stone.x - offset, y);
        }
    }
}

/// Where two stones' exclusion circles meet, in enumeration order.
fn pair_candidates(radius: f64, first: Stone, second: Stone, emit: &mut impl FnMut(f64, f64)) {
    let diameter = 2.0 * radius;
    let dx = second.x - first.x;
    let dy = second.y - first.y;
    let separation = numeric::length(dx, dy);
    if separation < numeric::EDGE_EPSILON
        || separation > 2.0 * diameter + numeric::COORDINATE_EPSILON
    {
        return;
    }
    let along = separation / 2.0;
    let height_squared = diameter.mul_add(diameter, -along * along);
    if height_squared < -numeric::COORDINATE_EPSILON {
        return;
    }
    let height = height_squared.max(0.0).sqrt();
    let middle_x = (first.x + second.x) / 2.0;
    let middle_y = (first.y + second.y) / 2.0;
    let ux = dx / separation;
    let uy = dy / separation;
    emit(middle_x - uy * height, middle_y + ux * height);
    emit(middle_x + uy * height, middle_y - ux * height);
}

fn enumerate(position: &Position, buckets: Option<&StoneBuckets>) -> Enumeration {
    let stones = position.stones();
    let radius = position.radius();
    let mut result = Enumeration {
        points: Vec::new(),
        sections: Vec::new(),
        keys: Vec::new(),
        dropped_duplicate: false,
    };
    let mut seen = HashSet::new();
    let mut push = |x: f64, y: f64, section: u32, skip_a: Option<usize>, skip_b: Option<usize>| {
        if !admissible(position, buckets, x, y, skip_a, skip_b) {
            return;
        }
        let key = dedup_key(x, y);
        if seen.insert(key) {
            result.points.push(Point::new(x, y));
            result.sections.push(section);
            result.keys.push(key);
        } else {
            result.dropped_duplicate = true;
        }
    };

    for x in [radius, 1.0 - radius] {
        for y in [radius, 1.0 - radius] {
            push(x, y, CORNER_SECTION, None, None);
        }
    }
    for (index, &stone) in stones.iter().enumerate() {
        wall_candidates(radius, stone, &mut |x, y| {
            push(x, y, WALL_SECTION, Some(index), None);
        });
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
            pair_candidates(radius, stones[first], stones[second], &mut |x, y| {
                push(
                    x,
                    y,
                    FIRST_PAIR_SECTION + first as u32,
                    Some(first),
                    Some(second),
                );
            });
        }
    }
    result
}

/// [`enumerate`] for `position`, whose stones are `parent`'s plus one appended.
///
/// Adding a stone can only rule out candidates and add ones involving it, so the
/// parent's list, filtered and merged with the new ones in order, is the answer.
/// `None` when a rounding collision means that argument does not hold, which the
/// full enumeration then settles. Proof: [E13].
fn extend(
    parent: &LegalSetIndex,
    position: &Position,
    buckets: Option<&StoneBuckets>,
) -> Option<Enumeration> {
    if parent.dropped_duplicate {
        return None;
    }
    let stones = position.stones();
    let radius = position.radius();
    let placed = stones.len() - 1;
    let stone = stones[placed];
    let minimum = 2.0 * radius - numeric::COORDINATE_EPSILON;
    let minimum_squared = minimum * minimum;

    // New candidates, in the order the merge below wants them.
    let mut fresh: Vec<(u32, Point)> = Vec::with_capacity(8);
    let mut consider = |x: f64, y: f64, section: u32, skip_b: Option<usize>| {
        if admissible(position, buckets, x, y, Some(placed), skip_b) {
            fresh.push((section, Point::new(x, y)));
        }
    };
    wall_candidates(radius, stone, &mut |x, y| {
        consider(x, y, WALL_SECTION, None);
    });
    let mut partners: Vec<usize> = Vec::with_capacity(32);
    match buckets {
        Some(buckets) => buckets.lower_partners_into(stone.x, stone.y, placed, &mut partners),
        None => partners.extend(0..placed),
    }
    for &first in &partners {
        // The placed stone is `second`, as the full enumeration has it. The pair
        // test applies its own cut, so an extra partner here costs a length only.
        pair_candidates(radius, stones[first], stone, &mut |x, y| {
            consider(x, y, FIRST_PAIR_SECTION + first as u32, Some(placed));
        });
    }

    let capacity = parent.vertices.len() + fresh.len();
    let mut result = Enumeration {
        points: Vec::with_capacity(capacity),
        sections: Vec::with_capacity(capacity),
        keys: Vec::with_capacity(capacity),
        dropped_duplicate: false,
    };
    let mut pending = fresh.iter().peekable();
    let insert = |result: &mut Enumeration, section: u32, point: Point, key: (i64, i64)| {
        result.points.push(point);
        result.sections.push(section);
        result.keys.push(key);
    };
    for (index, (&point, &section)) in parent.vertices.iter().zip(&parent.sections).enumerate() {
        // Each section's new candidates follow everything the parent had in it,
        // because the placed stone has the highest index.
        while let Some(&&(new_section, new_point)) = pending.peek() {
            if new_section >= section {
                break;
            }
            insert(
                &mut result,
                new_section,
                new_point,
                dedup_key(new_point.x, new_point.y),
            );
            pending.next();
        }
        let (dx, dy) = (point.x - stone.x, point.y - stone.y);
        if dx.mul_add(dx, dy * dy) >= minimum_squared {
            insert(&mut result, section, point, parent.keys[index]);
        }
    }
    for &(new_section, new_point) in pending {
        insert(
            &mut result,
            new_section,
            new_point,
            dedup_key(new_point.x, new_point.y),
        );
    }
    // A new candidate rounding onto anything kept is the collision to refuse.
    for &(_, point) in &fresh {
        let key = dedup_key(point.x, point.y);
        if result.keys.iter().filter(|&&other| other == key).count() > 1 {
            return None;
        }
    }
    Some(result)
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
    sections: Vec<u32>,
    keys: Vec<(i64, i64)>,
    dropped_duplicate: bool,
}

impl LegalSetIndex {
    fn from_enumeration(
        position: &Position,
        buckets: Option<StoneBuckets>,
        enumeration: Enumeration,
    ) -> Self {
        let vertex_buckets = VertexBuckets::build(&enumeration.points, 4.0 * position.radius());
        Self {
            buckets,
            vertices: enumeration.points,
            vertex_buckets,
            sections: enumeration.sections,
            keys: enumeration.keys,
            dropped_duplicate: enumeration.dropped_duplicate,
        }
    }

    #[must_use]
    pub fn build(position: &Position) -> Self {
        let buckets = (position.stones().len() >= BUCKET_MINIMUM_STONES)
            .then(|| StoneBuckets::build(position));
        let enumeration = enumerate(position, buckets.as_ref());
        Self::from_enumeration(position, buckets, enumeration)
    }

    /// The index for `position`, whose stones are this index's plus one appended.
    #[must_use]
    pub(crate) fn extended(&self, position: &Position) -> Self {
        let buckets = (position.stones().len() >= BUCKET_MINIMUM_STONES)
            .then(|| StoneBuckets::build(position));
        let enumeration = extend(self, position, buckets.as_ref())
            .unwrap_or_else(|| enumerate(position, buckets.as_ref()));
        Self::from_enumeration(position, buckets, enumeration)
    }

    #[must_use]
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    /// Visits the index of every stone within `reach` of `point`, plus possibly
    /// others. Returns false without visiting when no stone buckets were built.
    pub(crate) fn for_each_stone_near(
        &self,
        point: Point,
        reach: f64,
        visit: &mut impl FnMut(usize),
    ) -> bool {
        match &self.buckets {
            Some(buckets) if reach.is_finite() => {
                buckets.for_each_near(point, reach, visit);
                true
            }
            _ => false,
        }
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
