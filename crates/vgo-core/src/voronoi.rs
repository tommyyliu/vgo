use std::sync::Arc;

use crate::{Position, numeric};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        numeric::length(self.x - other.x, self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EdgeSource {
    Board(BoardSide),
    Bisector { stone: usize, other: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub start: Point,
    pub end: Point,
    pub source: Option<EdgeSource>,
}

impl Edge {}

#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub polygon: Vec<Point>,
    pub area: f64,
    pub edges: Vec<Edge>,
    /// A stone added strictly beyond this distance leaves this cell's bits
    /// unchanged. See [`compute`].
    reach: f64,
    degenerate_edges: usize,
    unclassified_edges: usize,
}

/// One cell per stone, shared between a position and its children.
pub type SharedCells = Arc<Vec<Arc<Cell>>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeometryDiagnostics {
    pub unclassified_edges: usize,
    pub degenerate_edges: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub cells: SharedCells,
    pub adjacency: Vec<Vec<usize>>,
    pub groups: Vec<usize>,
    pub diagnostics: GeometryDiagnostics,
}

#[derive(Clone, Copy)]
struct Constraint {
    nx: f64,
    ny: f64,
    offset: f64,
    source: EdgeSource,
}

fn distance_squared(a: Point, b: Point) -> f64 {
    (a.x - b.x).mul_add(a.x - b.x, (a.y - b.y) * (a.y - b.y))
}

/// A polygon vertex and the constraint that produced the edge leaving it.
///
/// Clipping a convex polygon against a half-plane already knows which
/// constraint each surviving edge lies on: the edge from vertex `i` to `i + 1`
/// is whatever most recently cut there. Recording it as the clip happens
/// replaces the reverse lookup `edge_source` used to do -- a search over every
/// constraint for the one both endpoints sit on, which cost a `residual` (and
/// its square root) per constraint per edge per cell, and grew as the cube of
/// the stone count.
#[derive(Clone, Copy)]
struct Vertex {
    point: Point,
    /// Source of the edge from this vertex to the next, `None` while a vertex
    /// is only a corner of the starting square that nothing has cut yet.
    outgoing: Option<EdgeSource>,
}

/// Buffers the clip path writes through, so a cell costs no allocation.
///
/// `clip_half_plane` and `normalize_polygon` each produced a fresh `Vec` per
/// call, and both run once per clip -- about eight clips a cell, three
/// allocations each. Counted with a wrapping allocator, `Analysis::new` made
/// 643 allocations at 20 stones and 1,425 at 51, roughly 30 a cell. The
/// polygons are a handful of vertices, so almost all of that is allocator
/// traffic rather than memory the geometry needs.
///
/// One scratch is threaded through a whole `compute`, which reuses whatever
/// capacity the widest polygon so far required.
#[derive(Default)]
struct Scratch {
    output: Vec<Vertex>,
    deduped: Vec<Vertex>,
    clean: Vec<Vertex>,
}

fn normalize_polygon(points: &mut Vec<Vertex>, scratch: &mut Scratch) {
    if points.len() < 3 {
        return;
    }
    // Dropping a vertex merges its outgoing edge into its predecessor's. The
    // survivor keeps the *later* edge's source, because that is the edge the
    // merged span actually lies on: the dropped vertex was coincident with or
    // collinear to its neighbour, so the span continues along where it led.
    let deduped = &mut scratch.deduped;
    deduped.clear();
    for vertex in points.iter().copied() {
        if deduped.last().is_none_or(|last| {
            distance_squared(vertex.point, last.point) > numeric::EDGE_EPSILON.powi(2)
        }) {
            deduped.push(vertex);
        } else if let Some(last) = deduped.last_mut() {
            last.outgoing = vertex.outgoing;
        }
    }
    if deduped.len() > 1
        && distance_squared(deduped[0].point, deduped.last().expect("nonempty").point)
            <= numeric::EDGE_EPSILON.powi(2)
    {
        let tail = deduped.pop().expect("nonempty");
        if let Some(last) = deduped.last_mut() {
            last.outgoing = tail.outgoing;
        }
    }
    std::mem::swap(points, deduped);

    let polygon = points;
    loop {
        if polygon.len() < 3 {
            return;
        }
        let mut changed = false;
        let clean = &mut scratch.clean;
        clean.clear();
        let mut pending_wrap: Option<Option<EdgeSource>> = None;
        for index in 0..polygon.len() {
            let a = polygon[(index + polygon.len() - 1) % polygon.len()].point;
            let b = polygon[index];
            let c = polygon[(index + 1) % polygon.len()].point;
            let abx = b.point.x - a.x;
            let aby = b.point.y - a.y;
            let bcx = c.x - b.point.x;
            let bcy = c.y - b.point.y;
            let cross = abx.mul_add(bcy, -aby * bcx);
            let scale = numeric::length(abx, aby) + numeric::length(bcx, bcy);
            if cross.abs() <= numeric::COLLINEAR_EPSILON * scale.max(1.0) {
                changed = true;
                // `b` is collinear between its neighbours, so the span from the
                // previous vertex through `b` to `c` is one edge. It carries
                // b's outgoing source, which is the half of the span that
                // survives as the edge to `c`.
                //
                // Dropping index 0 leaves nothing to fix up yet; the polygon is
                // cyclic, so the predecessor is the last vertex and it is not
                // pushed until the end of this pass. `pending_wrap` carries the
                // source until then.
                match clean.last_mut() {
                    Some(previous) => previous.outgoing = b.outgoing,
                    None => pending_wrap = Some(b.outgoing),
                }
            } else {
                clean.push(b);
            }
        }
        if let (Some(source), Some(last)) = (pending_wrap, clean.last_mut()) {
            last.outgoing = source;
        }
        std::mem::swap(polygon, clean);
        if !changed {
            return;
        }
    }
}

fn clip_half_plane(polygon: &mut Vec<Vertex>, constraint: Constraint, scratch: &mut Scratch) {
    // Taken out of the scratch so `polygon` and the output can be borrowed at
    // once; put back at the end, capacity intact.
    let mut output = std::mem::take(&mut scratch.output);
    output.clear();
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let fa = constraint.nx.mul_add(a.point.x, constraint.ny * a.point.y) - constraint.offset;
        let fb = constraint.nx.mul_add(b.point.x, constraint.ny * b.point.y) - constraint.offset;
        let a_inside = fa <= numeric::COORDINATE_EPSILON;
        let b_inside = fb <= numeric::COORDINATE_EPSILON;
        if a_inside {
            // `a` survives with the edge it already had, unless that edge is
            // about to be cut short -- then the span from the crossing point
            // onward belongs to this constraint, recorded on the new vertex
            // below.
            output.push(a);
        }
        if a_inside != b_inside {
            let denominator = fa - fb;
            if denominator.abs() > f64::EPSILON {
                let t = fa / denominator;
                let crossing = Point::new(
                    a.point.x + t * (b.point.x - a.point.x),
                    a.point.y + t * (b.point.y - a.point.y),
                );
                // Leaving the half-plane: the polygon now runs along this
                // constraint from here to wherever it re-enters, so the new
                // vertex owns a stretch of this constraint. Re-entering: the
                // constraint's stretch ends here and the original edge resumes,
                // which is `a`'s source, already carried by the edge that
                // brought us to this crossing.
                let outgoing = if a_inside {
                    Some(constraint.source)
                } else {
                    // Re-entering along the original a->b edge, so the stretch
                    // from here to `b` is still that edge.
                    a.outgoing
                };
                output.push(Vertex {
                    point: crossing,
                    outgoing,
                });
            }
        }
    }
    // The clipped result becomes the polygon, and the polygon's old buffer
    // goes back to the scratch as next clip's output -- so the two keep
    // trading the same two allocations for the whole of `compute`.
    std::mem::swap(polygon, &mut output);
    scratch.output = output;
    normalize_polygon(polygon, scratch);
}

fn polygon_area(polygon: &[Point]) -> f64 {
    let mut twice_area = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        twice_area += a.x.mul_add(b.y, -b.x * a.y);
    }
    twice_area.abs() / 2.0
}

#[cfg(test)]
fn board_constraint(side: BoardSide) -> Constraint {
    match side {
        BoardSide::Left => Constraint {
            nx: -1.0,
            ny: 0.0,
            offset: 0.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Right => Constraint {
            nx: 1.0,
            ny: 0.0,
            offset: 1.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Top => Constraint {
            nx: 0.0,
            ny: -1.0,
            offset: 0.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Bottom => Constraint {
            nx: 0.0,
            ny: 1.0,
            offset: 1.0,
            source: EdgeSource::Board(side),
        },
    }
}

fn bisector_constraint(position: &Position, stone: usize, other: usize) -> Constraint {
    let a = position.stones()[stone];
    let b = position.stones()[other];
    let nx = b.x - a.x;
    let ny = b.y - a.y;
    Constraint {
        nx,
        ny,
        offset: nx.mul_add((a.x + b.x) / 2.0, ny * (a.y + b.y) / 2.0),
        source: EdgeSource::Bisector { stone, other },
    }
}

#[cfg(test)]
fn residual(constraint: Constraint, point: Point) -> f64 {
    let length = numeric::length(constraint.nx, constraint.ny);
    if length == 0.0 {
        return f64::INFINITY;
    }
    (constraint.nx.mul_add(point.x, constraint.ny * point.y) - constraint.offset).abs() / length
}

/// The reverse lookup `compute` used before edge provenance was recorded at
/// clip time: given an edge, find the constraint both endpoints lie on. Kept
/// as the specification the recorded sources are tested against -- it is the
/// definition of a correct classification, just an expensive way to get it.
#[cfg(test)]
fn edge_source(constraints: &[Constraint], start: Point, end: Point) -> Option<EdgeSource> {
    constraints
        .iter()
        .map(|constraint| {
            (
                residual(*constraint, start).max(residual(*constraint, end)),
                constraint.source,
            )
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .filter(|(error, _)| *error <= numeric::COORDINATE_EPSILON * 4.0)
        .map(|(_, source)| source)
}

/// Circumradius of a polygon about a site: the furthest its boundary reaches.
///
/// The cutoff in `compute` compares candidate distances against twice this, so
/// an empty polygon has to answer zero rather than the identity of a maximum
/// over nothing -- a cell clipped away entirely can admit no further bisector.
fn circumradius(polygon: &[Vertex], site: Point) -> f64 {
    polygon
        .iter()
        .map(|vertex| numeric::length(vertex.point.x - site.x, vertex.point.y - site.y))
        .fold(0.0, f64::max)
}

/// Stones sorted per step of [`compute`]'s visit order; doubles on each step.
const FIRST_SORTED_PREFIX: usize = 16;

/// Sorts the next block of `order[sorted..]` into place, returning the new length
/// of the sorted prefix. Everything after it stays unsorted but no smaller.
fn sort_next_prefix(order: &mut [u128], sorted: usize) -> usize {
    let block = sorted.max(FIRST_SORTED_PREFIX);
    let rest = &mut order[sorted..];
    if block >= rest.len() {
        rest.sort_unstable();
        return order.len();
    }
    let (near, _, _) = rest.select_nth_unstable(block);
    near.sort_unstable();
    sorted + block
}

/// [N3] A distance and stone index as one integer whose order is `f64::total_cmp` on
/// the distance, then the index, so sorting needs no float comparator.
#[inline]
fn order_key(distance: f64, index: usize) -> u128 {
    let bits = distance.to_bits() as i64;
    // `total_cmp`'s own transform: negative floats sort by reversed magnitude.
    let total = bits ^ (((bits >> 63) as u64) >> 1) as i64;
    (u128::from((total as u64) ^ (1 << 63)) << 64) | index as u128
}

#[inline]
fn decode_order_key(key: u128) -> (f64, usize) {
    let total = (((key >> 64) as u64) ^ (1 << 63)) as i64;
    let bits = total ^ (((total >> 63) as u64) >> 1) as i64;
    (f64::from_bits(bits as u64), key as u64 as usize)
}

/// Reusable buffers for [`compute_cell`].
#[derive(Default)]
struct CellBuffers {
    order: Vec<u128>,
    clip: Scratch,
}

/// One stone's cell, clipped from the board square.
fn compute_cell(position: &Position, stone: usize, buffers: &mut CellBuffers) -> Cell {
    let count = position.stones().len();
    let CellBuffers {
        order,
        clip: scratch,
    } = buffers;
    // The starting square's sides are board edges, each vertex owning the
    // side that leaves it: (0,0) to (1,0) is the top, and so on round.
    let mut polygon = vec![
        Vertex {
            point: Point::new(0.0, 0.0),
            outgoing: Some(EdgeSource::Board(BoardSide::Top)),
        },
        Vertex {
            point: Point::new(1.0, 0.0),
            outgoing: Some(EdgeSource::Board(BoardSide::Right)),
        },
        Vertex {
            point: Point::new(1.0, 1.0),
            outgoing: Some(EdgeSource::Board(BoardSide::Bottom)),
        },
        Vertex {
            point: Point::new(0.0, 1.0),
            outgoing: Some(EdgeSource::Board(BoardSide::Left)),
        },
    ];
    let site = {
        let s = position.stones()[stone];
        Point::new(s.x, s.y)
    };
    order.clear();
    for other in 0..count {
        if other == stone {
            continue;
        }
        let o = position.stones()[other];
        order.push(order_key(
            numeric::length(o.x - site.x, o.y - site.y),
            other,
        ));
    }
    // `total_cmp` so a NaN coordinate cannot make the comparator
    // inconsistent; ties break by index, which keeps the clip sequence a
    // function of the position rather than of sort implementation details.
    //
    // Sorted in growing prefixes rather than whole, because the cutoff usually
    // stops after a few neighbours. The comparator is a total order, so this
    // visits exactly the sequence a full sort would.
    let mut sorted = 0;
    let mut position_in_order = 0;
    let mut last_clip = 0.0_f64;
    // The stopping distance for the polygon as it stands, once one was computed.
    let mut stopping_at = None;
    loop {
        if position_in_order == sorted {
            if sorted == order.len() {
                break;
            }
            sorted = sort_next_prefix(order, sorted);
        }
        let (distance, other) = decode_order_key(order[position_in_order]);
        position_in_order += 1;
        if polygon.is_empty() {
            break;
        }
        // [E4] Claim 1: |p - s| <= R for every p in the polygon, so a site
        // further than 2R has |p - o| >= |o - s| - |p - s| > R >= |p - s|
        // and its bisector cannot cut. Claim 2: R is non-increasing and
        // `distance` non-decreasing, so this holds for every later
        // candidate too and the loop stops rather than skipping.
        //
        // The margin is one epsilon of slack against `R` reading an ulp
        // high, which would stop a clip early and leave the cell wrong.
        // Erring the other way costs a no-op clip.
        let stop = 2.0f64.mul_add(circumradius(&polygon, site), numeric::COORDINATE_EPSILON);
        if distance > stop {
            stopping_at = Some(stop);
            break;
        }
        let constraint = bisector_constraint(position, stone, other);
        clip_half_plane(&mut polygon, constraint, scratch);
        last_clip = distance;
    }
    // A later stone beyond both of these sorts after the last clip and trips
    // the cutoff, so the clip sequence, and with it every rounding, is the same.
    let reach = if polygon.is_empty() {
        f64::INFINITY
    } else {
        let stop = stopping_at.unwrap_or_else(|| {
            2.0f64.mul_add(circumradius(&polygon, site), numeric::COORDINATE_EPSILON)
        });
        stop.max(last_clip)
    };
    normalize_polygon(&mut polygon, scratch);
    let mut degenerate_edges = 0;
    let mut unclassified_edges = 0;
    let mut edges = Vec::with_capacity(polygon.len());
    for index in 0..polygon.len() {
        let start = polygon[index];
        let end = polygon[(index + 1) % polygon.len()].point;
        if start.point.distance(end) <= numeric::EDGE_EPSILON {
            degenerate_edges += 1;
            continue;
        }
        let source = start.outgoing;
        if source.is_none() {
            unclassified_edges += 1;
        }
        edges.push(Edge {
            start: start.point,
            end,
            source,
        });
    }
    let points: Vec<Point> = polygon.iter().map(|vertex| vertex.point).collect();
    Cell {
        area: polygon_area(&points),
        polygon: points,
        edges,
        reach,
        degenerate_edges,
        unclassified_edges,
    }
}

/// The diagram of `position`, reusing what it can from its parent's.
///
/// When `position` is its parent plus one appended stone, a parent cell is reused
/// if the new stone lies strictly beyond the cell's `reach`. Not cutting the
/// final polygon would not be enough: a stone can cut an intermediate polygon
/// and be clipped away later, which leaves the same shape with different
/// rounding. Beyond `reach` it never enters the clip sequence at all. Proof: [E5].
#[must_use]
pub fn compute(position: &Position) -> Geometry {
    let count = position.stones().len();
    let parent = position
        .parent_cells()
        .filter(|cells| cells.len() + 1 == count);
    let mut cells = Vec::with_capacity(count);
    let mut adjacency = vec![Vec::new(); count];
    let mut diagnostics = GeometryDiagnostics::default();
    // One set of buffers for the whole call; see `Scratch`.
    let mut buffers = CellBuffers {
        order: Vec::with_capacity(count.saturating_sub(1)),
        clip: Scratch::default(),
    };

    for (stone, neighbors) in adjacency.iter_mut().enumerate() {
        let reused = parent.and_then(|parent| {
            let cell = parent.get(stone)?;
            let site = position.stones()[stone];
            let added = position.stones()[count - 1];
            (numeric::length(added.x - site.x, added.y - site.y) > cell.reach)
                .then(|| Arc::clone(cell))
        });
        let cell = reused.unwrap_or_else(|| Arc::new(compute_cell(position, stone, &mut buffers)));
        diagnostics.degenerate_edges += cell.degenerate_edges;
        diagnostics.unclassified_edges += cell.unclassified_edges;
        for edge in &cell.edges {
            if let Some(EdgeSource::Bisector { other, .. }) = edge.source {
                neighbors.push(other);
            }
        }
        cells.push(cell);
    }

    for stone in 0..count {
        let neighbors = adjacency[stone].clone();
        for other in neighbors {
            if !adjacency[other].contains(&stone) {
                adjacency[other].push(stone);
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut parent: Vec<usize> = (0..count).collect();
    fn find(parent: &mut [usize], mut index: usize) -> usize {
        let mut root = index;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[index] != index {
            let next = parent[index];
            parent[index] = root;
            index = next;
        }
        root
    }
    for (stone, neighbors) in adjacency.iter().enumerate() {
        for &other in neighbors {
            if position.stones()[stone].color == position.stones()[other].color {
                let a = find(&mut parent, stone);
                let b = find(&mut parent, other);
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let groups = (0..count).map(|index| find(&mut parent, index)).collect();
    Geometry {
        cells: Arc::new(cells),
        adjacency,
        groups,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use crate::{Color, Point, Position, Stone};

    use super::{EdgeSource, compute, edge_source};

    fn bits(geometry: &super::Geometry) -> Vec<u64> {
        let mut out = Vec::new();
        for cell in geometry.cells.iter() {
            out.push(cell.area.to_bits());
            out.extend(
                cell.polygon
                    .iter()
                    .flat_map(|p| [p.x.to_bits(), p.y.to_bits()]),
            );
            for edge in &cell.edges {
                out.extend([edge.start.x.to_bits(), edge.start.y.to_bits()]);
                out.extend([edge.end.x.to_bits(), edge.end.y.to_bits()]);
            }
        }
        out
    }

    /// A cell shared with the parent must be the cell a full build would make,
    /// to the last bit, at every stone count from sparse to dense.
    #[test]
    fn a_child_diagram_matches_a_full_build_bit_for_bit() {
        let radius = 1.0 / 38.0;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (mut reused, mut total) = (0, 0);
        for _ in 0..6 {
            let mut stones: Vec<Stone> = Vec::new();
            for _ in 0..6000 {
                let x = radius + next() * (1.0 - 2.0 * radius);
                let y = radius + next() * (1.0 - 2.0 * radius);
                if stones
                    .iter()
                    .any(|s| (s.x - x).hypot(s.y - y) < 2.0 * radius)
                {
                    continue;
                }
                let color = if stones.len().is_multiple_of(2) {
                    Color::Black
                } else {
                    Color::White
                };
                let parent = Position::new(radius, stones.clone(), Color::Black);
                let shared = compute(&parent).cells;
                stones.push(Stone::new(x, y, color));
                let plain = Position::new(radius, stones.clone(), Color::Black);
                let mut child = plain.clone();
                child.link_parent_cells(std::sync::Arc::clone(&shared));

                let fast = compute(&child);
                let full = compute(&plain);
                assert_eq!(bits(&fast), bits(&full), "{} stones", stones.len());
                assert_eq!(fast.adjacency, full.adjacency);
                assert_eq!(fast.groups, full.groups);
                assert_eq!(fast.diagnostics, full.diagnostics);
                total += shared.len();
                reused += shared
                    .iter()
                    .zip(fast.cells.iter())
                    .filter(|(a, b)| std::sync::Arc::ptr_eq(a, b))
                    .count();
            }
        }
        assert!(reused * 2 > total, "reused only {reused} of {total} cells");
    }

    /// Clipping against every stone, the way `compute` did before the cutoff.
    ///
    /// The specification the cutoff is measured against: it must produce the
    /// same diagram, not a close one. See docs/history/VORONOI_CUTOFF.md.
    fn compute_exhaustively(position: &Position) -> Vec<(Vec<Point>, f64)> {
        let count = position.stones().len();
        let mut out = Vec::with_capacity(count);
        let mut scratch = super::Scratch::default();
        for stone in 0..count {
            let mut polygon = vec![
                super::Vertex {
                    point: Point::new(0.0, 0.0),
                    outgoing: None,
                },
                super::Vertex {
                    point: Point::new(1.0, 0.0),
                    outgoing: None,
                },
                super::Vertex {
                    point: Point::new(1.0, 1.0),
                    outgoing: None,
                },
                super::Vertex {
                    point: Point::new(0.0, 1.0),
                    outgoing: None,
                },
            ];
            // Same order the cutoff visits in, so only the stopping differs.
            let site = {
                let s = position.stones()[stone];
                Point::new(s.x, s.y)
            };
            let mut order: Vec<(f64, usize)> = (0..count)
                .filter(|&other| other != stone)
                .map(|other| {
                    let o = position.stones()[other];
                    (super::numeric::length(o.x - site.x, o.y - site.y), other)
                })
                .collect();
            order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for &(_, other) in &order {
                if polygon.is_empty() {
                    break;
                }
                super::clip_half_plane(
                    &mut polygon,
                    super::bisector_constraint(position, stone, other),
                    &mut scratch,
                );
            }
            super::normalize_polygon(&mut polygon, &mut scratch);
            let points: Vec<Point> = polygon.iter().map(|v| v.point).collect();
            let area = super::polygon_area(&points);
            out.push((points, area));
        }
        out
    }

    #[test]
    fn prefix_sorting_visits_the_full_sort_order() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for count in (0..300).chain([1000, 5000]) {
            // Repeated distances, so ties have to break by index too.
            let mut order: Vec<u128> = (0..count)
                .map(|index| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    super::order_key((state % 40) as f64 / 7.0, index)
                })
                .collect();
            let mut expected = order.clone();
            expected.sort_unstable();
            let mut sorted = 0;
            while sorted < order.len() {
                let next = super::sort_next_prefix(&mut order, sorted);
                assert!(next > sorted);
                assert_eq!(order[..next], expected[..next], "count {count}");
                sorted = next;
            }
        }
    }

    #[test]
    fn order_keys_sort_like_total_cmp_and_round_trip() {
        let values = [
            f64::NEG_INFINITY,
            -2.5,
            -1e-300,
            -0.0,
            0.0,
            1e-320,
            1e-300,
            0.5,
            2.5,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
            -f64::NAN,
        ];
        for &a in &values {
            for &b in &values {
                for (i, j) in [(1usize, 2usize), (2, 1), (1, 1)] {
                    let expected = a.total_cmp(&b).then(i.cmp(&j));
                    assert_eq!(
                        super::order_key(a, i).cmp(&super::order_key(b, j)),
                        expected
                    );
                }
            }
            let (decoded, index) = super::decode_order_key(super::order_key(a, 7));
            assert_eq!((decoded.to_bits(), index), (a.to_bits(), 7));
        }
    }

    /// The distance cutoff must not change the diagram, only how long it takes.
    ///
    /// Stopping early is sound only if a stone beyond twice the current
    /// circumradius truly cannot cut the polygon (claim 1) and no later one can
    /// either (claim 2). Both are geometric arguments about exact reals; this
    /// checks they survive contact with f64 over boards including the tangent
    /// packing real games produce, where stones sit at exactly 2r and the
    /// bound is tightest.
    #[test]
    fn the_distance_cutoff_matches_exhaustive_clipping() {
        let radius = 0.055_714_285_714_285_716_f64;
        let mut state = 0x7A3C_1F58_D02E_9B46_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut cells, mut boards) = (0_usize, 0_usize);
        for round in 0..240 {
            // Half the boards pack stones at exactly 2r from an existing one.
            let tangent = round % 2 == 1;
            let mut stones: Vec<Stone> = Vec::new();
            let mut tries = 0;
            while stones.len() < 26 && tries < 40_000 {
                tries += 1;
                let (x, y) = if tangent && !stones.is_empty() && next() < 0.7 {
                    let s = stones[(next() * stones.len() as f64) as usize % stones.len()];
                    let angle = next() * std::f64::consts::TAU;
                    (
                        s.x + 2.0 * radius * angle.cos(),
                        s.y + 2.0 * radius * angle.sin(),
                    )
                } else {
                    (
                        radius + next() * (1.0 - 2.0 * radius),
                        radius + next() * (1.0 - 2.0 * radius),
                    )
                };
                if x < radius || x > 1.0 - radius || y < radius || y > 1.0 - radius {
                    continue;
                }
                if stones.iter().all(|s: &Stone| {
                    super::numeric::length(s.x - x, s.y - y) >= 2.0 * radius - 1.0e-12
                }) {
                    let colour = if stones.len().is_multiple_of(2) {
                        Color::Black
                    } else {
                        Color::White
                    };
                    stones.push(Stone::new(x, y, colour));
                }
            }
            if stones.len() < 20 {
                continue;
            }
            let position = Position::new(radius, stones, Color::Black);
            let fast = compute(&position);
            let slow = compute_exhaustively(&position);
            boards += 1;
            for (index, (points, area)) in slow.iter().enumerate() {
                let cell = &fast.cells[index];
                assert_eq!(
                    cell.polygon.len(),
                    points.len(),
                    "cell {index} has {} vertices, exhaustive has {}",
                    cell.polygon.len(),
                    points.len()
                );
                for (a, b) in cell.polygon.iter().zip(points.iter()) {
                    assert_eq!(
                        (a.x.to_bits(), a.y.to_bits()),
                        (b.x.to_bits(), b.y.to_bits()),
                        "cell {index} vertex differs: {a:?} vs {b:?}"
                    );
                }
                assert_eq!(
                    cell.area.to_bits(),
                    area.to_bits(),
                    "cell {index} area differs: {} vs {area}",
                    cell.area
                );
                cells += 1;
            }
        }
        assert!(boards > 100, "only {boards} boards exercised");
        assert!(cells > 2_000, "only {cells} cells compared");
    }

    /// Provenance recorded at clip time must agree with the reverse lookup.
    ///
    /// `edge_source` classified an edge by searching every constraint for the
    /// one both endpoints sit on, within four coordinate epsilons. Recording
    /// the constraint as the clip happens is the same answer arrived at
    /// directly, and this replays both over random boards to say so -- edge
    /// sources decide adjacency, adjacency decides groups, and groups decide
    /// which stones a move captures, so a disagreement here is a different
    /// game rather than a slower one.
    ///
    /// Kept after `edge_source` stopped being reachable from `compute`: it is
    /// the specification this optimization is measured against.
    #[test]
    fn recorded_edge_sources_match_the_reverse_lookup() {
        let radius = 0.055_714_285_714_285_716_f64;
        let mut state = 0x51ED_2C4A_9B3F_1E7D_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut edges, mut agreed) = (0_usize, 0_usize);
        for _ in 0..200 {
            let mut stones: Vec<Stone> = Vec::new();
            let mut tries = 0;
            while stones.len() < 24 && tries < 20_000 {
                tries += 1;
                let x = radius + next() * (1.0 - 2.0 * radius);
                let y = radius + next() * (1.0 - 2.0 * radius);
                if stones
                    .iter()
                    .all(|s: &Stone| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt() >= 2.0 * radius)
                {
                    let colour = if stones.len().is_multiple_of(2) {
                        Color::Black
                    } else {
                        Color::White
                    };
                    stones.push(Stone::new(x, y, colour));
                }
            }
            if stones.len() < 24 {
                continue;
            }
            let position = Position::new(radius, stones, Color::Black);
            let geometry = compute(&position);
            for (stone, cell) in geometry.cells.iter().enumerate() {
                // The constraint list the reverse lookup used to search.
                let mut constraints = vec![
                    super::board_constraint(super::BoardSide::Left),
                    super::board_constraint(super::BoardSide::Right),
                    super::board_constraint(super::BoardSide::Top),
                    super::board_constraint(super::BoardSide::Bottom),
                ];
                for other in 0..position.stones().len() {
                    if other != stone {
                        constraints.push(super::bisector_constraint(&position, stone, other));
                    }
                }
                for edge in &cell.edges {
                    let looked_up = edge_source(&constraints, edge.start, edge.end);
                    edges += 1;
                    if edge.source == looked_up {
                        agreed += 1;
                    } else {
                        panic!(
                            "edge {:?}->{:?} of cell {stone}: recorded {:?}, lookup {:?}",
                            edge.start, edge.end, edge.source, looked_up
                        );
                    }
                }
            }
        }
        assert!(edges > 5_000, "only {edges} edges compared");
        assert_eq!(edges, agreed);
    }

    #[test]
    fn two_sites_split_the_board_and_retain_provenance() {
        let position = Position::new(
            0.05,
            vec![
                Stone::new(0.25, 0.5, Color::Black),
                Stone::new(0.75, 0.5, Color::White),
            ],
            Color::Black,
        );
        let geometry = compute(&position);
        assert!((geometry.cells[0].area - 0.5).abs() < 1.0e-12);
        assert!((geometry.cells[1].area - 0.5).abs() < 1.0e-12);
        assert_eq!(geometry.adjacency, vec![vec![1], vec![0]]);
        assert!(
            geometry.cells[0]
                .edges
                .iter()
                .any(|edge| { edge.source == Some(EdgeSource::Bisector { stone: 0, other: 1 }) })
        );
        assert_eq!(geometry.diagnostics.unclassified_edges, 0);
    }

    #[test]
    fn point_contacts_do_not_connect_diagonal_groups() {
        let position = Position::new(
            0.05,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.75, 0.75, Color::Black),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let geometry = compute(&position);
        assert!(!geometry.adjacency[0].contains(&2));
        assert_ne!(geometry.groups[0], geometry.groups[2]);
    }
}
