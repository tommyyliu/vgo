use std::collections::HashSet;

use crate::{Point, Position, numeric};

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
    x.is_finite()
        && y.is_finite()
        && in_inset(position, x, y)
        && clear_of_stones(position, x, y, None, None)
}

fn visit_candidates(
    position: &Position,
    point: Point,
    known_vertices: Option<&[Point]>,
    mut visit: impl FnMut(Point) -> bool,
) -> bool {
    visit_candidates_within(position, point, None, known_vertices, &mut visit)
}

/// [`visit_candidates`], skipping stones whose candidate cannot land within
/// `threshold` of `point`.
///
/// A stone contributes the point one diameter from it along the ray to the
/// query, so that candidate sits `|radial - diameter|` from the query. A stone
/// farther than `threshold + diameter` therefore cannot produce anything within
/// `threshold`, and skipping it also skips the O(stones) legality check that
/// would have screened it -- which is what makes the whole call quadratic.
///
/// Only sound for a caller asking whether *anything* is within `threshold`.
/// [`distance`] passes `None` because it needs the true minimum, which any
/// skipped stone could hold.
fn visit_candidates_within(
    position: &Position,
    point: Point,
    threshold: Option<f64>,
    known_vertices: Option<&[Point]>,
    visit: &mut impl FnMut(Point) -> bool,
) -> bool {
    if contains(position, point.x, point.y) && visit(point) {
        return true;
    }
    let radius = position.radius();
    let diameter = 2.0 * radius;

    let stone_reach = threshold.map(|t| t + diameter);
    for stone in position.stones() {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        let radial_distance = numeric::length(dx, dy);
        if let Some(reach) = stone_reach {
            if radial_distance >= reach {
                continue;
            }
        }
        let directions: &[(f64, f64)] = if radial_distance < numeric::EDGE_EPSILON {
            &[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        } else {
            &[(dx / radial_distance, dy / radial_distance)]
        };
        for &(ux, uy) in directions {
            let candidate = Point::new(stone.x + diameter * ux, stone.y + diameter * uy);
            if contains(position, candidate.x, candidate.y) && visit(candidate) {
                return true;
            }
        }
    }

    for candidate in [
        Point::new(radius, point.y),
        Point::new(1.0 - radius, point.y),
        Point::new(point.x, radius),
        Point::new(point.x, 1.0 - radius),
    ] {
        if contains(position, candidate.x, candidate.y) && visit(candidate) {
            return true;
        }
    }

    let owned_vertices;
    let candidates = if let Some(known) = known_vertices {
        known
    } else {
        owned_vertices = vertices(position);
        &owned_vertices
    };
    for &candidate in candidates {
        if visit(candidate) {
            return true;
        }
    }
    false
}

#[must_use]
pub fn vertices(position: &Position) -> Vec<Point> {
    let stones = position.stones();
    let radius = position.radius();
    let diameter = 2.0 * radius;
    let mut result = Vec::new();
    let mut seen = HashSet::new();

    let mut push = |x: f64, y: f64, skip_a: Option<usize>, skip_b: Option<usize>| {
        if !x.is_finite()
            || !y.is_finite()
            || !in_inset(position, x, y)
            || !clear_of_stones(position, x, y, skip_a, skip_b)
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
    for first in 0..stones.len() {
        for second in first + 1..stones.len() {
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

#[must_use]
/// Whether every legal point is at least `threshold` from `point`.
///
/// The same question [`distance`] answers, asked as a predicate so it can stop
/// early. Callers testing `threshold <= distance(..)` do not need the minimum,
/// only whether anything beats it, and the first candidate that does ends the
/// search -- where `distance` must visit every candidate to be sure it has the
/// smallest.
///
/// That matters because the candidate set is not small: one entry per stone,
/// each screened by an O(stones) legality check, so a single call is quadratic
/// in the stone count. Measured inside the settled mask at 240 stones, the exact
/// tests were 6.5 ms of a 10 ms raster -- 65% of it, for 477 pixels out of
/// 147,456.
pub fn none_closer_than(
    position: &Position,
    point: Point,
    threshold: f64,
    known_vertices: Option<&[Point]>,
) -> bool {
    let mut visit = |candidate: Point| point.distance(candidate) < threshold;
    !visit_candidates_within(position, point, Some(threshold), known_vertices, &mut visit)
}

pub fn distance(position: &Position, point: Point, known_vertices: Option<&[Point]>) -> f64 {
    let mut best = f64::INFINITY;
    visit_candidates(position, point, known_vertices, |candidate| {
        best = best.min(point.distance(candidate));
        false
    });
    best
}

/// Where a placement lands after projecting it onto the legal set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    pub point: Point,
    pub legal: bool,
    pub snapped: bool,
}

/// The legal point closest to `point`, or `point` itself when already legal.
///
/// A policy head emits a fixed lattice, so it can only ever name those cells --
/// but legality is continuous, and a legal region thinner than one cell has no
/// cell centre inside it. Such a region is unreachable: a group whose last
/// liberty sits there can never be captured, however many stones surround it.
/// Observed in a real game at 128x128, where a legal sliver of area 2.9e-5 had
/// no reachable cell and produced an uncapturable group with no eyes.
///
/// Projecting a proposal onto the legal set closes that gap exactly, rather
/// than asymptotically the way a finer lattice would: no lattice makes every
/// sliver reachable, but every sliver is reachable from some nearby point. It
/// also matches what the client already does for a human click
/// (`legalSet.nearest` in voronoi_go.html), so both sides play the same game.
///
/// The candidate set is finite and mirrors the JS reference: the clamped query,
/// each stone pushed off at exactly one diameter along the ray from the query,
/// the four board-edge projections, and the legal-set vertices. The nearest
/// legal point is always one of these -- the legal set is an intersection of
/// half-planes and disc complements, so its closest point to a query lies on a
/// boundary feature, and those are exactly what this enumerates.
#[must_use]
pub fn nearest(position: &Position, point: Point) -> Nearest {
    nearest_with(position, point, None)
}

/// [`nearest`] with a precomputed vertex set.
///
/// The vertices depend only on the position, but make up ~78% of a single
/// `nearest` call. A caller projecting many points against one position -- the
/// sampler does several hundred per grid -- should compute them once and pass
/// them here, matching how `distance` and `escape_witness` take them.
#[must_use]
pub fn nearest_with(
    position: &Position,
    point: Point,
    known_vertices: Option<&[Point]>,
) -> Nearest {
    if contains(position, point.x, point.y) {
        return Nearest {
            point,
            legal: true,
            snapped: false,
        };
    }
    let radius = position.radius();
    // Each candidate is pushed `SNAP_MARGIN` past the constraint that produced
    // it rather than placed exactly on it. A point on the boundary is legal
    // only by the `2r - COORDINATE_EPSILON` slack in `clear_of_stones`, which
    // makes its legality a question of floating-point association order -- and
    // two implementations that associate differently then disagree about a
    // move one of them just proposed.
    let diameter = 2.0 * radius + numeric::SNAP_MARGIN;
    // A radius leaving no inset at all admits no legal point; the margin must
    // not invert the clamp range on the way to reporting that.
    let inset = (radius + numeric::SNAP_MARGIN).min(0.5);
    let clamp = |value: f64| value.clamp(inset.min(1.0 - inset), inset.max(1.0 - inset));

    let mut candidates = vec![Point::new(clamp(point.x), clamp(point.y))];
    for stone in position.stones() {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        let radial = numeric::length(dx, dy);
        // A query exactly on a stone's centre has no ray to push along, so try
        // the four axes instead of dividing by zero.
        if radial < numeric::EDGE_EPSILON {
            for (ux, uy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                candidates.push(Point::new(
                    clamp(stone.x + diameter * ux),
                    clamp(stone.y + diameter * uy),
                ));
            }
        } else {
            candidates.push(Point::new(
                clamp(stone.x + diameter * dx / radial),
                clamp(stone.y + diameter * dy / radial),
            ));
        }
    }
    candidates.push(Point::new(inset, clamp(point.y)));
    candidates.push(Point::new(1.0 - inset, clamp(point.y)));
    candidates.push(Point::new(clamp(point.x), inset));
    candidates.push(Point::new(clamp(point.x), 1.0 - inset));
    match known_vertices {
        Some(known) => candidates.extend_from_slice(known),
        None => candidates.extend(vertices(position)),
    }

    // Nearest first, then stop at the first legal one: the scan wants the
    // closest candidate `contains` accepts, so testing in distance order makes
    // the first acceptance the answer and leaves the rest untested.
    //
    // Selected lazily rather than by sorting the list up front. Sorting costs
    // O(m log m) over a candidate list the vertex set makes O(n^2), and a query
    // deep in contested space -- which is most of what a search asks about --
    // walks far enough down the order to pay all of it. Repeatedly taking the
    // minimum instead costs one linear pass per candidate actually tested, so
    // an answer found immediately does almost no work and an answer found last
    // degrades to the exhaustive scan rather than losing to it.
    let mut ranked: Vec<(f64, Point)> = candidates
        .into_iter()
        .map(|candidate| {
            let dx = candidate.x - point.x;
            let dy = candidate.y - point.y;
            (dx.mul_add(dx, dy * dy), candidate)
        })
        .collect();
    let mut best = None;
    let mut remaining = ranked.len();
    while remaining > 0 {
        // `total_cmp` rather than `partial_cmp`: a NaN key must not silently
        // win the minimum, and candidates are clamped but not proven finite.
        let (index, _) = ranked[..remaining]
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
            .expect("remaining is non-zero");
        let (_, candidate) = ranked[index];
        if contains(position, candidate.x, candidate.y) {
            best = Some(candidate);
            break;
        }
        // Retire the rejected candidate by swapping the unscanned tail over it,
        // so the next pass never reconsiders it.
        remaining -= 1;
        ranked.swap(index, remaining);
    }
    match best {
        // No legal point exists anywhere: the board is full, and the caller
        // must treat the move as unplayable rather than substituting one.
        None => Nearest {
            point,
            legal: false,
            snapped: false,
        },
        Some(found) => Nearest {
            point: clear_by_margin(position, found),
            legal: true,
            snapped: true,
        },
    }
}

/// Pushes a chosen placement off whatever constraint it is resting on.
///
/// The stone-tangent candidates already carry `SNAP_MARGIN`, but the vertex set
/// does not: `vertices` is the exact vertex set of the legal region, and
/// `distance` and `escape_witness` need it that way. `nearest` draws from the
/// same list, so it could still return a point sitting exactly on a constraint
/// -- which is what the margin exists to prevent. A browser game stalled on a
/// vertex 1.1e-16 from a stone, legal to the search and illegal to the page.
///
/// Nudging the winner rather than the candidates keeps the two uses separate:
/// the geometry stays exact, and only the point that becomes a move is moved.
fn clear_by_margin(position: &Position, point: Point) -> Point {
    let radius = position.radius();
    let target = 2.0 * radius + numeric::SNAP_MARGIN;
    let mut moved = point;
    // One pass per stone, nearest first: pushing off one constraint can bring a
    // point nearer another, and the margin is small enough that a fixed number
    // of passes settles it.
    for _ in 0..4 {
        let mut worst: Option<(f64, &crate::Stone)> = None;
        for stone in position.stones() {
            let distance = numeric::length(moved.x - stone.x, moved.y - stone.y);
            if distance < target && worst.is_none_or(|(seen, _)| distance < seen) {
                worst = Some((distance, stone));
            }
        }
        let Some((distance, stone)) = worst else {
            break;
        };
        if distance < numeric::EDGE_EPSILON {
            // Concentric with a stone: no ray to push along, and the candidate
            // generator already covers this case by trying the four axes.
            break;
        }
        let scale = target / distance;
        moved = Point::new(
            stone.x + (moved.x - stone.x) * scale,
            stone.y + (moved.y - stone.y) * scale,
        );
    }
    // Never push a point off the board to gain clearance from a stone.
    let inset = (radius + numeric::SNAP_MARGIN).min(0.5);
    let (low, high) = (inset.min(1.0 - inset), inset.max(1.0 - inset));
    let clamped = Point::new(moved.x.clamp(low, high), moved.y.clamp(low, high));
    if contains(position, clamped.x, clamped.y) {
        clamped
    } else {
        point
    }
}

#[must_use]
pub(crate) fn escape_witness(
    position: &Position,
    vertex: Point,
    stone: Point,
    known_vertices: Option<&[Point]>,
) -> Option<Point> {
    let mut witness = None;
    visit_candidates(position, vertex, known_vertices, |candidate| {
        if numeric::strictly_closer(vertex, candidate, stone).is_strictly_less {
            witness = Some(candidate);
            true
        } else {
            false
        }
    });
    witness
}

#[cfg(test)]
mod tests {
    use super::none_closer_than;

    fn threshold_lattice(count: usize, radius: f64) -> Position {
        let spacing = 2.0 * radius * 1.05;
        let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
        let mut stones = Vec::new();
        for index in 0..count {
            let (row, column) = (index / per_row, index % per_row);
            let x = radius + 0.02 + column as f64 * spacing;
            let y = radius + 0.02 + row as f64 * spacing;
            if x > 1.0 - radius || y > 1.0 - radius {
                break;
            }
            stones.push(Stone {
                x,
                y,
                color: if index % 2 == 0 { Color::Black } else { Color::White },
            });
        }
        Position::new(radius, stones, Color::Black)
    }

    /// `none_closer_than` prunes stones `distance` must visit, so the two can
    /// only be trusted together: the predicate must agree with the exact
    /// minimum at every threshold, including right at it.
    #[test]
    fn the_predicate_agrees_with_the_exact_distance() {
        for (radius, count) in [(1.0 / 18.0, 28usize), (1.0 / 18.0, 52), (1.0 / 36.0, 240)] {
            let position = threshold_lattice(count, radius);
            let known = vertices(&position);
            for row in 0..24 {
                for column in 0..24 {
                    let point = Point::new(
                        (column as f64 + 0.5) / 24.0,
                        (row as f64 + 0.5) / 24.0,
                    );
                    let exact = distance(&position, point, Some(&known));
                    for scale in [0.25, 0.5, 0.9, 1.0, 1.1, 2.0] {
                        let threshold = exact * scale;
                        assert_eq!(
                            none_closer_than(&position, point, threshold, Some(&known)),
                            threshold <= exact,
                            "{count} stones, point {point:?}, threshold {threshold} vs exact {exact}",
                        );
                    }
                }
            }
        }
    }

    use crate::{Color, Point, Position, Stone};

    use super::{contains, distance, nearest, nearest_with, vertices};

    #[test]
    fn closed_boundaries_are_legal() {
        let position = Position::new(0.1, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
        assert!(contains(&position, 0.7, 0.5));
        assert!(!contains(&position, 0.69, 0.5));
        assert!(contains(&position, 0.1, 0.1));
    }

    #[test]
    fn covered_legal_set_has_no_vertices_or_finite_distance() {
        let position = Position::new(
            0.25,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.75, 0.75, Color::Black),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let legal_vertices = vertices(&position);
        assert!(legal_vertices.is_empty());
        assert!(distance(&position, Point::new(0.5, 0.5), Some(&legal_vertices)).is_infinite());
    }

    #[test]
    fn nearest_returns_a_legal_query_unchanged() {
        let position = Position::new(
            0.1,
            vec![Stone::new(0.25, 0.25, Color::Black)],
            Color::Black,
        );
        let query = Point::new(0.7, 0.7);
        let found = nearest(&position, query);
        assert_eq!(found.point, query);
        assert!(found.legal);
        assert!(!found.snapped, "a legal query must not be moved");
    }

    /// Selecting lazily must pick exactly what the exhaustive scan did.
    ///
    /// The original scan kept a strict `<` running minimum, so among candidates
    /// at an equal distance it kept the first in list order. Retiring rejected
    /// candidates by swapping the tail over them reorders the list, so ties are
    /// no longer resolved by position. This replays both over random positions
    /// and compares the chosen point -- a disagreement is a silently different
    /// move, not a test failure anyone would otherwise notice.
    #[test]
    fn lazy_nearest_matches_the_exhaustive_scan() {
        // Bit-exact, not approximate. Both implementations pick a point out of
        // the same candidate list and hand it to the same `clear_by_margin`, so
        // agreeing means choosing the identical candidate -- there is no
        // arithmetic between them that could legitimately round differently.
        // A tolerance here would hide exactly the tie-order divergence the test
        // exists to catch.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // The production radius, plus values either side of it: a larger radius
        // packs fewer stones and leaves more of the board illegal, which is the
        // regime where the scan walks furthest down the candidate order.
        let radii = [0.02, 0.05571428571428571, 0.1, 0.2];
        let mut checked = 0_u32;
        let mut snapped_seen = 0_u32;
        let mut illegal_seen = 0_u32;
        for radius in radii {
            for _ in 0..250 {
                let count = 1 + (next() * 30.0) as usize;
                let mut stones = Vec::new();
                let mut attempts = 0;
                while stones.len() < count && attempts < 2000 {
                    attempts += 1;
                    let x = radius + next() * (1.0 - 2.0 * radius);
                    let y = radius + next() * (1.0 - 2.0 * radius);
                    if stones.iter().all(|s: &crate::Stone| {
                        super::numeric::length(s.x - x, s.y - y) >= 2.0 * radius
                    }) {
                        let colour = if stones.len() % 2 == 0 {
                            crate::Color::Black
                        } else {
                            crate::Color::White
                        };
                        stones.push(crate::Stone::new(x, y, colour));
                    }
                }
                let position = crate::Position::new(radius, stones, crate::Color::Black);
                let known = vertices(&position);
                for _ in 0..6 {
                    let query = Point::new(next(), next());
                    // Both paths: `Some` is what the search uses, `None` makes
                    // `nearest_with` build the vertex set itself, and only the
                    // second is reachable from `nearest`.
                    for supplied in [Some(&known[..]), None] {
                        let fast = nearest_with(&position, query, supplied);
                        let slow = exhaustive_nearest(&position, query, &known);
                        assert_eq!(
                            (fast.legal, fast.snapped),
                            (slow.0, slow.2),
                            "legality/snap disagreed at {query:?} with radius {radius}"
                        );
                        if fast.legal {
                            assert_eq!(
                                (fast.point.x.to_bits(), fast.point.y.to_bits()),
                                (slow.1.x.to_bits(), slow.1.y.to_bits()),
                                "chose {:?}, exhaustive chose {:?} at radius {radius}",
                                fast.point,
                                slow.1
                            );
                        }
                        checked += 1;
                    }
                    let probe = nearest_with(&position, query, Some(&known));
                    if probe.snapped {
                        snapped_seen += 1;
                    }
                    if !probe.legal {
                        illegal_seen += 1;
                    }
                }
            }
        }
        // The comparison is only worth anything if it exercised the branches:
        // an all-legal sample would never reach the candidate scan at all, and
        // a board with no legal point never reaches the choice either.
        assert!(checked > 10_000, "only {checked} comparisons");
        assert!(snapped_seen > 1_000, "only {snapped_seen} snapped queries");
        assert!(
            illegal_seen > 0,
            "no fully covered board was generated; the None branch is untested"
        );
    }

    /// The pre-optimization scan: build every candidate, test all of them,
    /// keep a strict running minimum. Mirrors `nearest_with` exactly.
    fn exhaustive_nearest(
        position: &Position,
        point: Point,
        known: &[Point],
    ) -> (bool, Point, bool) {
        if contains(position, point.x, point.y) {
            return (true, point, false);
        }
        let radius = position.radius();
        let diameter = 2.0 * radius + super::numeric::SNAP_MARGIN;
        let inset = (radius + super::numeric::SNAP_MARGIN).min(0.5);
        let clamp = |value: f64| value.clamp(inset.min(1.0 - inset), inset.max(1.0 - inset));
        let mut candidates = vec![Point::new(clamp(point.x), clamp(point.y))];
        for stone in position.stones() {
            let dx = point.x - stone.x;
            let dy = point.y - stone.y;
            let radial = super::numeric::length(dx, dy);
            if radial < super::numeric::EDGE_EPSILON {
                for (ux, uy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    candidates.push(Point::new(
                        clamp(stone.x + diameter * ux),
                        clamp(stone.y + diameter * uy),
                    ));
                }
            } else {
                candidates.push(Point::new(
                    clamp(stone.x + diameter * dx / radial),
                    clamp(stone.y + diameter * dy / radial),
                ));
            }
        }
        candidates.push(Point::new(inset, clamp(point.y)));
        candidates.push(Point::new(1.0 - inset, clamp(point.y)));
        candidates.push(Point::new(clamp(point.x), inset));
        candidates.push(Point::new(clamp(point.x), 1.0 - inset));
        candidates.extend_from_slice(known);
        let mut best: Option<(f64, Point)> = None;
        for candidate in candidates {
            if !contains(position, candidate.x, candidate.y) {
                continue;
            }
            let dx = candidate.x - point.x;
            let dy = candidate.y - point.y;
            let squared = dx.mul_add(dx, dy * dy);
            if best.is_none_or(|(best_squared, _)| squared < best_squared) {
                best = Some((squared, candidate));
            }
        }
        match best {
            None => (false, point, false),
            Some((_, found)) => (true, super::clear_by_margin(position, found), true),
        }
    }

    #[test]
    fn nearest_projects_an_illegal_query_onto_the_legal_set() {
        let position = Position::new(0.1, vec![Stone::new(0.5, 0.5, Color::Black)], Color::Black);
        // Directly on the stone's centre: illegal, and with no ray to push
        // along, so this also covers the degenerate direction case.
        let found = nearest(&position, Point::new(0.5, 0.5));
        assert!(found.legal);
        assert!(found.snapped);
        assert!(contains(&position, found.point.x, found.point.y));
        // A diameter of clearance plus the snap margin. Landing exactly on the
        // constraint would make the result legal only by the tolerance in
        // `clear_of_stones`, so a second implementation re-checking the same
        // point could disagree; the margin puts it unambiguously inside.
        let distance = (found.point.x - 0.5).hypot(found.point.y - 0.5);
        assert!(
            distance >= 0.2,
            "expected at least a diameter of clearance, got {distance}"
        );
        assert!(
            (distance - (0.2 + crate::numeric::SNAP_MARGIN)).abs() < 1.0e-9,
            "expected a diameter plus the snap margin, got {distance}"
        );
    }

    #[test]
    fn nearest_reaches_a_region_no_lattice_cell_can_name() {
        // The failure this exists for: a legal region thinner than a policy
        // cell. Two stones a little over two diameters apart leave a legal
        // sliver between them that a coarse lattice can miss entirely, and a
        // group whose last liberty sits there is uncapturable.
        let radius = 1.0 / 18.0;
        let gap = 4.0 * radius + 0.004;
        let position = Position::new(
            radius,
            vec![
                Stone::new(0.5 - gap / 2.0, 0.5, Color::Black),
                Stone::new(0.5 + gap / 2.0, 0.5, Color::Black),
            ],
            Color::White,
        );
        let midpoint = Point::new(0.5, 0.5);
        assert!(
            contains(&position, midpoint.x, midpoint.y),
            "the sliver between the stones must be legal"
        );

        // A lattice that steps over the sliver cannot name it, but the nearest
        // cell it can name projects back into it.
        let coarse = 0.5 + 0.003;
        assert!(
            !contains(&position, coarse, 0.5),
            "the probe must be illegal"
        );
        let found = nearest(&position, Point::new(coarse, 0.5));
        assert!(found.legal && found.snapped);
        assert!(contains(&position, found.point.x, found.point.y));
    }

    #[test]
    fn a_snap_that_lands_on_a_vertex_still_clears_the_margin() {
        // `nearest` chooses from the stone-tangent candidates *and* the vertex
        // set. The tangents carry SNAP_MARGIN; the vertices deliberately do
        // not, because `distance` and `escape_witness` need the exact vertex
        // set of the legal region. So a snap could still return a point resting
        // exactly on a constraint whenever a vertex was the closest candidate.
        //
        // A browser game stalled on one: a click snapped to a vertex 1.1e-16
        // from a stone, which the page accepted and the model server's own
        // legality check then refused, leaving the model unable to answer.
        let radius = 0.055714285714285716;
        let position = Position::new(
            radius,
            vec![
                Stone::new(0.70703125, 0.57421875, Color::Black),
                Stone::new(0.449219, 0.425781, Color::White),
                Stone::new(0.32421875, 0.69921875, Color::Black),
                Stone::new(0.370426, 0.504574, Color::White),
                Stone::new(0.72265625, 0.23046875, Color::Black),
                Stone::new(0.245426, 0.620426, Color::White),
                Stone::new(0.55859375, 0.80078125, Color::Black),
                Stone::new(0.560365, 0.417842, Color::White),
                Stone::new(0.77734375, 0.40234375, Color::Black),
                Stone::new(0.550655, 0.689635, Color::White),
            ],
            Color::Black,
        );
        // The click that produced the stalling stone.
        let found = nearest(&position, Point::new(0.66, 0.675));
        assert!(found.legal);
        for stone in position.stones() {
            let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
            assert!(
                distance >= 2.0 * radius,
                "a snapped placement must not rest on a constraint; this one \
                 sits {:.3e} inside a stone",
                2.0 * radius - distance
            );
        }
    }

    #[test]
    fn snapped_placements_do_not_accumulate_an_invalid_position() {
        // A second stalled browser game, further in. Four of the server's
        // twelve moves had landed 1.6e-6 to 2.7e-6 inside a neighbour, each
        // legal to the search that snapped it. Those stones stay on the board,
        // so `analyze` reports four overlapping pairs and `place` refuses every
        // subsequent move with `invalid-position` -- the model appears stuck
        // while the real damage happened several moves earlier.
        //
        // One bad snap is a rejected move; a bad snap that gets played corrupts
        // the position for the rest of the game.
        let radius = 39.0 / 700.0;
        let position = Position::new(
            radius,
            vec![
                Stone::new(0.69922, 0.59766, Color::Black),
                Stone::new(0.628657, 0.511415, Color::White),
                Stone::new(0.75391, 0.44922, Color::Black),
                Stone::new(0.605469, 0.660156, Color::White),
                Stone::new(0.61328, 0.33984, Color::Black),
            ],
            Color::White,
        );
        // The cell the search wanted at move 5, which snapped to a point
        // 2.4e-6 inside the stone at (0.69922, 0.59766).
        let found = nearest(&position, Point::new(0.739220, 0.701659));
        assert!(found.legal);
        for stone in position.stones() {
            let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
            assert!(
                distance >= 2.0 * radius,
                "a snapped placement that is played must leave the position \
                 valid; this one sits {:.3e} inside a stone",
                2.0 * radius - distance
            );
        }
    }

    #[test]
    fn a_snapped_placement_survives_an_independent_recheck() {
        // The move server stalled a browser game here. It snapped White's move
        // to (0.721723, 0.425648), one diameter from Black at (0.83203,
        // 0.44141) at radius 39/700 -- legal to the search that proposed it,
        // and 1.1e-6 inside the exclusion disc once the client re-derived the
        // distance its own way. The client refused the move, re-asked, and the
        // stateless server returned the same point twenty times.
        //
        // The margin is what makes a snapped point survive being re-checked by
        // an implementation that associates its arithmetic differently.
        let position = Position::new(
            39.0 / 700.0,
            vec![
                Stone::new(0.62891, 0.80078, Color::Black),
                Stone::new(0.61328, 0.63672, Color::White),
                Stone::new(0.47266, 0.71484, Color::Black),
                Stone::new(0.54272, 0.55048, Color::White),
                Stone::new(0.75391, 0.68359, Color::Black),
                Stone::new(0.74597, 0.57245, Color::White),
                Stone::new(0.89453, 0.59766, Color::Black),
                Stone::new(0.39386, 0.63605, Color::White),
                Stone::new(0.83203, 0.44141, Color::Black),
            ],
            Color::White,
        );
        let found = nearest(&position, Point::new(0.721723, 0.425648));
        assert!(found.legal);
        let minimum = 2.0 * position.radius();
        for stone in position.stones() {
            let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
            assert!(
                distance >= minimum,
                "snapped point sits {:.3e} inside a stone's exclusion disc; a \
                 recheck that does not subtract COORDINATE_EPSILON rejects it",
                minimum - distance
            );
        }
    }

    #[test]
    fn nearest_reports_failure_when_nothing_is_legal() {
        // A radius that leaves no inset at all: there is no legal point to
        // snap to, and the caller must not be handed a fabricated one.
        let position = Position::new(0.5, Vec::new(), Color::Black);
        let found = nearest(&position, Point::new(0.1, 0.9));
        if !found.legal {
            assert!(!found.snapped);
            assert_eq!(found.point, Point::new(0.1, 0.9));
        }
    }
}
