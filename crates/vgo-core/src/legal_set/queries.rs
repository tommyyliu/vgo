use crate::{Point, Position, numeric};

use super::buckets::StoneBuckets;
use super::{LegalSetIndex, contains_with, vertices};

fn visit_candidates(
    position: &Position,
    point: Point,
    known_vertices: Option<&[Point]>,
    mut visit: impl FnMut(Point) -> bool,
) -> bool {
    visit_candidates_within::<false>(position, point, None, known_vertices, None, &mut visit)
}

/// Whether a candidate is legal and accepted, in the order the caller can afford.
#[inline]
fn admit<const VISIT_FIRST: bool>(
    position: &Position,
    buckets: Option<&StoneBuckets>,
    candidate: Point,
    visit: &mut impl FnMut(Point) -> bool,
) -> bool {
    if VISIT_FIRST {
        visit(candidate) && contains_with(position, buckets, candidate.x, candidate.y)
    } else {
        contains_with(position, buckets, candidate.x, candidate.y) && visit(candidate)
    }
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
/// Proof: [E7].
///
/// Only sound for a caller asking whether *anything* is within `threshold`.
/// [`distance`] passes `None` because it needs the true minimum, which any
/// skipped stone could hold.
///
/// With `VISIT_FIRST`, `visit` runs before the legality test. Only for a `visit`
/// that is a pure predicate: it then sees illegal candidates too, so one that
/// records what it was shown would keep a candidate that goes on to fail.
fn visit_candidates_within<const VISIT_FIRST: bool>(
    position: &Position,
    point: Point,
    threshold: Option<f64>,
    known_vertices: Option<&[Point]>,
    buckets: Option<&StoneBuckets>,
    visit: &mut impl FnMut(Point) -> bool,
) -> bool {
    if admit::<VISIT_FIRST>(position, buckets, point, visit) {
        return true;
    }
    let radius = position.radius();
    let diameter = 2.0 * radius;

    let stone_reach = threshold.map(|t| t + diameter);
    let mut visit_stone = |stone: &crate::Stone| {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        let radial_distance = numeric::length(dx, dy);
        if let Some(reach) = stone_reach
            && radial_distance >= reach
        {
            return false;
        }
        let directions: &[(f64, f64)] = if radial_distance < numeric::EDGE_EPSILON {
            &[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        } else {
            &[(dx / radial_distance, dy / radial_distance)]
        };
        for &(ux, uy) in directions {
            let candidate = Point::new(stone.x + diameter * ux, stone.y + diameter * uy);
            if admit::<VISIT_FIRST>(position, buckets, candidate, &mut *visit) {
                return true;
            }
        }
        false
    };
    // Threshold queries need only stones whose radial projection can reach
    // the query disk. The retained index already has the spatial buckets;
    // use them for this enumeration as well as for candidate clearance.
    // Keep the same distance/projection arithmetic and final reach test.
    let found = match (buckets, stone_reach) {
        (Some(buckets), Some(reach))
            if reach.is_finite() && reach > 0.0 && point.x.is_finite() && point.y.is_finite() =>
        {
            buckets.visit_near(position, point, reach, &mut visit_stone)
        }
        _ => position.stones().iter().any(&mut visit_stone),
    };
    if found {
        return true;
    }

    for candidate in [
        Point::new(radius, point.y),
        Point::new(1.0 - radius, point.y),
        Point::new(point.x, radius),
        Point::new(point.x, 1.0 - radius),
    ] {
        if admit::<VISIT_FIRST>(position, buckets, candidate, visit) {
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

/// [`none_closer_than`], against a prebuilt index.
#[must_use]
pub fn none_closer_than_indexed(
    position: &Position,
    point: Point,
    threshold: f64,
    index: &LegalSetIndex,
) -> bool {
    let mut visit =
        |candidate: Point| numeric::length_below(point.x - candidate.x, point.y - candidate.y, threshold);
    !visit_candidates_within::<true>(
        position,
        point,
        Some(threshold),
        Some(&index.vertices),
        index.buckets.as_ref(),
        &mut visit,
    )
}

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
#[must_use]
pub fn none_closer_than(
    position: &Position,
    point: Point,
    threshold: f64,
    known_vertices: Option<&[Point]>,
) -> bool {
    let mut visit =
        |candidate: Point| numeric::length_below(point.x - candidate.x, point.y - candidate.y, threshold);
    !visit_candidates_within::<true>(
        position,
        point,
        Some(threshold),
        known_vertices,
        None,
        &mut visit,
    )
}

pub fn distance(position: &Position, point: Point, known_vertices: Option<&[Point]>) -> f64 {
    let mut best = f64::INFINITY;
    visit_candidates(position, point, known_vertices, |candidate| {
        best = best.min(point.distance(candidate));
        false
    });
    best
}

#[must_use]
pub(crate) fn escape_witness(
    position: &Position,
    vertex: Point,
    stone: Point,
    known_vertices: Option<&[Point]>,
) -> Option<Point> {
    let mut witness = None;
    // A candidate can only win by sitting strictly nearer `vertex` than `stone`
    // does, so nothing beyond that distance ever qualifies -- and
    // `visit_candidates_within` already knows how to skip the stones that
    // cannot reach inside it. This was the call that never passed the bound, so
    // every cell of every position paid the O(stones) clearance test once per
    // stone on the board, to screen candidates that lost on distance anyway.
    //
    // The bound is the exact predicate, not the locality theorem: a stone
    // contributes the point one diameter along the ray to the query, so a stone
    // farther than `reach + diameter` cannot land within `reach`. `SNAP_MARGIN`
    // keeps the geometric cut strictly looser than the interval-arithmetic
    // comparison below, so the two cannot disagree about a boundary case.
    let reach = numeric::length(vertex.x - stone.x, vertex.y - stone.y) + numeric::SNAP_MARGIN;
    visit_candidates_within::<false>(
        position,
        vertex,
        Some(reach),
        known_vertices,
        None,
        &mut |candidate| {
            if numeric::strictly_closer(vertex, candidate, stone).is_strictly_less {
                witness = Some(candidate);
                true
            } else {
                false
            }
        },
    );
    witness
}
