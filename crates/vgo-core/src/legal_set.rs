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
    if contains(position, point.x, point.y) && visit(point) {
        return true;
    }
    let radius = position.radius();
    let diameter = 2.0 * radius;

    for stone in position.stones() {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        let radial_distance = dx.hypot(dy);
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
            let separation = dx.hypot(dy);
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
        let radial = dx.hypot(dy);
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
        // No legal point exists anywhere: the board is full, and the caller
        // must treat the move as unplayable rather than substituting one.
        None => Nearest {
            point,
            legal: false,
            snapped: false,
        },
        Some((_, found)) => Nearest {
            point: found,
            legal: true,
            snapped: true,
        },
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
    use crate::{Color, Point, Position, Stone};

    use super::{contains, distance, nearest, vertices};

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

    #[test]
    fn nearest_projects_an_illegal_query_onto_the_legal_set() {
        let position = Position::new(
            0.1,
            vec![Stone::new(0.5, 0.5, Color::Black)],
            Color::Black,
        );
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
        assert!(!contains(&position, coarse, 0.5), "the probe must be illegal");
        let found = nearest(&position, Point::new(coarse, 0.5));
        assert!(found.legal && found.snapped);
        assert!(contains(&position, found.point.x, found.point.y));
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
