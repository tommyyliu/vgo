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

    use super::{contains, distance, vertices};

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
}
