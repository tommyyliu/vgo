//! The settled region: board a future placement can no longer reach.
//!
//! A point is settled when it is closer to some stone than to the legal set:
//! `||x - s|| <= dist(x, L)`. Nothing can ever be played nearer to it than that
//! stone already is, so the territory is decided.
//!
//! A port of `reference/src/geometry/settled-contour.js`, which the client draws
//! as its "lock" overlay. Three facts from that derivation carry over and are
//! what make this affordable:
//!
//!   A15  The Voronoi partition drops out. The settled set is the union over
//!        stones of `R_s = { x : ||x-s|| <= dist(x,L) }`, because a stone that
//!        is not nearest only makes the test harder to satisfy.
//!
//!   A16  Each `R_s` is star-shaped about its own stone, so its boundary is a
//!        single-valued radial function `T(u)` and a two-dimensional level set
//!        becomes one scalar equation per direction.
//!
//!   A17  That equation has a closed form: a minimum over the same candidate
//!        families the legal set is built from, with `t_c >= ||c-s||/2` by the
//!        triangle inequality, so candidates sorted by distance admit an exact
//!        early stop.
//!
//! Nothing here samples a field or iterates to a root.
//!
//! Why this exists rather than a per-pixel `distance_to_legal_set`: the direct
//! form measured 42x the cost of the whole rest of the raster, which on a real
//! shard is 186% of the CPU budget across every actor thread. Solving per stone
//! and testing each pixel against its own stone's boundary moves the work off
//! the pixel loop.

use crate::{Point, Position, legal_set, numeric};

/// Per-stone data for evaluating one stone's settled boundary.
///
/// Built once per stone and reused for every direction, which is the whole
/// point: the candidate orderings are most of the cost and they do not depend
/// on the ray.
pub struct SettledRegion<'a> {
    position: &'a Position,
    origin: Point,
    diameter: f64,
    /// Stones by distance from `origin`, with the floor each one implies.
    stone_order: Vec<(usize, f64)>,
    /// Legal-set vertices by distance from `origin`, with their floors.
    vertex_order: Vec<(Point, f64)>,
    /// Set when the legal set is empty: every point is settled.
    unbounded: bool,
}

impl<'a> SettledRegion<'a> {
    /// Prepares the radial solve for one stone.
    #[must_use]
    pub fn new(position: &'a Position, stone: usize, vertices: &[Point]) -> Self {
        let radius = position.radius();
        let origin = {
            let s = position.stones()[stone];
            Point::new(s.x, s.y)
        };
        let diameter = 2.0 * radius;

        // An empty legal set makes `dist(., L)` infinite everywhere, so every
        // point is settled. Deciding it here also keeps the solve away from its
        // one bad regime, where no candidate is ever admissible.
        let unbounded = legal_set::distance(position, origin, Some(vertices)).is_infinite();

        let mut stone_order: Vec<(usize, f64)> = position
            .stones()
            .iter()
            .enumerate()
            .map(|(index, other)| {
                let separation = (other.x - origin.x).hypot(other.y - origin.y);
                (index, separation)
            })
            .collect();
        stone_order.sort_by(|a, b| a.1.total_cmp(&b.1));
        // `t_c >= ||c-s||/2`, so a candidate further than twice the current best
        // cannot improve on it and neither can anything after it.
        for entry in &mut stone_order {
            entry.1 = (entry.1 - diameter).max(0.0) / 2.0;
        }

        let mut vertex_order: Vec<(Point, f64)> = vertices
            .iter()
            .map(|&vertex| (vertex, origin.distance(vertex)))
            .collect();
        vertex_order.sort_by(|a, b| a.1.total_cmp(&b.1));
        for entry in &mut vertex_order {
            entry.1 /= 2.0;
        }


        Self {
            position,
            origin,
            diameter,
            stone_order,
            vertex_order,
            unbounded,
        }
    }

    /// Distance from the stone to its settled boundary along `(ux, uy)`.
    ///
    /// `cap` bounds the search, and callers pass the distance at which the ray
    /// leaves the board: a convex board is left once (A16), so truncating there
    /// loses nothing inside it.
    #[must_use]
    pub fn radius_along(&self, ux: f64, uy: f64, cap: f64) -> f64 {
        if self.unbounded {
            return f64::INFINITY;
        }
        let mut best = cap;
        let stones = self.position.stones();

        // Circular candidates: a placement tangent to another stone.
        for &(index, floor) in &self.stone_order {
            if floor >= best {
                break;
            }
            let other = stones[index];
            let wx = self.origin.x - other.x;
            let wy = self.origin.y - other.y;
            let along = ux * wx + uy * wy;
            let numerator = self.diameter.mul_add(self.diameter, -(wx * wx + wy * wy));
            for branch in 0..2 {
                let denominator = 2.0
                    * if branch == 1 {
                        along - self.diameter
                    } else {
                        along + self.diameter
                    };
                if denominator == 0.0 {
                    continue;
                }
                let t = numerator / denominator;
                if !(t > 0.0) || t >= best {
                    continue;
                }
                let px = self.origin.x + t * ux - other.x;
                let py = self.origin.y + t * uy - other.y;
                let span = px.hypot(py);
                if span < numeric::EDGE_EPSILON {
                    continue;
                }
                // The root must be the branch we solved for, not its twin.
                if ((span - self.diameter).abs() - t).abs() > numeric::EDGE_EPSILON {
                    continue;
                }
                let foot_x = other.x + self.diameter * px / span;
                let foot_y = other.y + self.diameter * py / span;
                if !legal_set::contains(self.position, foot_x, foot_y) {
                    continue;
                }
                best = t;
            }
        }

        // Linear candidates: the four board edges, inset by the radius.
        let radius = self.position.radius();
        for side in 0..4 {
            let vertical = side < 2;
            let line = if side % 2 == 1 { 1.0 - radius } else { radius };
            let component = if vertical { ux } else { uy };
            let offset = line - if vertical { self.origin.x } else { self.origin.y };
            for branch in 0..2 {
                let denominator = if branch == 1 {
                    component - 1.0
                } else {
                    component + 1.0
                };
                if denominator == 0.0 {
                    continue;
                }
                let t = offset / denominator;
                if !(t > 0.0) || t >= best {
                    continue;
                }
                let reached_x = self.origin.x + t * ux;
                let reached_y = self.origin.y + t * uy;
                let (foot_x, foot_y) = if vertical {
                    (line, reached_y)
                } else {
                    (reached_x, line)
                };
                if ((foot_x - reached_x).hypot(foot_y - reached_y) - t).abs()
                    > numeric::EDGE_EPSILON
                {
                    continue;
                }
                if !legal_set::contains(self.position, foot_x, foot_y) {
                    continue;
                }
                best = t;
            }
        }

        // Vertex candidates: the perpendicular bisector toward each vertex.
        for &(vertex, floor) in &self.vertex_order {
            if floor >= best {
                break;
            }
            let gx = vertex.x - self.origin.x;
            let gy = vertex.y - self.origin.y;
            let along = ux * gx + uy * gy;
            if !(along > 0.0) {
                continue;
            }
            let t = gx.mul_add(gx, gy * gy) / (2.0 * along);
            if t < best {
                best = t;
            }
        }

        best
    }

    /// Whether `point` lies inside this stone's settled region.
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        let dx = point.x - self.origin.x;
        let dy = point.y - self.origin.y;
        let distance = dx.hypot(dy);
        if distance < numeric::EDGE_EPSILON {
            return true;
        }
        if self.unbounded {
            return true;
        }
        let (ux, uy) = (dx / distance, dy / distance);
        // The ray leaves a convex board once; beyond that there is nothing to
        // find, so the exit distance is a sound cap.
        let cap = self.exit_distance(ux, uy);
        if distance >= cap {
            return false;
        }
        distance <= self.radius_along(ux, uy, cap)
    }

    /// The region's boundary as a closed polygon.
    ///
    /// Star-shaped about the stone (A16), so a sweep at increasing angle closes
    /// by construction and needs no topology repair. Subdivision concentrates
    /// points where the boundary bends, so a smooth arc costs few of them and a
    /// corner costs many.
    ///
    /// This is what makes the region affordable to rasterize: the boundary is
    /// solved once per stone, and pixels are then tested against a polygon
    /// instead of each running its own radial solve.
    #[must_use]
    pub fn contour(&self) -> Vec<Point> {
        self.contour_within(2.0e-5)
    }

    /// [`contour`](Self::contour) at a caller-chosen chord tolerance.
    ///
    /// The client subdivides to 2e-5 because it draws vectors that a viewer can
    /// zoom. A raster cannot show more than one pixel of detail, so a caller
    /// rendering at 128x128 -- where a pixel spans 1/128 -- can stop far
    /// earlier, and subdivision is most of the cost.
    #[must_use]
    pub fn contour_within(&self, tolerance: f64) -> Vec<Point> {
        let mut points = Vec::new();
        self.contour_within_into(tolerance, &mut points);
        points
    }

    /// [`contour_within`](Self::contour_within), writing into reusable storage.
    ///
    /// Rasterization traces one region per stone, so retaining the largest
    /// contour's capacity avoids allocating and growing another vector for
    /// every stone at every encoded search leaf.
    pub fn contour_within_into(&self, tolerance: f64, points: &mut Vec<Point>) {
        const BASE_RAYS: usize = 16;
        const MAX_DEPTH: u32 = 9;
        let tolerance = tolerance.max(0.0);
        points.clear();

        if self.unbounded {
            // Every point is settled; the caller wants the whole board.
            points.extend([
                Point::new(-0.02, -0.02),
                Point::new(1.02, -0.02),
                Point::new(1.02, 1.02),
                Point::new(-0.02, 1.02),
            ]);
            return;
        }

        let ray = |angle: f64| {
            let (uy, ux) = angle.sin_cos();
            let cap = self.exit_distance(ux, uy) + 0.02;
            (ux, uy, self.radius_along(ux, uy, cap))
        };
        let at = |t: f64, ux: f64, uy: f64| {
            Point::new(self.origin.x + t * ux, self.origin.y + t * uy)
        };

        #[allow(clippy::too_many_arguments)]
        fn flatten(
            region: &SettledRegion,
            loop_points: &mut Vec<Point>,
            angle_a: f64,
            ta: f64,
            ax: f64,
            ay: f64,
            angle_b: f64,
            tb: f64,
            bx: f64,
            by: f64,
            depth: u32,
            tolerance: f64,
            max_depth: u32,
        ) {
            let middle = 0.5 * (angle_a + angle_b);
            let (uy, ux) = middle.sin_cos();
            let cap = region.exit_distance(ux, uy) + 0.02;
            let tm = region.radius_along(ux, uy, cap);
            let chord_x = 0.5
                * ((region.origin.x + ta * ax) + (region.origin.x + tb * bx));
            let chord_y = 0.5
                * ((region.origin.y + ta * ay) + (region.origin.y + tb * by));
            let deviation = (region.origin.x + tm * ux - chord_x)
                .hypot(region.origin.y + tm * uy - chord_y);
            if depth >= max_depth || deviation <= tolerance {
                loop_points.push(Point::new(
                    region.origin.x + tm * ux,
                    region.origin.y + tm * uy,
                ));
                loop_points.push(Point::new(
                    region.origin.x + tb * bx,
                    region.origin.y + tb * by,
                ));
                return;
            }
            flatten(
                region, loop_points, angle_a, ta, ax, ay, middle, tm, ux, uy,
                depth + 1, tolerance, max_depth,
            );
            flatten(
                region, loop_points, middle, tm, ux, uy, angle_b, tb, bx, by,
                depth + 1, tolerance, max_depth,
            );
        }

        let mut rays = [(0.0, 0.0, 0.0, 0.0); BASE_RAYS + 1];
        for k in 0..=BASE_RAYS {
            let angle = std::f64::consts::TAU * k as f64 / BASE_RAYS as f64;
            let (ux, uy, t) = ray(angle);
            rays[k] = (angle, t, ux, uy);
        }

        points.push(at(rays[0].1, rays[0].2, rays[0].3));
        for k in 0..BASE_RAYS {
            let (angle_a, ta, ax, ay) = rays[k];
            let (angle_b, tb, bx, by) = rays[k + 1];
            flatten(
                self, points, angle_a, ta, ax, ay, angle_b, tb, bx, by, 0,
                tolerance, MAX_DEPTH,
            );
        }
        points.pop(); // the closing point repeats the first
    }

    fn exit_distance(&self, ux: f64, uy: f64) -> f64 {
        let mut t = f64::INFINITY;
        if ux > 0.0 {
            t = t.min((1.0 - self.origin.x) / ux);
        } else if ux < 0.0 {
            t = t.min(-self.origin.x / ux);
        }
        if uy > 0.0 {
            t = t.min((1.0 - self.origin.y) / uy);
        } else if uy < 0.0 {
            t = t.min(-self.origin.y / uy);
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Color, Stone};

    fn scattered(count: usize) -> Position {
        let radius = 1.0 / 18.0;
        let mut placed: Vec<Stone> = Vec::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut attempts = 0;
        while placed.len() < count && attempts < 20_000 {
            attempts += 1;
            let x = radius + next() * (1.0 - 2.0 * radius);
            let y = radius + next() * (1.0 - 2.0 * radius);
            if placed
                .iter()
                .all(|s| (s.x - x).hypot(s.y - y) >= 2.0 * radius)
            {
                let colour = if placed.len() % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                };
                placed.push(Stone::new(x, y, colour));
            }
        }
        Position::new(radius, placed, Color::Black)
    }

    /// The radial solve must agree with the definition it is derived from.
    ///
    /// `||x - s|| <= dist(x, L)` for the nearest stone is the whole predicate;
    /// everything in this module exists only to evaluate it without a distance
    /// query per pixel. A fast answer that disagrees is worse than no channel.
    #[test]
    fn radial_solve_agrees_with_the_distance_definition() {
        for count in [1usize, 5, 12, 24, 35] {
            let position = scattered(count);
            let vertices = legal_set::vertices(&position);
            let regions: Vec<_> = (0..position.stones().len())
                .map(|index| SettledRegion::new(&position, index, &vertices))
                .collect();
            let grid = 96;
            let mut disagreements = 0;
            for row in 0..grid {
                let y = (f64::from(row) + 0.5) / f64::from(grid);
                for column in 0..grid {
                    let x = (f64::from(column) + 0.5) / f64::from(grid);
                    let point = Point::new(x, y);
                    let free = legal_set::distance(&position, point, Some(&vertices));
                    // A15: the union over stones, not only the nearest one.
                    let exact = position
                        .stones()
                        .iter()
                        .any(|s| (x - s.x).hypot(y - s.y) <= free);
                    let solved = regions.iter().any(|region| region.contains(point));
                    if exact != solved {
                        disagreements += 1;
                    }
                }
            }
            assert_eq!(
                disagreements, 0,
                "{count} stones: radial solve disagreed with the definition at \
                 {disagreements} of {} sample points",
                grid * grid
            );
        }
    }
}
