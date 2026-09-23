//! Stable placement witnesses. Availability is a blocker count, independent of
//! Voronoi vertex motion. All guarantees are sufficient; absent guarantees fall
//! back to the normal analytic settlement search.
use crate::{MoveError, MoveResult, Point, Position, Ruleset, Settlement, Stone, numeric};
use std::{borrow::Cow, cell::RefCell, collections::HashSet};

#[derive(Clone, Copy, Debug)]
pub enum SupportPolicy {
    All,
    /// Retain up to four spatially separated points per supporting stone.
    DiverseFour,
    AllWithNegatives,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SupportStats {
    pub stages: usize,
    pub groups: usize,
    pub certified_groups: usize,
    pub reactivated_points: usize,
    pub eligible_negative_cells: usize,
}

pub struct SupportPosition<'a> {
    parent: Cow<'a, Position>,
    points: Vec<Point>,
    blockers: Vec<usize>,
    /// Reverse edges: points made available when this stone disappears.
    blocked_by: Vec<Vec<usize>>,
    supports: Vec<Vec<usize>>,
    support_squared: f64,
    negative_polygons: Vec<Option<Vec<Point>>>,
}

fn blocks(stone: Stone, point: Point, minimum_squared: f64) -> bool {
    let dx = point.x - stone.x;
    let dy = point.y - stone.y;
    dx.mul_add(dx, dy * dy) < minimum_squared
}

impl<'a> SupportPosition<'a> {
    pub fn new(parent: &'a Position, policy: SupportPolicy) -> Self {
        // The mathematical bound is d² < 8r². Existing placement and setup
        // predicates tolerate a slightly shorter separation. Use a smaller
        // separation bound and an outward-rounded distance upper bound; decline
        // this fast path at radii comparable to the coordinate tolerance.
        let clearance = 2.0 * parent.radius() - 4.0 * numeric::COORDINATE_EPSILON;
        let support_squared = if clearance > 0.0 {
            2.0 * (clearance * clearance).next_down()
        } else {
            0.0
        };
        let enabled = parent.ruleset() == Ruleset::Vgo
            && parent.radius() > 8.0 * numeric::COORDINATE_EPSILON
            && parent.validate().is_playable();
        let points = if enabled {
            candidates(parent)
        } else {
            Vec::new()
        };
        let mut blockers = vec![0; points.len()];
        let mut blocked_by = vec![Vec::new(); parent.stones().len()];
        let mut supports = vec![Vec::new(); parent.stones().len()];
        let minimum = 2.0 * parent.radius() - numeric::COORDINATE_EPSILON;
        for (i, &s) in parent.stones().iter().enumerate() {
            for (j, &p) in points.iter().enumerate() {
                if blocks(s, p, minimum * minimum) {
                    blockers[j] += 1;
                    blocked_by[i].push(j);
                }
                if numeric::squared_distance_upper(Point::new(s.x, s.y), p) < support_squared {
                    supports[i].push(j);
                }
            }
        }
        if matches!(policy, SupportPolicy::DiverseFour) {
            for list in &mut supports {
                select_diverse(list, &points, &blockers);
            }
        }
        let negative_polygons = if enabled && matches!(policy, SupportPolicy::AllWithNegatives) {
            let analysis = crate::Analysis::new(parent);
            analysis
                .geometry
                .cells
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let s = parent.stones()[i];
                    let alive = cell.polygon.iter().any(|&v| {
                        crate::legal_set::escape_witness(
                            parent,
                            v,
                            Point::new(s.x, s.y),
                            Some(&analysis.legal_vertices),
                        )
                        .is_some()
                    });
                    (!alive).then(|| cell.polygon.clone())
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            parent: Cow::Borrowed(parent),
            points,
            blockers,
            blocked_by,
            supports,
            support_squared,
            negative_polygons,
        }
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Move the prepared arrays into an independently owned cache entry.
    pub fn into_owned(self) -> SupportPosition<'static> {
        SupportPosition {
            parent: Cow::Owned(self.parent.into_owned()),
            points: self.points,
            blockers: self.blockers,
            blocked_by: self.blocked_by,
            supports: self.supports,
            support_squared: self.support_squared,
            negative_polygons: self.negative_polygons,
        }
    }

    /// Exact cache identity, including metadata. Hash matches alone never suffice.
    pub fn matches(&self, other: &Position) -> bool {
        let parent = &self.parent;
        parent.radius().to_bits() == other.radius().to_bits()
            && parent.komi().to_bits() == other.komi().to_bits()
            && parent.ruleset() == other.ruleset()
            && parent.to_move() == other.to_move()
            && parent.phase() == other.phase()
            && parent.consecutive_passes() == other.consecutive_passes()
            && parent.stones().len() == other.stones().len()
            && parent.stones().iter().zip(other.stones()).all(|(a, b)| {
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.color == b.color
            })
    }

    /// Reserved retained payload for an owned entry, excluding allocator headers.
    /// The owned parent is cloned into an exact-length stone vector.
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + std::mem::size_of_val(self.parent.stones())
            + self.points.capacity() * size_of::<Point>()
            + self.blockers.capacity() * size_of::<usize>()
            + (self.blocked_by.capacity() + self.supports.capacity()) * size_of::<Vec<usize>>()
            + self
                .blocked_by
                .iter()
                .chain(&self.supports)
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + self.negative_polygons.capacity() * size_of::<Option<Vec<Point>>>()
            + self
                .negative_polygons
                .iter()
                .flatten()
                .map(|v| v.capacity() * size_of::<Point>())
                .sum::<usize>()
    }
    pub fn dormant_count(&self) -> usize {
        self.blockers.iter().filter(|&&n| n > 0).count()
    }
    pub fn support_count(&self) -> usize {
        self.supports.iter().map(Vec::len).sum()
    }

    pub fn negative_count(&self) -> usize {
        self.negative_polygons
            .iter()
            .filter(|p| p.is_some())
            .count()
    }

    pub fn place(&self, x: f64, y: f64) -> Result<MoveResult, MoveError> {
        self.place_profiled(x, y).0
    }

    pub fn place_profiled(&self, x: f64, y: f64) -> (Result<MoveResult, MoveError>, SupportStats) {
        let stats = RefCell::new(SupportStats::default());
        let added = Stone::new(x, y, self.parent.to_move());
        let result = crate::game::place_with(&self.parent, x, y, |current| {
            Settlement::build_with_facts(current, |geometry| {
                let mut counts = self.blockers.clone();
                let mut mapping = vec![None; self.parent.stones().len()];
                // Capture removal preserves order and the appended stone is last.
                let mut cursor = 0;
                for (old, s) in self.parent.stones().iter().enumerate() {
                    if current.stones().get(cursor) == Some(s) {
                        mapping[old] = Some(cursor);
                        cursor += 1;
                    } else {
                        for &point in &self.blocked_by[old] {
                            counts[point] -= 1;
                        }
                    }
                }
                let added_index = (current.stones().get(cursor) == Some(&added)).then_some(cursor);
                let minimum = 2.0 * current.radius() - numeric::COORDINATE_EPSILON;
                if added_index.is_some() {
                    for (count, &p) in counts.iter_mut().zip(&self.points) {
                        *count += usize::from(blocks(added, p, minimum * minimum));
                    }
                }
                let mut alive = HashSet::new();
                for (old, mapped) in mapping.iter().enumerate() {
                    let Some(index) = *mapped else {
                        continue;
                    };
                    let group = geometry.groups[index];
                    if !alive.contains(&group) && self.supports[old].iter().any(|&p| counts[p] == 0)
                    {
                        alive.insert(group);
                    }
                }
                if let Some(index) = added_index {
                    let group = geometry.groups[index];
                    if !alive.contains(&group)
                        && counts.iter().zip(&self.points).any(|(&count, &p)| {
                            count == 0
                                && numeric::squared_distance_upper(Point::new(x, y), p)
                                    < self.support_squared
                        })
                    {
                        alive.insert(group);
                    }
                }
                let mut negative = Vec::new();
                // Exact rules allow every old negative cell through insertion.
                // This prototype additionally requires identical polygon values
                // so it does not infer containment from rounded new vertices.
                // Any removal invalidates the whole negative cache.
                if !self.negative_polygons.is_empty() && mapping.iter().all(Option::is_some) {
                    negative.resize(current.stones().len(), false);
                    for (old, polygon) in self.negative_polygons.iter().enumerate() {
                        if let Some(polygon) = polygon {
                            let new = mapping[old].expect("all old stones survive");
                            negative[new] = geometry.cells[new].polygon == *polygon;
                        }
                    }
                }
                let mut stats = stats.borrow_mut();
                stats.eligible_negative_cells += negative.iter().filter(|&&v| v).count();
                stats.stages += 1;
                // Geometry group IDs are their minimum member index.
                stats.groups += geometry
                    .groups
                    .iter()
                    .enumerate()
                    .filter(|(i, g)| *i == **g)
                    .count();
                stats.certified_groups += alive.len();
                stats.reactivated_points += counts
                    .iter()
                    .zip(&self.blockers)
                    .filter(|(now, before)| **now == 0 && **before > 0)
                    .count();
                (alive, negative)
            })
        });
        (result, stats.into_inner())
    }
}

fn select_diverse(list: &mut Vec<usize>, points: &[Point], blockers: &[usize]) {
    if list.len() <= 4 {
        return;
    }
    let mut selected = Vec::new();
    while selected.len() < 4 {
        // Prefer currently available points, then maximize separation from the
        // selected set. Blocked candidates remain eligible when none are free.
        let next = list
            .iter()
            .copied()
            .filter(|p| !selected.contains(p))
            .max_by(|&a, &b| {
                let spread = |p: usize| {
                    selected
                        .iter()
                        .map(|&s| points[p].distance(points[s]))
                        .fold(f64::INFINITY, f64::min)
                };
                blockers[b]
                    .cmp(&blockers[a])
                    .then_with(|| spread(a).total_cmp(&spread(b)))
                    .then_with(|| b.cmp(&a))
            })
            .expect("at least five candidates");
        selected.push(next);
    }
    *list = selected;
}

fn candidates(position: &Position) -> Vec<Point> {
    let r = position.radius();
    let diameter = 2.0 * r;
    let mut points = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |p: Point| {
        // Keep candidate coordinates strictly inside the inset. Declining a
        // rounded boundary point loses only a fast certificate, never a move.
        if p.x.is_finite()
            && p.y.is_finite()
            && p.x >= r
            && p.x <= 1.0 - r
            && p.y >= r
            && p.y <= 1.0 - r
            && seen.insert((p.x.to_bits(), p.y.to_bits()))
        {
            points.push(p);
        }
    };
    for x in [r, 1.0 - r] {
        for y in [r, 1.0 - r] {
            push(Point::new(x, y));
        }
    }
    for s in position.stones() {
        // Full circles without intersections still need representative points.
        for (ux, uy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            push(Point::new(s.x + diameter * ux, s.y + diameter * uy));
        }
        for x in [r, 1.0 - r] {
            let d = diameter.mul_add(diameter, -(x - s.x).powi(2));
            if d >= 0.0 {
                for sign in [-1.0, 1.0] {
                    push(Point::new(x, s.y + sign * d.sqrt()));
                }
            }
        }
        for y in [r, 1.0 - r] {
            let d = diameter.mul_add(diameter, -(y - s.y).powi(2));
            if d >= 0.0 {
                for sign in [-1.0, 1.0] {
                    push(Point::new(s.x + sign * d.sqrt(), y));
                }
            }
        }
    }
    for (i, a) in position.stones().iter().enumerate() {
        for b in &position.stones()[i + 1..] {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let d = numeric::length(dx, dy);
            if d == 0.0 || d > 2.0 * diameter {
                continue;
            }
            let h2 = diameter.mul_add(diameter, -(d / 2.0).powi(2));
            if h2 < 0.0 {
                continue;
            }
            let h = h2.sqrt();
            for sign in [-1.0, 1.0] {
                push(Point::new(
                    (a.x + b.x) / 2.0 + sign * dy / d * h,
                    (a.y + b.y) / 2.0 - sign * dx / d * h,
                ));
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Color;

    #[test]
    fn capture_reactivates_previously_blocked_points() {
        let p = Position::new(
            0.25,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let support = SupportPosition::new(&p, SupportPolicy::All);
        assert!(support.dormant_count() > 0);
        let (result, stats) = support.place_profiled(0.75, 0.75);
        super::super::assert_same(&crate::place(&p, 0.75, 0.75), &result);
        assert!(stats.reactivated_points > 0);
        assert!(stats.certified_groups > 0);
        assert_eq!(result.unwrap().captured, 2);
        let negative = SupportPosition::new(&p, SupportPolicy::AllWithNegatives);
        assert!(negative.negative_count() > 0);
        let (result, stats) = negative.place_profiled(0.75, 0.75);
        super::super::assert_same(&crate::place(&p, 0.75, 0.75), &result);
        assert_eq!(
            stats.eligible_negative_cells, 1,
            "only the unchanged parent cell before capture is reusable"
        );
    }

    #[test]
    fn full_circle_has_support_without_intersection_vertices() {
        let p = Position::new(0.05, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
        let support = SupportPosition::new(&p, SupportPolicy::All);
        assert_eq!(support.support_count(), 4);
        let (result, stats) = support.place_profiled(0.6, 0.5);
        super::super::assert_same(&crate::place(&p, 0.6, 0.5), &result);
        assert!(stats.certified_groups > 0);
    }

    #[test]
    fn support_variants_match_on_independent_seeded_trajectories() {
        let mut checked = 0;
        let mut captures = 0;
        for radius in [0.1, 1.0 / 18.0, 1.0 / 38.0] {
            for seed in [3u64, 17, 53, 99] {
                let mut rng = seed;
                let mut random = || {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng >> 11) as f64 / (1u64 << 53) as f64
                };
                let mut parent = Position::new(radius, vec![], Color::Black).with_komi(0.104);
                for _ in 0..64 {
                    let mut points = Vec::new();
                    for _ in 0..128 {
                        let p = Point::new(
                            radius + random() * (1.0 - 2.0 * radius),
                            radius + random() * (1.0 - 2.0 * radius),
                        );
                        if crate::is_legal_placement(&parent, p.x, p.y) {
                            points.push(p);
                            break;
                        }
                    }
                    let vertices = crate::legal_set_vertices(&parent);
                    if !vertices.is_empty() {
                        points.push(vertices[(random() * vertices.len() as f64) as usize]);
                    }
                    if points.is_empty() {
                        break;
                    }
                    let all = SupportPosition::new(&parent, SupportPolicy::All);
                    let four = SupportPosition::new(&parent, SupportPolicy::DiverseFour);
                    let negative = SupportPosition::new(&parent, SupportPolicy::AllWithNegatives);
                    let mut next = None;
                    for p in points {
                        let expected = crate::place(&parent, p.x, p.y);
                        for cache in [&all, &four, &negative] {
                            super::super::assert_same(&expected, &cache.place(p.x, p.y));
                        }
                        let result = expected.unwrap();
                        captures += usize::from(result.captured > 0);
                        checked += 1;
                        next = Some(result.position);
                    }
                    parent = next.unwrap();
                    if parent.phase() == crate::Phase::Finished {
                        break;
                    }
                }
            }
        }
        assert!(checked > 500, "corpus unexpectedly small: {checked}");
        assert!(captures > 0);
    }
}
