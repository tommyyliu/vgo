use crate::{Point, Position, Stone, numeric};

use super::buckets::StoneBuckets;
use super::{LegalSetIndex, contains_with, vertices};

/// Where a placement lands after projecting it onto the legal set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    pub point: Point,
    pub legal: bool,
    pub snapped: bool,
}

/// `value` as an integer ordered exactly as `f64::total_cmp` orders floats.
#[inline]
fn total_order_key(value: f64) -> i64 {
    let bits = value.to_bits() as i64;
    bits ^ (((bits >> 63) as u64) >> 1) as i64
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
    nearest_from(position, point, known_vertices, None, None)
}

/// [`nearest_with`], against a prebuilt index.
#[must_use]
pub fn nearest_indexed(position: &Position, point: Point, index: &LegalSetIndex) -> Nearest {
    nearest_from(position, point, Some(&index.vertices), index.buckets.as_ref(), None)
}

/// [`nearest_indexed`] for a caller that only accepts answers within `reach` of
/// `point`.
///
/// When the answer is within `reach`, this returns exactly what
/// [`nearest_indexed`] returns. Otherwise it may return `legal: false` instead,
/// so a result is only meaningful once checked against `reach`.
#[must_use]
pub fn nearest_indexed_within(
    position: &Position,
    point: Point,
    index: &LegalSetIndex,
    reach: f64,
) -> Nearest {
    nearest_from(position, point, Some(&index.vertices), index.buckets.as_ref(), Some(reach))
}

/// The geometry of snapping: how far candidates sit off the constraints that
/// produced them, and the box they are clamped to.
///
/// Each candidate is pushed `SNAP_MARGIN` past its constraint rather than placed
/// exactly on it. A point on the boundary is legal only by the
/// `2r - COORDINATE_EPSILON` slack in `clear_of_stones`, which makes its legality a
/// question of floating-point association order, and two implementations that
/// associate differently then disagree about a move one of them just proposed.
struct Snap {
    diameter: f64,
    inset: f64,
}

impl Snap {
    fn new(radius: f64) -> Self {
        Self {
            diameter: 2.0 * radius + numeric::SNAP_MARGIN,
            // A radius leaving no inset at all admits no legal point. The margin
            // must not invert the clamp range on the way to reporting that.
            inset: (radius + numeric::SNAP_MARGIN).min(0.5),
        }
    }

    fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.inset.min(1.0 - self.inset), self.inset.max(1.0 - self.inset))
    }

    fn clamped_query(&self, point: Point) -> Point {
        Point::new(self.clamp(point.x), self.clamp(point.y))
    }

    /// The query pushed one diameter out from `stone`, along the ray to the query.
    fn off_stone(&self, point: Point, stone: Stone, emit: &mut impl FnMut(Point)) {
        let (dx, dy) = (point.x - stone.x, point.y - stone.y);
        let radial = numeric::length(dx, dy);
        // A query exactly on a stone's centre has no ray to push along, so try
        // the four axes instead of dividing by zero.
        if radial < numeric::EDGE_EPSILON {
            for (ux, uy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                emit(Point::new(
                    self.clamp(stone.x + self.diameter * ux),
                    self.clamp(stone.y + self.diameter * uy),
                ));
            }
        } else {
            emit(Point::new(
                self.clamp(stone.x + self.diameter * dx / radial),
                self.clamp(stone.y + self.diameter * dy / radial),
            ));
        }
    }

    /// The query projected onto each side of the inset.
    fn on_sides(&self, point: Point) -> [Point; 4] {
        [
            Point::new(self.inset, self.clamp(point.y)),
            Point::new(1.0 - self.inset, self.clamp(point.y)),
            Point::new(self.clamp(point.x), self.inset),
            Point::new(self.clamp(point.x), 1.0 - self.inset),
        ]
    }
}

/// The result of a search that ended on `found`, or on nothing at all.
fn snapped(
    position: &Position,
    point: Point,
    found: Option<Point>,
    buckets: Option<&StoneBuckets>,
) -> Nearest {
    match found {
        Some(found) => Nearest {
            point: clear_by_margin(position, found, buckets),
            legal: true,
            snapped: true,
        },
        // No legal point exists anywhere: the board is full, and the caller must
        // treat the move as unplayable rather than substituting one.
        None => Nearest { point, legal: false, snapped: false },
    }
}

/// [`nearest_from`] over only the candidates that can land within `reach`, or
/// `None` when an exact tie needs the full scan to decide it. Proof: [E6].
fn nearest_bounded(
    position: &Position,
    point: Point,
    reach: f64,
    buckets: &StoneBuckets,
    known_vertices: Option<&[Point]>,
    snap: &Snap,
) -> Option<Nearest> {
    let padded = reach + 64.0 * numeric::SNAP_MARGIN;
    let limit = padded * padded;
    let mut near: Vec<(i64, Point)> = Vec::with_capacity(16);
    let mut consider = |candidate: Point| {
        let (dx, dy) = (candidate.x - point.x, candidate.y - point.y);
        let square = dx.mul_add(dx, dy * dy);
        if square <= limit {
            near.push((total_order_key(square), candidate));
        }
    };
    consider(snap.clamped_query(point));
    buckets.for_each_near(point, padded + snap.diameter, &mut |index| {
        snap.off_stone(point, position.stones()[index], &mut consider);
    });
    snap.on_sides(point).into_iter().for_each(&mut consider);
    known_vertices?.iter().copied().for_each(&mut consider);

    near.sort_unstable_by_key(|&(key, _)| key);
    if near.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return None;
    }
    let found = near
        .iter()
        .map(|&(_, candidate)| candidate)
        .find(|candidate| contains_with(position, Some(buckets), candidate.x, candidate.y));
    Some(snapped(position, point, found, Some(buckets)))
}

fn nearest_from(
    position: &Position,
    point: Point,
    known_vertices: Option<&[Point]>,
    buckets: Option<&StoneBuckets>,
    within: Option<f64>,
) -> Nearest {
    if contains_with(position, buckets, point.x, point.y) {
        return Nearest {
            point,
            legal: true,
            snapped: false,
        };
    }
    let snap = Snap::new(position.radius());
    if let (Some(reach), Some(buckets)) = (within, buckets)
        && let Some(answer) = nearest_bounded(position, point, reach, buckets, known_vertices, &snap)
    {
        return answer;
    }

    let mut candidates = vec![snap.clamped_query(point)];
    for &stone in position.stones() {
        snap.off_stone(point, stone, &mut |candidate| candidates.push(candidate));
    }
    candidates.extend(snap.on_sides(point));
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
    //
    // Keys are `f64::total_cmp`'s integer form, not floats: a NaN key must not
    // silently win the minimum, candidates are clamped but not proven finite, and
    // a minimum over integers vectorizes where a comparator closure does not.
    let mut keys: Vec<i64> = candidates
        .iter()
        .map(|candidate| {
            let dx = candidate.x - point.x;
            let dy = candidate.y - point.y;
            total_order_key(dx.mul_add(dx, dy * dy))
        })
        .collect();
    let mut best = None;
    let mut remaining = keys.len();
    while remaining > 0 {
        let unscanned = &keys[..remaining];
        let smallest = unscanned.iter().copied().fold(i64::MAX, i64::min);
        // The first of equals, which is what the choice has always been.
        let index = unscanned
            .iter()
            .position(|&key| key == smallest)
            .expect("remaining is non-zero");
        let candidate = candidates[index];
        if contains_with(position, buckets, candidate.x, candidate.y) {
            best = Some(candidate);
            break;
        }
        // Retire the rejected candidate by swapping the unscanned tail over it,
        // so the next pass never reconsiders it.
        remaining -= 1;
        keys.swap(index, remaining);
        candidates.swap(index, remaining);
    }
    snapped(position, point, best, buckets)
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
pub(super) fn clear_by_margin(position: &Position, point: Point, buckets: Option<&StoneBuckets>) -> Point {
    let radius = position.radius();
    let target = 2.0 * radius + numeric::SNAP_MARGIN;
    let mut moved = point;
    // One pass per stone, nearest first: pushing off one constraint can bring a
    // point nearer another, and the margin is small enough that a fixed number
    // of passes settles it.
    for _ in 0..4 {
        // Ties go to the lower index, as they do scanning stones in order, which
        // is what lets the buckets visit them in any order.
        let mut worst: Option<(f64, usize)> = None;
        let mut consider = |index: usize| {
            let stone = &position.stones()[index];
            let (dx, dy) = (moved.x - stone.x, moved.y - stone.y);
            if numeric::length_below(dx, dy, target) {
                let distance = numeric::length(dx, dy);
                if worst.is_none_or(|(seen, first)| {
                    distance < seen || (distance == seen && index < first)
                }) {
                    worst = Some((distance, index));
                }
            }
        };
        match buckets {
            Some(buckets) => buckets.for_each_near(moved, target, &mut consider),
            None => (0..position.stones().len()).for_each(&mut consider),
        }
        let Some((distance, index)) = worst else {
            break;
        };
        let stone = &position.stones()[index];
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
    if contains_with(position, buckets, clamped.x, clamped.y) {
        clamped
    } else {
        point
    }
}
