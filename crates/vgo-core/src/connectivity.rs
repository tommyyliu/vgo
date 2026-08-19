//! Which same-colour pairs an enemy cannot wedge apart.
//!
//! Two stones of one colour are **connected** when no enemy placement can
//! separate their cells into different groups. It is not a rule -- nothing here
//! changes what is legal or what is captured -- but it is the fact a player is
//! actually reading when they look at a shape, and it is hard to see from a
//! Voronoi diagram alone. `voronoigo.com` draws it as a line between the two
//! stones.
//!
//! # Three answers, and why the third is not a verdict
//!
//! [`CutKind::TooFar`] means the question was declined, not answered either way.
//! Past [`MAX_PAIR_CUT_DISTANCE`] an enemy stone fits *between* the two ends and
//! the two-placement search below stops being trustworthy: the best first
//! placement can crowd its partner out of a spot it would have had, so the
//! search can report a connection that is not there. A false "connected" is the
//! one answer this must not give, so it declines instead.
//!
//! # The search
//!
//! An enemy needs two placements to cut, one on each side of the line, so the
//! test plays the two best ones and asks whether the pair came apart:
//!
//! 1. Below [`SAFE_PAIR_DISTANCE`] there is no room for a cutting stone at all,
//!    and the answer is [`Connected`](CutKind::Connected) without any geometry.
//!    This is deliberately optimistic -- it assumes a cutter respects the dead
//!    zone, which a forced eye can break -- because paying for the full test on
//!    every close pair costs more than the cases it would catch.
//! 2. Otherwise place an enemy stone at the legal point nearest the midpoint,
//!    then a second at the legal point nearest the midpoint *given the first*,
//!    and ask whether `a` and `b` still share a group.
//!
//! This is a reimplementation of the idea, not a port. The reference is
//! `csun/voronoi-go-rs`, which is AGPL; a model input has to be computable
//! wherever the model runs, including a browser bundle, so linking it is not
//! available to us. Its constants are documented as experimental rather than
//! derived, so ours will not agree with it everywhere. That is acceptable here
//! in a way it would not be for a rule: this channel informs the network, and a
//! disagreement costs accuracy rather than legality.

use crate::{Color, Point, Position, Stone, nearest_legal_placement, numeric::length};

/// Furthest apart a pair may be for the question to be answered at all.
///
/// `2*sqrt(3)` stone radii, and derived rather than tuned: a cutting stone keeps
/// a diameter from both ends, so the nearest it can come to the line between
/// them is `sqrt(4 - (d/2)^2)` radii, which falls to one radius exactly at
/// `d = 2*sqrt(3)`. Past that an enemy fits between the ends.
pub const MAX_PAIR_CUT_DISTANCE: f64 = 3.464_101_615_137_754_6;

/// Below this separation a pair is uncuttable outright: the dead zone leaves no
/// room for a stone that could wedge between them.
///
/// `2*sqrt(2)` stone radii. Experimental in the reference, and kept here for the
/// same reason -- it is where the cheap answer stops being safe, not a
/// consequence of anything above it.
pub const SAFE_PAIR_DISTANCE: f64 = 2.828_427_124_746_190_3;

/// What the geometry says about the line between two same-colour stones.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutKind {
    /// No enemy pair separates them.
    Connected,
    /// An enemy pair does. A cut target.
    Cuttable,
    /// Further apart than [`MAX_PAIR_CUT_DISTANCE`], so nothing was computed.
    /// Not a verdict in either direction.
    TooFar,
}

/// Whether an enemy can wedge `a` and `b` apart, by stone index.
///
/// Returns `None` when the question is malformed rather than unanswerable: the
/// same stone twice, or two colours. Those are caller errors, and folding them
/// into [`CutKind`] is what would let one be acted on by mistake.
#[must_use]
pub fn pair_cut(position: &Position, a: usize, b: usize) -> Option<CutKind> {
    let stones = position.stones();
    if a == b || a >= stones.len() || b >= stones.len() {
        return None;
    }
    let (first, second) = (stones[a], stones[b]);
    if first.color != second.color {
        return None;
    }
    let radius = position.radius();
    let separation = length(first.x - second.x, first.y - second.y);
    if separation > MAX_PAIR_CUT_DISTANCE * radius {
        return Some(CutKind::TooFar);
    }
    if separation < SAFE_PAIR_DISTANCE * radius {
        return Some(CutKind::Connected);
    }

    let midpoint = Point::new((first.x + second.x) / 2.0, (first.y + second.y) / 2.0);
    let enemy = first.color.other();
    let mut probe = position.clone();
    // Two placements, the second aware of the first. If either has nowhere to
    // go, the enemy cannot mount the cut.
    for _ in 0..2 {
        let nearest = nearest_legal_placement(&probe, midpoint);
        if !nearest.legal {
            break;
        }
        let mut with = probe.stones().to_vec();
        with.push(Stone::new(nearest.point.x, nearest.point.y, enemy));
        probe = probe.with_stones_public(with);
    }
    if probe.stones().len() == stones.len() {
        return Some(CutKind::Connected);
    }

    // Did the pair come apart? Two Voronoi cells belong to one group only if
    // they are adjacent somewhere, so this asks whether the bisector between
    // them still owns any board -- which is a local question about half-planes,
    // not a reason to rebuild the diagram.
    if shares_a_boundary(&probe, first, second) {
        Some(CutKind::Connected)
    } else {
        Some(CutKind::Cuttable)
    }
}

/// Whether `a` and `b` own a positive-length stretch of their common bisector.
///
/// That is exactly Voronoi adjacency, and it costs one pass over the stones
/// rather than a diagram. Walk the bisector as `m + t*u`; every other stone
/// clips `t` to a half-line, since `|x-a| <= |x-c|` is linear in `t`, and the
/// board clips it to a box. If what survives has length, the two cells touch.
///
/// Rebuilding the diagram instead measured 0.45 ms per position at 28 stones and
/// 5.9 ms at 52 -- more than the whole raster, and eight times it, for one
/// channel.
fn shares_a_boundary(position: &Position, a: Stone, b: Stone) -> bool {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let span = length(dx, dy);
    if span <= 0.0 {
        return false;
    }
    let midpoint = Point::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    // Unit vector along the bisector, perpendicular to a->b.
    let (ux, uy) = (-dy / span, dx / span);

    // slope * t <= bound, applied to the running interval. A free function
    // rather than a closure so the loop below can still read the interval to
    // bail early.
    fn clip(low: &mut f64, high: &mut f64, slope: f64, bound: f64) {
        if slope.abs() < 1.0e-15 {
            // No dependence on t: either every t satisfies it, or none does.
            if bound < 0.0 {
                *low = 1.0;
                *high = 0.0;
            }
        } else if slope > 0.0 {
            *high = high.min(bound / slope);
        } else {
            *low = low.max(bound / slope);
        }
    }
    let (mut low, mut high) = (f64::NEG_INFINITY, f64::INFINITY);

    // The board, so an unbounded bisector does not count as shared edge off it.
    clip(&mut low, &mut high, ux, 1.0 - midpoint.x);
    clip(&mut low, &mut high, -ux, midpoint.x);
    clip(&mut low, &mut high, uy, 1.0 - midpoint.y);
    clip(&mut low, &mut high, -uy, midpoint.y);

    let square = |p: f64, q: f64| p * p + q * q;
    for other in position.stones() {
        let same = (other.x - a.x).abs() < f64::EPSILON && (other.y - a.y).abs() < f64::EPSILON;
        let twin = (other.x - b.x).abs() < f64::EPSILON && (other.y - b.y).abs() < f64::EPSILON;
        if same || twin {
            continue;
        }
        // |x-a|^2 <= |x-other|^2  =>  2 x.(other-a) <= |other|^2 - |a|^2
        let (cx, cy) = (other.x - a.x, other.y - a.y);
        let bound = square(other.x, other.y) - square(a.x, a.y)
            - 2.0 * (midpoint.x * cx + midpoint.y * cy);
        clip(&mut low, &mut high, 2.0 * (ux * cx + uy * cy), bound);
        if low >= high {
            return false;
        }
    }
    high - low > crate::COORDINATE_EPSILON
}

/// Every same-colour pair/// Every same-colour pair the geometry says cannot be separated.
///
/// Only pairs within [`MAX_PAIR_CUT_DISTANCE`] are considered, which is what
/// keeps this near-linear rather than quadratic in a real position: a stone has
/// a bounded number of neighbours that close.
#[must_use]
pub fn connected_pairs(position: &Position) -> Vec<(usize, usize)> {
    let stones = position.stones();
    let reach = MAX_PAIR_CUT_DISTANCE * position.radius();
    let mut pairs = Vec::new();
    for a in 0..stones.len() {
        for b in (a + 1)..stones.len() {
            if stones[a].color != stones[b].color {
                continue;
            }
            if length(stones[a].x - stones[b].x, stones[a].y - stones[b].y) > reach {
                continue;
            }
            if pair_cut(position, a, b) == Some(CutKind::Connected) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

/// The colour of a connected pair, for a caller painting them.
#[must_use]
pub fn pair_color(position: &Position, pair: (usize, usize)) -> Color {
    position.stones()[pair.0].color
}
