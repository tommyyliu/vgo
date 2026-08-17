use std::cmp::Ordering;

use num_bigint::BigInt;

use crate::Point;

pub const COORDINATE_EPSILON: f64 = 1.0e-7;

/// How far past a constraint a snapped placement is pushed.
///
/// `nearest` projects onto the boundary of the legal set, so its result sits
/// exactly on a constraint: `contains` accepts it only because the clearance
/// test allows `2r - COORDINATE_EPSILON`. That is fine for one implementation
/// evaluating its own point, and not fine across two. The move server proposed
/// a snapped point 1.1e-6 inside a stone's exclusion disc, which its own search
/// accepted and the browser rejected; the client then re-asked, the stateless
/// server returned the identical point, and the game stalled for 20 requests.
///
/// An order of magnitude above `COORDINATE_EPSILON`, so a snapped point clears
/// the constraint by more than the tolerance either side compares with, while
/// staying far below the resolution any board is played at -- a 1/18 radius
/// spans 0.056, and this moves a point by 0.00001% of that.
pub const SNAP_MARGIN: f64 = 1.0e-6;
pub const EDGE_EPSILON: f64 = 1.0e-10;
pub const COLLINEAR_EPSILON: f64 = 1.0e-11;
pub const COMPARISON_EPSILON: f64 = 1.0e-10;

/// Euclidean length of `(dx, dy)`.
///
/// `f64::hypot` guards against squaring overflowing or underflowing, which
/// costs a call into libm that cannot be inlined or vectorized. Board
/// coordinates are normalized to roughly [0, 1] and the widest intermediate
/// here is a stone separation, so `dx * dx` lands nowhere near either limit and
/// the guard buys nothing. `sqrt` is one instruction and leaves the surrounding
/// loop open to autovectorization.
///
/// Sampled at 27% of self-play CPU time, spread over a dozen call sites in this
/// crate; see the module tests for the agreement bound against `hypot`.
#[inline]
#[must_use]
pub fn length(dx: f64, dy: f64) -> f64 {
    dx.mul_add(dx, dy * dy).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StrictDistanceComparison {
    pub is_strictly_less: bool,
    pub signed_margin: f64,
    pub error_bound: f64,
    pub used_exact_fallback: bool,
}

#[derive(Clone, Copy)]
struct Interval {
    lower: f64,
    upper: f64,
}

fn subtraction_interval(left: f64, right: f64) -> Interval {
    let value = left - right;
    Interval {
        lower: value.next_down(),
        upper: value.next_up(),
    }
}

fn square_interval(value: Interval) -> Interval {
    let lower = if value.lower <= 0.0 && value.upper >= 0.0 {
        0.0
    } else {
        (value.lower * value.lower)
            .min(value.upper * value.upper)
            .next_down()
    };
    let upper = (value.lower * value.lower)
        .max(value.upper * value.upper)
        .next_up();
    Interval { lower, upper }
}

fn squared_distance_interval(first: Point, second: Point) -> Interval {
    let x = square_interval(subtraction_interval(first.x, second.x));
    let y = square_interval(subtraction_interval(first.y, second.y));
    Interval {
        lower: (x.lower + y.lower).next_down(),
        upper: (x.upper + y.upper).next_up(),
    }
}

#[derive(Clone, Debug)]
struct Dyadic {
    coefficient: BigInt,
    exponent: i32,
}

impl Dyadic {
    fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (significand, exponent) = if raw_exponent == 0 {
            (fraction, -1_074)
        } else {
            ((1_u64 << 52) | fraction, raw_exponent - 1_023 - 52)
        };
        let coefficient = if negative {
            -BigInt::from(significand)
        } else {
            BigInt::from(significand)
        };
        Self {
            coefficient,
            exponent,
        }
    }

    fn subtract(self, other: Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        Self {
            coefficient: (self.coefficient << (self.exponent - exponent) as usize)
                - (other.coefficient << (other.exponent - exponent) as usize),
            exponent,
        }
    }

    fn square(self) -> Self {
        Self {
            coefficient: &self.coefficient * &self.coefficient,
            exponent: self.exponent * 2,
        }
    }

    fn add(self, other: Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        Self {
            coefficient: (self.coefficient << (self.exponent - exponent) as usize)
                + (other.coefficient << (other.exponent - exponent) as usize),
            exponent,
        }
    }

    fn compare(&self, other: &Self) -> Ordering {
        let exponent = self.exponent.min(other.exponent);
        (&self.coefficient << (self.exponent - exponent) as usize)
            .cmp(&(&other.coefficient << (other.exponent - exponent) as usize))
    }
}

fn exact_squared_distance(first: Point, second: Point) -> Dyadic {
    let x = Dyadic::from_f64(first.x)
        .subtract(Dyadic::from_f64(second.x))
        .square();
    let y = Dyadic::from_f64(first.y)
        .subtract(Dyadic::from_f64(second.y))
        .square();
    x.add(y)
}

pub(crate) fn strictly_closer(
    origin: Point,
    candidate: Point,
    threshold: Point,
) -> StrictDistanceComparison {
    let candidate_interval = squared_distance_interval(origin, candidate);
    let threshold_interval = squared_distance_interval(origin, threshold);
    let margin_interval = Interval {
        lower: (threshold_interval.lower - candidate_interval.upper).next_down(),
        upper: (threshold_interval.upper - candidate_interval.lower).next_up(),
    };
    let candidate_squared =
        (origin.x - candidate.x).mul_add(origin.x - candidate.x, (origin.y - candidate.y).powi(2));
    let threshold_squared =
        (origin.x - threshold.x).mul_add(origin.x - threshold.x, (origin.y - threshold.y).powi(2));
    let signed_margin = threshold_squared - candidate_squared;
    let error_bound = (signed_margin - margin_interval.lower)
        .abs()
        .max((margin_interval.upper - signed_margin).abs());

    if margin_interval.lower > 0.0 {
        return StrictDistanceComparison {
            is_strictly_less: true,
            signed_margin,
            error_bound,
            used_exact_fallback: false,
        };
    }
    if margin_interval.upper <= 0.0 {
        return StrictDistanceComparison {
            is_strictly_less: false,
            signed_margin,
            error_bound,
            used_exact_fallback: false,
        };
    }

    let exact_candidate = exact_squared_distance(origin, candidate);
    let exact_threshold = exact_squared_distance(origin, threshold);
    StrictDistanceComparison {
        is_strictly_less: exact_candidate.compare(&exact_threshold) == Ordering::Less,
        signed_margin,
        error_bound,
        used_exact_fallback: true,
    }
}

#[cfg(test)]
mod tests {
    use crate::Point;

    use super::{length, strictly_closer};

    /// `length` replaced `hypot` at a dozen call sites, so it has to agree with
    /// it over the range those sites actually see.
    ///
    /// Board coordinates are normalized to [0, 1], so every difference fed to
    /// it lies in [-1, 1] and the widest legitimate result is the diagonal.
    /// Exact equality is not the bar -- `hypot` is correctly rounded and
    /// `sqrt(x*x + y*y)` is not -- but the two must not disagree by more than
    /// rounding, which is far below every epsilon in this module.
    #[test]
    fn length_agrees_with_hypot_over_board_coordinates() {
        let mut worst: f64 = 0.0;
        let steps = 200;
        for i in 0..=steps {
            for j in 0..=steps {
                let dx = -1.0 + 2.0 * f64::from(i) / f64::from(steps);
                let dy = -1.0 + 2.0 * f64::from(j) / f64::from(steps);
                let expected = dx.hypot(dy);
                let actual = length(dx, dy);
                let error = (actual - expected).abs();
                // Relative, because the absolute gap grows with the magnitude.
                let tolerance = 4.0 * f64::EPSILON * expected.max(1.0);
                assert!(
                    error <= tolerance,
                    "length({dx}, {dy}) = {actual}, hypot = {expected}"
                );
                worst = worst.max(error);
            }
        }
        // Well under COLLINEAR_EPSILON, the tightest tolerance any caller uses.
        assert!(worst < super::COLLINEAR_EPSILON, "worst error {worst}");
    }

    /// The degenerate inputs the call sites guard on must survive the swap:
    /// several branch on `< EDGE_EPSILON` before dividing by the result.
    #[test]
    fn length_handles_zero_and_axis_aligned_inputs() {
        assert_eq!(length(0.0, 0.0), 0.0);
        assert_eq!(length(3.0, 0.0), 3.0);
        assert_eq!(length(0.0, -4.0), 4.0);
        assert_eq!(length(3.0, 4.0), 5.0);
    }

    #[test]
    fn robust_distance_comparison_handles_capture_boundary() {
        let origin = Point::new(0.0, 0.0);
        let threshold = Point::new(1.0, 0.0);

        let definite_escape = strictly_closer(origin, Point::new(0.5, 0.0), threshold);
        assert!(definite_escape.is_strictly_less);
        assert!(!definite_escape.used_exact_fallback);

        let definite_capture = strictly_closer(origin, Point::new(1.5, 0.0), threshold);
        assert!(!definite_capture.is_strictly_less);
        assert!(!definite_capture.used_exact_fallback);

        let old_deadband_escape = strictly_closer(origin, Point::new(1.0 - 5.0e-8, 0.0), threshold);
        assert!(old_deadband_escape.is_strictly_less);
        assert!(!old_deadband_escape.used_exact_fallback);

        let old_deadband_capture =
            strictly_closer(origin, Point::new(1.0 + 5.0e-8, 0.0), threshold);
        assert!(!old_deadband_capture.is_strictly_less);
        assert!(!old_deadband_capture.used_exact_fallback);

        let one_ulp_escape =
            strictly_closer(origin, Point::new(1.0_f64.next_down(), 0.0), threshold);
        assert!(one_ulp_escape.is_strictly_less);
        assert!(one_ulp_escape.used_exact_fallback);

        let one_ulp_capture =
            strictly_closer(origin, Point::new(1.0_f64.next_up(), 0.0), threshold);
        assert!(!one_ulp_capture.is_strictly_less);
        assert!(one_ulp_capture.used_exact_fallback);
    }

    #[test]
    fn exact_tie_is_not_strictly_closer() {
        let comparison = strictly_closer(
            Point::new(0.25, 0.25),
            Point::new(0.75, 0.75),
            Point::new(0.75, 0.75),
        );
        assert!(!comparison.is_strictly_less);
        assert!(comparison.used_exact_fallback);
        assert_eq!(comparison.signed_margin, 0.0);
        assert!(comparison.error_bound > 0.0);
    }
}

/// Distance from `point` to the segment `a`-`b`.
///
/// Clamped projection: a degenerate segment collapses to the distance to `a`,
/// which is what the callers below want rather than a NaN.
#[must_use]
pub fn distance_to_segment(point: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let square = dx.mul_add(dx, dy * dy);
    let t = if square <= 0.0 {
        0.0
    } else {
        (((point.x - a.x) * dx + (point.y - a.y) * dy) / square).clamp(0.0, 1.0)
    };
    length(point.x - (a.x + t * dx), point.y - (a.y + t * dy))
}
