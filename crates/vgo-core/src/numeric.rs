use std::cmp::Ordering;

use num_bigint::BigInt;

use crate::Point;

pub const COORDINATE_EPSILON: f64 = 1.0e-7;
pub const EDGE_EPSILON: f64 = 1.0e-10;
pub const COLLINEAR_EPSILON: f64 = 1.0e-11;
pub const COMPARISON_EPSILON: f64 = 1.0e-10;

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

    use super::strictly_closer;

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
