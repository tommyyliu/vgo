//! A saved run of stones, and how a later position differs from it.
//!
//! Every leaf of a search is the root's stones plus the few on its path, less any it
//! captured, so what a thread works out for the root mostly holds for its leaves.
//! This is the comparison that says which stones a leaf has to account for.

use crate::{Point, Stone};

/// Most stones a position can differ by and still be treated as related. Past this
/// it is not really a leaf of the run, and checking each one costs more than it saves.
const MOST_DIFFERING_STONES: usize = 32;

/// How a list of stones relates to a saved run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoneFit {
    /// The run is a prefix of the list.
    Extends,
    /// The list also lost stones from the run, as a capture does.
    Differs,
    /// Too different to be worth comparing.
    Unrelated,
}

/// A saved list of stones, and the stones the last list compared differed by.
///
/// Stones are compared by the exact bits of their coordinates. Colour plays no part,
/// because nothing built on this depends on it.
#[derive(Debug, Default)]
pub struct StoneRun {
    stones: Vec<Point>,
    differing: Vec<Point>,
}

impl StoneRun {
    /// Saves these stones as the run.
    pub fn set(&mut self, stones: &[Stone]) {
        self.stones.clear();
        self.stones
            .extend(stones.iter().map(|stone| Point::new(stone.x, stone.y)));
        self.differing.clear();
    }

    /// Forgets what the last comparison found, as an unrelated list does.
    pub fn clear_differing(&mut self) {
        self.differing.clear();
    }

    /// Compares `stones` with the run, and lists the stones they differ by.
    ///
    /// A stone keeps its place in a position's list until it is captured, so walking
    /// both lists together splits them into the stones they share, the run's stones
    /// the list lost, and the list's stones past the last one shared.
    pub fn fit(&mut self, stones: &[Stone]) -> StoneFit {
        self.differing.clear();
        let mut shared = 0;
        for saved in &self.stones {
            let same = stones.get(shared).is_some_and(|stone| {
                saved.x.to_bits() == stone.x.to_bits() && saved.y.to_bits() == stone.y.to_bits()
            });
            if same {
                shared += 1;
            } else {
                self.differing.push(*saved);
                if self.differing.len() > MOST_DIFFERING_STONES {
                    return StoneFit::Unrelated;
                }
            }
        }
        let lost = self.differing.len();
        self.differing.extend(
            stones[shared..]
                .iter()
                .map(|stone| Point::new(stone.x, stone.y)),
        );
        if self.differing.len() > MOST_DIFFERING_STONES {
            StoneFit::Unrelated
        } else if lost == 0 {
            StoneFit::Extends
        } else {
            StoneFit::Differs
        }
    }

    /// The stones the last list compared differed by: the run's stones it lost, then
    /// its stones past the run. For an [`StoneFit::Extends`] those are all new.
    #[must_use]
    pub fn differing(&self) -> &[Point] {
        &self.differing
    }

    /// Whether every differing stone is at least `reach` from `point`, by squares.
    #[inline]
    #[must_use]
    pub fn differing_beyond(&self, point: Point, reach_square: f64) -> bool {
        self.differing.iter().all(|stone| {
            let (dx, dy) = (stone.x - point.x, stone.y - point.y);
            dx * dx + dy * dy >= reach_square
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Color;

    fn stones(points: &[(f64, f64)]) -> Vec<Stone> {
        points
            .iter()
            .map(|&(x, y)| Stone::new(x, y, Color::Black))
            .collect()
    }

    const A: (f64, f64) = (0.1, 0.2);
    const B: (f64, f64) = (0.3, 0.4);
    const C: (f64, f64) = (0.5, 0.6);
    const D: (f64, f64) = (0.7, 0.8);

    fn differing(run: &StoneRun) -> Vec<(f64, f64)> {
        run.differing().iter().map(|p| (p.x, p.y)).collect()
    }

    #[test]
    fn a_list_past_the_run_extends_it() {
        let mut run = StoneRun::default();
        run.set(&stones(&[A, B]));
        assert_eq!(run.fit(&stones(&[A, B])), StoneFit::Extends);
        assert_eq!(differing(&run), []);
        assert_eq!(run.fit(&stones(&[A, B, C, D])), StoneFit::Extends);
        assert_eq!(differing(&run), [C, D]);
    }

    #[test]
    fn a_capture_lists_the_lost_stones_first() {
        let mut run = StoneRun::default();
        run.set(&stones(&[A, B, C]));
        assert_eq!(run.fit(&stones(&[A, C, D])), StoneFit::Differs);
        assert_eq!(differing(&run), [B, D]);
        assert_eq!(run.fit(&stones(&[A])), StoneFit::Differs);
        assert_eq!(differing(&run), [B, C]);
        assert_eq!(run.fit(&[]), StoneFit::Differs);
        assert_eq!(differing(&run), [A, B, C]);
    }

    #[test]
    fn a_stone_replayed_on_its_own_spot_counts_both_ways() {
        let mut run = StoneRun::default();
        run.set(&stones(&[A, B, C]));
        assert_eq!(run.fit(&stones(&[B, C, A])), StoneFit::Differs);
        assert_eq!(differing(&run), [A, A]);
    }

    #[test]
    fn both_coordinates_count() {
        let mut run = StoneRun::default();
        run.set(&stones(&[A]));
        assert_eq!(run.fit(&stones(&[(A.0, 0.9)])), StoneFit::Differs);
        assert_eq!(run.fit(&stones(&[(0.9, A.1)])), StoneFit::Differs);
    }

    #[test]
    fn too_many_differences_are_unrelated() {
        let many: Vec<(f64, f64)> = (0..40).map(|at| (f64::from(at) / 40.0, 0.5)).collect();
        let mut run = StoneRun::default();
        run.set(&[]);
        assert_eq!(run.fit(&stones(&many[..32])), StoneFit::Extends);
        assert_eq!(run.fit(&stones(&many[..33])), StoneFit::Unrelated);
        run.set(&stones(&many));
        assert_eq!(run.fit(&[]), StoneFit::Unrelated);
    }

    #[test]
    fn beyond_is_measured_from_every_differing_stone() {
        let mut run = StoneRun::default();
        run.set(&[]);
        assert_eq!(run.fit(&stones(&[A, C])), StoneFit::Extends);
        let point = Point::new(0.1, 0.2);
        assert!(run.differing_beyond(point, 0.0));
        assert!(!run.differing_beyond(point, 0.01));
        let far = Point::new(0.9, 0.2);
        // 0.32 from C and 0.64 from A, so the nearer stone is the one that counts.
        assert!(run.differing_beyond(far, 0.3));
        assert!(!run.differing_beyond(far, 0.4));
    }
}
