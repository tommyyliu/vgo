use crate::{Point, Position, numeric};

use super::clear_of_stones;

/// Below this many stones the quadratic loops in [`vertices`] are cheaper than
/// the grid that would prune them, and a grid over a nearly empty board is all
/// build cost and no saving.
pub(super) const BUCKET_MINIMUM_STONES: usize = 32;

/// A uniform bucket grid over the stones, one cell per stone diameter.
///
/// Both loops in [`vertices`] are quadratic in the stone count and neither needs
/// to be. A pair of stones only produces a vertex when they are within
/// `2 * diameter`, and a candidate point is rejected by the first stone within
/// `2 * radius` of it -- both local questions asked against every stone on the
/// board. Measured at 240 stones the clearance test alone walked 211,924 stones
/// across 1,781 calls, 119 apiece, where nine would have done; the pair scan was
/// 28,823 for the 873 pairs that survived it.
pub(super) struct StoneBuckets {
    cell: f64,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    indices: Vec<u32>,
}

impl StoneBuckets {
    pub(super) fn build(position: &Position) -> Self {
        let stones = position.stones();
        let cell = 2.0 * position.radius();
        let columns = ((1.0 / cell).ceil() as usize).clamp(1, 512);
        let rows = columns;
        let locate = |x: f64, y: f64| -> usize {
            let column = ((x / cell) as isize).clamp(0, columns as isize - 1) as usize;
            let row = ((y / cell) as isize).clamp(0, rows as isize - 1) as usize;
            row * columns + column
        };
        let mut starts = vec![0u32; columns * rows + 1];
        for stone in stones {
            starts[locate(stone.x, stone.y) + 1] += 1;
        }
        for index in 0..columns * rows {
            starts[index + 1] += starts[index];
        }
        let mut cursor = starts.clone();
        let mut indices = vec![0u32; stones.len()];
        for (index, stone) in stones.iter().enumerate() {
            let bucket = locate(stone.x, stone.y);
            indices[cursor[bucket] as usize] = index as u32;
            cursor[bucket] += 1;
        }
        Self {
            cell,
            columns,
            rows,
            starts,
            indices,
        }
    }

    pub(super) fn cell_of(&self, x: f64, y: f64) -> (usize, usize) {
        let column = ((x / self.cell) as isize).clamp(0, self.columns as isize - 1) as usize;
        let row = ((y / self.cell) as isize).clamp(0, self.rows as isize - 1) as usize;
        (column, row)
    }

    /// Every stone in `columns` of `row`. Consecutive cells of a row are
    /// consecutive in `indices`, so a range of them is one slice, in cell order.
    #[inline]
    fn row_span(&self, row: usize, columns: std::ops::RangeInclusive<usize>) -> &[u32] {
        let base = row * self.columns;
        let first = self.starts[base + columns.start()] as usize;
        let last = self.starts[base + columns.end() + 1] as usize;
        &self.indices[first..last]
    }

    /// Visit a conservative bounding square of the radial candidate reach.
    /// The caller still checks Euclidean distance, so extra buckets affect
    /// only cost. Coordinate slack keeps a rounded boundary from excluding a
    /// stone accepted by the existing floating-point radial test.
    pub(super) fn visit_near(
        &self,
        position: &Position,
        point: Point,
        reach: f64,
        visit: &mut impl FnMut(&crate::Stone) -> bool,
    ) -> bool {
        let margin = reach + numeric::COORDINATE_EPSILON;
        let (first_column, first_row) = self.cell_of(point.x - margin, point.y - margin);
        let (last_column, last_row) = self.cell_of(point.x + margin, point.y + margin);
        for row in first_row..=last_row {
            for &index in self.row_span(row, first_column..=last_column) {
                if visit(&position.stones()[index as usize]) {
                    return true;
                }
            }
        }
        false
    }

    /// The index of every stone within `reach` of `point`, plus possibly others.
    pub(super) fn for_each_near(&self, point: Point, reach: f64, visit: &mut impl FnMut(usize)) {
        let margin = reach + numeric::COORDINATE_EPSILON;
        let (first_column, first_row) = self.cell_of(point.x - margin, point.y - margin);
        let (last_column, last_row) = self.cell_of(point.x + margin, point.y + margin);
        for row in first_row..=last_row {
            for &index in self.row_span(row, first_column..=last_column) {
                visit(index as usize);
            }
        }
    }

    /// Every stone after `first` that could pair with the point, in index order.
    ///
    /// Three cells of reach, not two. The pair test admits a separation up to
    /// `2 * diameter + COORDINATE_EPSILON`, which is two cells *plus* whatever
    /// slack the point's offset within its own cell contributes, and a stone
    /// sitting a hair past a boundary lands in the third. The caller still
    /// applies the exact separation test, so the extra ring only costs
    /// distances, never correctness.
    pub(super) fn partners_into(&self, x: f64, y: f64, first: usize, out: &mut Vec<usize>) {
        let (column, row) = self.cell_of(x, y);
        let columns = column.saturating_sub(3)..=(column + 3).min(self.columns - 1);
        for r in row.saturating_sub(3)..=(row + 3).min(self.rows - 1) {
            for &index in self.row_span(r, columns.clone()) {
                if index as usize > first {
                    out.push(index as usize);
                }
            }
        }
        out.sort_unstable();
    }
}

/// [`clear_of_stones`], restricted to the stones that can possibly fail it.
///
/// A stone that fails is closer than `2 * radius` and the cells are that wide,
/// so it cannot be more than one cell away in either axis. With no buckets --
/// too few stones to have built any -- this is the original scan.
pub(super) fn clear_of_stones_pruned(
    position: &Position,
    buckets: Option<&StoneBuckets>,
    x: f64,
    y: f64,
    skip_a: Option<usize>,
    skip_b: Option<usize>,
) -> bool {
    let Some(buckets) = buckets else {
        return clear_of_stones(position, x, y, skip_a, skip_b);
    };
    let stones = position.stones();
    let minimum = 2.0 * position.radius() - numeric::COORDINATE_EPSILON;
    let minimum_squared = minimum * minimum;
    let (column, row) = buckets.cell_of(x, y);
    let columns = column.saturating_sub(1)..=(column + 1).min(buckets.columns - 1);
    for r in row.saturating_sub(1)..=(row + 1).min(buckets.rows - 1) {
        for &index in buckets.row_span(r, columns.clone()) {
            let index = index as usize;
            if Some(index) == skip_a || Some(index) == skip_b {
                continue;
            }
            let stone = &stones[index];
            let dx = x - stone.x;
            let dy = y - stone.y;
            if dx.mul_add(dx, dy * dy) < minimum_squared {
                return false;
            }
        }
    }
    true
}
