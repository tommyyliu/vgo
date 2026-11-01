//! The settled mask as a comparison of two distance fields.
//!
//! ```text
//!     settled(x)  <=>  D_S(x) <= D_L(x)
//! ```
//!
//! `D_S` is the distance to the nearest stone and `D_L` the distance to the legal
//! set (A15 in `spec/LEMMAS.md`). `D_S` is exact, from continuous coordinates.
//! `D_L` comes from a distance transform of the legal set sampled on the grid, which
//! bounds it from both sides. Those bounds decide most pixels, and an exact test
//! decides the band between them. The cost is linear in pixels, plus the band.
//!
//! This is not proved exact on every position: see [O1].

use std::sync::Arc;

use vgo_core::{
    COORDINATE_EPSILON, LegalSetIndex, Point, Position, none_closer_than_except_vertices,
};

use crate::RasterConfig;
use crate::edt::{EdtScratch, squared_distance_transform_into};

/// Pixel columns processed together by the nearest-stone search. The same
/// chunking shape is used by the packed writer: it amortizes the grid-ring
/// walk without making the chunk's bound so wide that nearby stones are kept.
const NEAREST_CHUNK: usize = 128;

/// Below this many stones, the flat vectorizable scan is faster than building
/// and walking a spatial index.
const NEAREST_SEARCH_MINIMUM_STONES: usize = 96;

/// Buffers retained by a caller rendering many positions at one raster size.
#[derive(Debug, Default)]
pub(crate) struct SettledScratch {
    pub(crate) edt: EdtScratch,
    legal: Vec<bool>,
    fine_column_xs: Vec<f64>,
    fine_row_ys: Vec<f64>,
    column_xs: Vec<f64>,
    nearest_squares: Vec<f64>,
}

/// Stones bucketed into a uniform grid for nearest-distance queries.
///
/// A settled mask needs only the nearest stone, but the old row sweep still
/// visits every stone for every output pixel. At late-game stone counts this
/// is the remaining O(pixels * stones) part of the settled path. Querying
/// chunks of pixels outward through this grid lets the nearest-distance bound
/// discard distant buckets while retaining exact point-to-stone distances.
struct StoneGrid {
    cell: f64,
    width: usize,
    height: usize,
    starts: Vec<u32>,
    indices: Vec<u32>,
}

impl StoneGrid {
    fn build(position: &Position, cell: f64) -> Self {
        let stones = position.stones();
        let width = (1.0 / cell).ceil() as usize + 1;
        let height = width;
        let cells = width * height;
        let index_of = |x: f64, y: f64| {
            let cx = ((x / cell) as usize).min(width - 1);
            let cy = ((y / cell) as usize).min(height - 1);
            cy * width + cx
        };
        let mut counts = vec![0u32; cells + 1];
        for stone in stones {
            counts[index_of(stone.x, stone.y) + 1] += 1;
        }
        for index in 0..cells {
            counts[index + 1] += counts[index];
        }
        let mut cursor = counts.clone();
        let mut indices = vec![0u32; stones.len()];
        for (stone, value) in stones.iter().enumerate() {
            let cell_index = index_of(value.x, value.y);
            indices[cursor[cell_index] as usize] = stone as u32;
            cursor[cell_index] += 1;
        }
        Self {
            cell,
            width,
            height,
            starts: counts,
            indices,
        }
    }

    #[inline]
    fn bucket(&self, cx: usize, cy: usize) -> &[u32] {
        let cell = cy * self.width + cx;
        let start = self.starts[cell] as usize;
        let end = self.starts[cell + 1] as usize;
        &self.indices[start..end]
    }
}

/// Exact nearest squared distances for one output row, using a spatial search
/// once the stone count makes the flat scan too expensive.
fn nearest_row_chunked(
    position: &Position,
    grid: &StoneGrid,
    y: f64,
    xs: &[f64],
    nearest_squares: &mut [f64],
) {
    nearest_squares.fill(f64::INFINITY);
    let stones = position.stones();
    let cell = grid.cell;
    let row_cell = ((y / cell) as usize).min(grid.height - 1);
    let longest = grid.width.max(grid.height);

    for start in (0..xs.len()).step_by(NEAREST_CHUNK) {
        let end = (start + NEAREST_CHUNK).min(xs.len());
        let low_x = xs[start];
        let high_x = xs[end - 1];
        let cx0 = ((low_x / cell) as usize).min(grid.width - 1);
        let cx1 = ((high_x / cell) as usize).min(grid.width - 1);
        let mut bound = f64::INFINITY;

        for ring in 0..=longest {
            if ring > 0 {
                let reach = (ring - 1) as f64 * cell;
                if reach * reach >= bound {
                    break;
                }
            }
            for cy in row_cell.saturating_sub(ring)..=(row_cell + ring).min(grid.height - 1) {
                let edge_y = cy.abs_diff(row_cell) == ring;
                let from = cx0.saturating_sub(ring);
                let to = (cx1 + ring).min(grid.width - 1);
                for cx in from..=to {
                    let edge_x = cx + ring == cx0 || cx == cx1 + ring;
                    if !(edge_y || edge_x) {
                        continue;
                    }
                    for &index in grid.bucket(cx, cy) {
                        let stone = stones[index as usize];
                        let dy = y - stone.y;
                        let dy_squared = dy * dy;
                        let gap = if stone.x < low_x {
                            low_x - stone.x
                        } else if stone.x > high_x {
                            stone.x - high_x
                        } else {
                            0.0
                        };
                        if gap.mul_add(gap, dy_squared) >= bound {
                            continue;
                        }
                        for column in start..end {
                            let dx = xs[column] - stone.x;
                            let square = dx.mul_add(dx, dy_squared);
                            if square < nearest_squares[column] {
                                nearest_squares[column] = square;
                            }
                        }
                    }
                }
            }
            bound = nearest_squares[start..end]
                .iter()
                .copied()
                .fold(0.0, f64::max);
        }
    }
}

/// The legal set sampled at cell centres, plus the cell holding each of
/// `vertices`.
///
/// Sampling alone misses any legal component that contains no cell centre, and
/// that is not rare: two stones a hair over `2r` apart leave a sliver of legal
/// board between them, which is exactly the contestable gap a boundary move
/// lives in. Missing one makes the distance to the legal set an overestimate,
/// so the settled test calls the region beside it settled. Measured on real
/// 1/38 positions, that was 1.2% of every 256 raster's pixels, all of them
/// false "settled". Every bounded component has a vertex, so marking the vertex
/// cells means every component is seen. A marked centre is within half a cell
/// diagonal of a legal point rather than on one, which the caller's
/// "certainly unsettled" test allows for.
fn sampled_legal_set_into(
    position: &Position,
    fine_width: usize,
    fine_height: usize,
    vertices: &[Point],
    scratch: &mut SettledScratch,
) {
    let stones = position.stones();
    //
    // Testing every pixel against every stone is O(pixels · n) and dominates at
    // any useful oversample: 512² × 28 is 7.3M predicate calls. But a stone
    // only forbids a disc of radius 2r around itself, so scattering that disc
    // touches O(n · r²·pixels) cells instead -- about 370k for the same case,
    // and independent of how many stones are far away. Start from the inset
    // rectangle and clear each stone's exclusion disc.
    let radius = position.radius();
    let exclusion = 2.0 * radius - COORDINATE_EPSILON;
    let exclusion_squared = exclusion * exclusion;
    scratch.legal.resize(fine_width * fine_height, false);
    scratch.legal.fill(false);
    scratch.fine_column_xs.resize(fine_width, 0.0);
    for (column, x) in scratch.fine_column_xs.iter_mut().enumerate() {
        *x = (column as f64 + 0.5) / fine_width as f64;
    }
    scratch.fine_row_ys.resize(fine_height, 0.0);
    for (row, y) in scratch.fine_row_ys.iter_mut().enumerate() {
        *y = (row as f64 + 0.5) / fine_height as f64;
    }
    let inset_low = radius - COORDINATE_EPSILON;
    let inset_high = 1.0 - radius + COORDINATE_EPSILON;
    let first_column = ((inset_low * fine_width as f64 - 0.5).ceil().max(0.0)) as usize;
    let last_column = ((inset_high * fine_width as f64 - 0.5)
        .floor()
        .min(fine_width.saturating_sub(1) as f64)) as usize;
    for row in 0..fine_height {
        let y = scratch.fine_row_ys[row];
        if y < inset_low || y > inset_high {
            continue;
        }
        if first_column <= last_column {
            let base = row * fine_width;
            scratch.legal[base + first_column..=base + last_column].fill(true);
        }
    }
    for stone in stones {
        // Bounding box of the exclusion disc, clipped to the grid.
        let low_row =
            (((stone.y - exclusion) * fine_height as f64 - 0.5).floor()).max(0.0) as usize;
        let high_row = ((((stone.y + exclusion) * fine_height as f64 - 0.5).ceil()) as usize)
            .min(fine_height - 1);
        let low_column =
            (((stone.x - exclusion) * fine_width as f64 - 0.5).floor()).max(0.0) as usize;
        let high_column = ((((stone.x + exclusion) * fine_width as f64 - 0.5).ceil()) as usize)
            .min(fine_width - 1);
        for row in low_row..=high_row {
            let y = scratch.fine_row_ys[row];
            let dy = y - stone.y;
            let dy_squared = dy * dy;
            if dy_squared > exclusion_squared {
                continue;
            }
            let base = row * fine_width;
            for (legal, &x) in scratch.legal[base + low_column..=base + high_column]
                .iter_mut()
                .zip(&scratch.fine_column_xs[low_column..=high_column])
            {
                let dx = x - stone.x;
                // A store either way, so this vectorizes.
                *legal &= dx.mul_add(dx, dy_squared) >= exclusion_squared;
            }
        }
    }
    for vertex in vertices {
        let column = ((vertex.x * fine_width as f64) as usize).min(fine_width - 1);
        let row = ((vertex.y * fine_height as f64) as usize).min(fine_height - 1);
        scratch.legal[row * fine_width + column] = true;
    }
}

/// The settled mask, using distance-transform bounds to skip most of the work.
///
/// The sampled `D_L` is not merely approximate, it is approximate in a *known
/// direction*: it measures the distance to the nearest legal pixel centre, and
/// every such centre is a point of the legal set, so
///
/// ```text
///     D_L_true  <=  D_L_grid  <=  D_L_true + e
/// ```
///
/// where `e` is half a grid diagonal. That two-sided bound decides almost every
/// pixel outright:
///
///   * `D_S <= D_L_grid - e`  implies `D_S <= D_L_true`  — settled
///   * `D_S >  D_L_grid`      implies `D_S >  D_L_true`  — not settled
///
/// Only the band between them is genuinely undecided, and those pixels get the
/// exact continuous test, which is what made the direct formulation 42x too slow
/// to use everywhere. The band follows the settled boundary, which around an open
/// stone is its own disk, so it is not small: 1,600 of 16,384 pixels on a 1/38
/// board, before [E9] answers the ones inside a disk and leaves 950.
///
/// **The upper bound is not guaranteed.** It assumes every point of the legal
/// set has a sampled cell centre within half a diagonal, which fails where the
/// legal set is a sliver thinner than the grid — then `D_L_grid` overshoots by
/// more than the slack and a pixel can be called settled when it is not. The
/// lower direction (`D_S > D_L_grid` implies not settled) is always sound,
/// since every legal cell centre really is a point of the set.
///
/// In practice that costs at most one pixel of 16384 at the densest fixture
/// tested. Oversampling to 3 removed it entirely on every fixture.
///
/// Returns the mask and how many pixels needed the exact test, so callers can
/// see whether the band is staying small.
#[must_use]
pub fn settled_mask_by_bounded_distance(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
) -> (Vec<bool>, usize) {
    let mut scratch = SettledScratch::default();
    let mut mask = Vec::new();
    let tests =
        settled_by_bounded_distance_into(position, config, oversample, &mut scratch, &mut mask);
    (mask, tests)
}

/// The row-independent half of the `settled` test, so the packed writer can supply
/// the nearest-stone minima its own sweep already computed.
///
/// Both callers must compute squared distances by `mul_add`. `settled` is a bit
/// decided by comparing this distance against a sampled one, so unlike the ridge,
/// which quantizes to f16, one ulp here is a different pixel.
///
/// Not proved exact on every position: see [O1].
pub(crate) struct SettledRows {
    scale: usize,
    fine_width: usize,
    fine_height: usize,
    half_diagonal: f64,
    spacing_squared: f64,
    slack: f64,
    index: Arc<LegalSetIndex>,
    pub(crate) exact_tests: usize,
    undecided: Vec<bool>,
    /// Squared distance from a stone within which a pixel is certainly settled.
    disk_square: f64,
}

/// Column of `field` that `column` of the output reads.
#[inline]
const fn fine_index(index: usize, scale: usize, fine_size: usize) -> usize {
    let fine = index * scale + scale / 2;
    if fine < fine_size {
        fine
    } else {
        fine_size - 1
    }
}

pub(crate) fn prepare_settled_rows(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
    scratch: &mut SettledScratch,
) -> SettledRows {
    let scale = oversample.max(1) | 1;
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);
    let index = Arc::clone(position.legal_index());
    sampled_legal_set_into(position, fine_width, fine_height, index.vertices(), scratch);
    squared_distance_transform_into(&scratch.legal, fine_width, fine_height, &mut scratch.edt);
    let spacing = 1.0 / fine_width as f64;
    // How far the sampled distance can overstate the true one. Half a cell
    // diagonal is the tempting answer and it is wrong: it assumes every point of
    // the legal set has a cell centre within that distance, which fails all along
    // the set's boundary. A full diagonal covers that. Nothing covers a sliver
    // narrower than a cell, which is why this is not proved exact: see [O1].
    let slack = spacing * std::f64::consts::SQRT_2;
    SettledRows {
        scale,
        fine_width,
        fine_height,
        half_diagonal: 0.5 * slack,
        spacing_squared: spacing * spacing,
        slack,
        index,
        exact_tests: 0,
        undecided: Vec::new(),
        disk_square: (position.radius() - COORDINATE_EPSILON).powi(2),
    }
}

impl SettledRows {
    /// Classify one row. `nearest_squares` is the caller's per-column minimum over
    /// every stone, `field` the squared sampled distance transform, and `out` the
    /// row's slice of the settled mask.
    ///
    /// The cases are checked in squared-distance space where possible. Rounding
    /// here can only move a pixel between a cheap case and the undecided band,
    /// never from settled straight to unsettled, because `slack > 0` keeps the
    /// cases disjoint and ordered. A pixel in the band gets the exact test, which
    /// agrees with either cheap case, so only `exact_tests` can change.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn classify_row(
        &mut self,
        position: &Position,
        row: usize,
        y: f64,
        xs: &[f64],
        nearest_squares: &[f64],
        field: &[f64],
        out: &mut [bool],
    ) {
        let scale = self.scale;
        let base = fine_index(row, scale, self.fine_height) * self.fine_width;
        let width = xs.len();
        self.undecided.resize(width, false);
        // The two cheap cases for the whole row first, with no branch and no call
        // in the loop, so its roots and comparisons vectorize. Only what they leave
        // undecided costs anything per pixel.
        let (spacing_squared, slack, half_diagonal, disk_square) = (
            self.spacing_squared,
            self.slack,
            self.half_diagonal,
            self.disk_square,
        );
        let fine_row = &field[base..base + self.fine_width];
        for (column, ((&nearest_square, settled), undecided)) in nearest_squares
            .iter()
            .zip(out.iter_mut())
            .zip(self.undecided.iter_mut())
            .enumerate()
        {
            let sampled = fine_row[if scale == 1 {
                column
            } else {
                fine_index(column, scale, self.fine_width)
            }];
            let sampled = (sampled * spacing_squared).sqrt();
            let lower = sampled - slack;
            let upper = sampled + half_diagonal;
            // Inside a stone's own disk nothing legal can be closer: [E9].
            let inside =
                nearest_square <= disk_square || (lower > 0.0 && nearest_square <= lower * lower);
            let outside = nearest_square > upper * upper;
            *settled = inside;
            *undecided = !inside & !outside;
        }
        for (column, undecided) in self.undecided.iter().enumerate() {
            if !*undecided {
                continue;
            }
            let point = Point::new(xs[column], y);
            let distance = nearest_squares[column].sqrt();
            // A vertex is an exact legal point. If one is strictly closer than the
            // pixel's nearest stone the pixel is not settled, with no further
            // geometry. The undecided band is mostly pixels beside a thin legal
            // gap, and a gap's own vertices are right there.
            out[column] = !self.index.any_vertex_closer(point, distance) && {
                self.exact_tests += 1;
                none_closer_than_except_vertices(position, point, distance, &self.index)
            };
        }
    }
}

/// The reference path: computes its own nearest-stone minima, then classifies each
/// row the same way the packed writer does. Returns how many exact tests ran.
fn settled_by_bounded_distance_into(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
    scratch: &mut SettledScratch,
    mask: &mut Vec<bool>,
) -> usize {
    let stones = position.stones();
    mask.clear();
    mask.resize(config.pixels(), false);
    // Nothing owns anything on a stoneless board.
    if stones.is_empty() {
        return 0;
    }
    let mut rows = prepare_settled_rows(position, config, oversample, scratch);
    let width = config.width;
    scratch.column_xs.clear();
    scratch
        .column_xs
        .extend((0..width).map(|column| (column as f64 + 0.5) / width as f64));
    scratch.nearest_squares.resize(width, f64::INFINITY);
    let nearest_grid = (stones.len() >= NEAREST_SEARCH_MINIMUM_STONES).then(|| {
        let cell = (2.0 * position.radius()).max(1.0 / (stones.len() as f64).sqrt());
        StoneGrid::build(position, cell)
    });
    for (row, out) in mask.chunks_exact_mut(width).enumerate() {
        let y = (row as f64 + 0.5) / config.height as f64;
        if let Some(grid) = nearest_grid.as_ref() {
            nearest_row_chunked(
                position,
                grid,
                y,
                &scratch.column_xs,
                &mut scratch.nearest_squares,
            );
        } else {
            scratch.nearest_squares.fill(f64::INFINITY);
            for stone in stones {
                let dy = y - stone.y;
                let dy_square = dy * dy;
                // `mul_add`, matching `nearest_row_chunked`. The two paths must
                // compute the same distance, or the dispatch threshold silently
                // re-renders positions near it.
                for (nearest, &x) in scratch.nearest_squares.iter_mut().zip(&scratch.column_xs) {
                    let dx = x - stone.x;
                    *nearest = dx.mul_add(dx, dy_square).min(*nearest);
                }
            }
        }
        rows.classify_row(
            position,
            row,
            y,
            &scratch.column_xs,
            &scratch.nearest_squares,
            scratch.edt.field(),
            out,
        );
    }
    rows.exact_tests
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use vgo_core::{Color, Stone, distance_to_legal_set, legal_set_vertices};

    use super::*;
    use crate::{RasterKind, SETTLED_OVERSAMPLE};

    fn fixture(count: usize, radius: f64) -> Position {
        let spacing = 2.0 * radius * 1.08;
        let per_row = ((0.86_f64 / spacing).floor() as usize).max(1);
        let mut stones = Vec::new();
        for index in 0..count {
            let (row, column) = (index / per_row, index % per_row);
            let x = 0.07 + (column as f64 + 0.5) * spacing;
            let y = 0.07 + (row as f64 + 0.5) * spacing;
            if x > 0.96 || y > 0.96 {
                break;
            }
            let colour = if index % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            stones.push(Stone::new(x, y, colour));
        }
        Position::new(radius, stones, Color::Black).with_komi(0.104)
    }

    /// Random sequential packing, the case the regular lattice fixtures miss.
    ///
    /// Stones placed at random leave many near-tangent pairs, and each one is a
    /// sliver of legal board that no cell centre falls in. Before the legal-set
    /// vertices were marked on the sampled grid, every such sliver made the
    /// region beside it read as settled: 1.6% of a 256 raster here, and 1.2% on
    /// real 1/38 game positions -- all false "settled", all within two pixels of
    /// a vertex.
    #[test]
    fn random_packings_agree_with_the_definition() {
        fn packing(radius: f64, target: usize, seed: u64) -> Position {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1_u64 << 53) as f64
            };
            let mut stones: Vec<Stone> = Vec::new();
            for _ in 0..200_000 {
                if stones.len() == target {
                    break;
                }
                let x = radius + next() * (1.0 - 2.0 * radius);
                let y = radius + next() * (1.0 - 2.0 * radius);
                let clear = stones.iter().all(|s| {
                    ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt() > 2.0 * radius * 1.0001
                });
                if clear {
                    let colour = if stones.len().is_multiple_of(2) {
                        Color::Black
                    } else {
                        Color::White
                    };
                    stones.push(Stone::new(x, y, colour));
                }
            }
            Position::new(radius, stones, Color::Black)
        }
        for (size, radius, target) in [
            (128usize, 1.0 / 38.0, 250usize),
            (128, 1.0 / 38.0, 400),
            (96, 1.0 / 18.0, 90),
        ] {
            let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
            let position = packing(radius, target, target as u64);
            let vertices = legal_set_vertices(&position);
            let (mask, _) = settled_mask_by_bounded_distance(&position, config, SETTLED_OVERSAMPLE);
            let mut wrong = 0usize;
            for pixel in 0..config.pixels() {
                let x = ((pixel % size) as f64 + 0.5) / size as f64;
                let y = ((pixel / size) as f64 + 0.5) / size as f64;
                let nearest = position
                    .stones()
                    .iter()
                    .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let truth =
                    nearest <= distance_to_legal_set(&position, Point::new(x, y), Some(&vertices));
                if truth != mask[pixel] {
                    wrong += 1;
                }
            }
            assert_eq!(
                wrong, 0,
                "{size}px, {target} stones: {wrong} pixels disagree"
            );
        }
    }

    /// A coarse raster on a big board -- 128 at r = 1/38, 3.4 cells per radius --
    /// samples the legal set at the output resolution. It must still agree with
    /// the definition, including on the lattice whose legal gaps are about one
    /// cell wide.
    #[test]
    fn coarse_raster_agrees_with_the_definition() {
        let radius = 1.0 / 38.0;
        let config = RasterConfig::square_of(128, RasterKind::CompactRadius);
        for count in [28usize, 120, 240] {
            let position = fixture(count, radius);
            assert!(position.validate().is_playable());
            let vertices = legal_set_vertices(&position);
            let (mask, _) = settled_mask_by_bounded_distance(&position, config, SETTLED_OVERSAMPLE);
            let mut wrong = 0usize;
            for pixel in 0..config.pixels() {
                let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
                let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
                let nearest = position
                    .stones()
                    .iter()
                    .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let truth =
                    nearest <= distance_to_legal_set(&position, Point::new(x, y), Some(&vertices));
                if truth != mask[pixel] {
                    wrong += 1;
                }
            }
            assert_eq!(
                wrong, 0,
                "{count} stones: {wrong} pixels disagree with the definition"
            );
        }
    }

    /// [E9] on its own terms: every pixel the disk rule answers must also be
    /// settled by the exact test it replaces, on boards with near-tangent stones.
    #[test]
    fn a_pixel_inside_a_stone_disk_is_settled_by_the_exact_test() {
        let radius = 1.0 / 38.0;
        for count in [28usize, 120, 240] {
            let position = fixture(count, radius);
            let disk = (radius - COORDINATE_EPSILON).powi(2);
            let mut tested = 0;
            for row in 0..128 {
                for column in 0..128 {
                    let point =
                        Point::new((column as f64 + 0.5) / 128.0, (row as f64 + 0.5) / 128.0);
                    let nearest = position
                        .stones()
                        .iter()
                        .map(|s| {
                            let (dx, dy) = (point.x - s.x, point.y - s.y);
                            dx.mul_add(dx, dy * dy)
                        })
                        .fold(f64::INFINITY, f64::min);
                    if nearest <= disk {
                        tested += 1;
                        assert!(vgo_core::no_legal_point_closer_than_indexed(
                            &position,
                            point,
                            nearest.sqrt(),
                            position.legal_index(),
                        ));
                    }
                }
            }
            assert!(tested > 0);
        }
    }

    /// The bounded form must agree with the definition, not merely with the
    /// other implementation -- which walks a contour at 1/128 and is itself
    /// wrong on a pixel or two at high stone counts.
    #[test]
    fn bounded_distance_agrees_with_the_definition() {
        let radius = 0.055_714_285_714_285_716;
        let config = RasterConfig::square_of(128, RasterKind::Compact);
        for count in [1usize, 2, 4, 8, 14, 19, 28, 52] {
            let position = fixture(count, radius);
            if !position.validate().is_playable() || position.stones().is_empty() {
                continue;
            }
            let vertices = legal_set_vertices(&position);
            // Oversample 1 is what the raster uses: it is the fastest setting
            // and, with the slack corrected to a full cell diagonal, was exact on
            // every fixture here. The function is still not exact in general --
            // see its doc comment on slivers.
            let (mask, exact_tests) = settled_mask_by_bounded_distance(&position, config, 1);

            let mut wrong = 0usize;
            for pixel in 0..config.pixels() {
                let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
                let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
                let nearest = position
                    .stones()
                    .iter()
                    .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let truth =
                    nearest <= distance_to_legal_set(&position, Point::new(x, y), Some(&vertices));
                if truth != mask[pixel] {
                    wrong += 1;
                }
            }
            assert_eq!(
                wrong, 0,
                "{count} stones: {wrong} pixels disagree with the definition"
            );
            // The bound has to be doing the work; if the fallback ran
            // everywhere this would pass while being slower than the exact path.
            assert!(
                exact_tests * 20 < config.pixels(),
                "{count} stones: {exact_tests} exact tests is too many to be a fallback"
            );
        }
    }

    #[test]
    fn chunked_nearest_matches_the_flat_scan() {
        let radius = 1.0 / 38.0;
        let step = 2.2 * radius;
        let mut stones = Vec::new();
        'outer: for row in 0..24 {
            for column in 0..24 {
                let x = 0.04 + step * f64::from(column);
                let y = 0.04 + step * f64::from(row);
                if x > 0.97 || y > 0.97 {
                    continue;
                }
                stones.push(Stone::new(
                    x,
                    y,
                    if stones.len() % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    },
                ));
                if stones.len() == 240 {
                    break 'outer;
                }
            }
        }
        let position = Position::new(radius, stones, Color::Black);
        let config = RasterConfig::square_of(128, RasterKind::Compact);
        let cell = (2.0 * radius).max(1.0 / (position.stones().len() as f64).sqrt());
        let grid = StoneGrid::build(&position, cell);
        let xs: Vec<f64> = (0..config.width)
            .map(|column| (column as f64 + 0.5) / config.width as f64)
            .collect();
        let mut chunked = vec![f64::INFINITY; config.width];
        for row in 0..config.height {
            let y = (row as f64 + 0.5) / config.height as f64;
            nearest_row_chunked(&position, &grid, y, &xs, &mut chunked);
            for (column, &x) in xs.iter().enumerate() {
                let mut expected = f64::INFINITY;
                for stone in position.stones() {
                    let dx = x - stone.x;
                    let dy = y - stone.y;
                    let square = dx.mul_add(dx, dy * dy);
                    if square < expected {
                        expected = square;
                    }
                }
                assert_eq!(chunked[column], expected, "row {row}, column {column}");
            }
        }
    }

    #[test]
    fn an_empty_board_settles_nothing() {
        let config = RasterConfig::square_of(32, RasterKind::Compact);
        let position = Position::new(0.05, Vec::new(), Color::Black);
        let (mask, _) = settled_mask_by_bounded_distance(&position, config, 1);
        assert!(mask.iter().all(|settled| !settled));
    }
}

#[cfg(test)]
mod dispatch_parity {
    use super::*;
    use vgo_core::{Color, Position, Stone};

    /// A checksum of the settled mask, for checking that the nearest-stone
    /// dispatch does not change what is rendered.
    ///
    /// The threshold is a tuning constant, and a tuning constant that alters a
    /// network input is a trap: someone moves it for speed and silently
    /// re-renders every position near the boundary. Run this, move
    /// `NEAREST_SEARCH_MINIMUM_STONES` to 0 and to a huge value, and diff the
    /// output. It was identical across every configuration when last checked,
    /// with both paths computing distances by `mul_add`.
    ///
    /// Printed rather than asserted because the comparison is between two
    /// builds, which a single test cannot do.
    #[test]
    #[ignore]
    fn settled_checksum_across_stone_counts() {
        for units in [18usize, 38] {
            let radius = 1.0 / units as f64;
            let step = 2.2 * radius;
            for count in [40usize, 55, 60, 65, 80, 120, 240] {
                let mut stones = Vec::new();
                let mut placed = 0usize;
                'outer: for row in 0..40 {
                    for column in 0..40 {
                        let x = 0.04 + step * f64::from(column);
                        let y = 0.04 + step * f64::from(row);
                        if x > 0.97 || y > 0.97 {
                            continue;
                        }
                        stones.push(Stone::new(
                            x,
                            y,
                            if placed.is_multiple_of(2) {
                                Color::Black
                            } else {
                                Color::White
                            },
                        ));
                        placed += 1;
                        if placed == count {
                            break 'outer;
                        }
                    }
                }
                if stones.len() < count {
                    continue;
                }
                let position = Position::new(radius, stones, Color::White).with_komi(0.104);
                let config = RasterConfig::square_of(256, crate::RasterKind::CompactRadius);
                let mask = crate::settled_for_raster(&position, config);
                let set = mask.iter().filter(|b| **b).count();
                let mut hash = 1469598103934665603u64;
                for (index, bit) in mask.iter().enumerate() {
                    if *bit {
                        hash ^= index as u64;
                        hash = hash.wrapping_mul(1099511628211);
                    }
                }
                println!("  {units}u {count:>3} stones: {set:>6} settled  hash {hash:016x}");
            }
        }
    }
}
