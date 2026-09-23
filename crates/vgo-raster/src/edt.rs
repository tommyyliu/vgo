//! The settled mask as a comparison of two distance fields.
//!
//! `vgo-core`'s `settled.rs` solves a per-stone radial equation, extracts a
//! contour and scanline-fills it. That is `O(n²)` in the stone count and 92-96%
//! of rasterization cost. This is a different formulation of the same set:
//!
//! ```text
//!     settled(x)  <=>  D_S(x) <= D_L(x)
//! ```
//!
//! where `D_S` is the distance to the nearest stone and `D_L` the distance to
//! the legal set. Both are distance transforms, so the whole thing is two
//! sweeps over the grid instead of a geometric solve per stone — `O(pixels)`
//! rather than `O(pixels · n²)`, which means it stays flat as stones accumulate
//! instead of tripling between 28 and 52.
//!
//! `D_L` uses the exact separable Euclidean transform (Felzenszwalb and
//! Huttenlocher, *Distance Transforms of Sampled Functions*): a 1-D lower
//! envelope of parabolas per column, then per row, each linear in the row
//! length. `D_S` is taken from the continuous stone coordinates rather than the
//! grid, because it is cheap and exact there.
//!
//! **This is an approximation, unlike the other two implementations.** `D_L` is
//! the distance to the nearest pixel *centre* lying in the legal set, not to the
//! continuous set, so it is an overestimate by up to half a pixel diagonal. An
//! overestimated `D_L` makes the comparison easier to satisfy, so this errs
//! toward reporting too much settled. `resolution` oversamples the mask to
//! shrink that; `examples/settled_edt.rs` measures what it costs.

use vgo_core::{
    COORDINATE_EPSILON, LegalSetIndex, Point, Position,
    no_legal_point_closer_than_indexed,
};

use crate::RasterConfig;

/// Stands in for "no source here".
///
/// A literal infinity breaks the parabola intersection below: two infinite
/// parabolas give inf - inf, and the resulting NaN or -inf walks `k` off the
/// bottom of the hull. A finite sentinel keeps every intersection finite and
/// well-ordered, and is far beyond any real squared distance on a unit board
/// (the largest is 2, in pixel units at most `2 * size²`).
const ABSENT: f64 = 1.0e20;

/// Squared exact Euclidean distance transform of `f`, in place into `d`.
///
/// `f` holds 0 where the set is present and [`ABSENT`] elsewhere. Scratch
/// buffers are passed in so the 2-D driver can reuse them across every row and
/// column.
fn transform_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = intersection(f, q, v[k]);
        // `k > 0` is a guard, not an optimisation: z[0] is -inf and a finite
        // sentinel keeps s finite, so s <= z[0] is false in the well-behaved
        // case -- but a degenerate row should not be able to underflow.
        while k > 0 && s <= z[k] {
            k -= 1;
            s = intersection(f, q, v[k]);
        }
        if k == 0 && s <= z[0] {
            v[0] = q;
            z[1] = f64::INFINITY;
            continue;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let offset = q as f64 - v[k] as f64;
        d[q] = offset * offset + f[v[k]];
    }
}

fn intersection(f: &[f64], q: usize, vk: usize) -> f64 {
    let (fq, fv) = (f[q], f[vk]);
    let (qf, vf) = (q as f64, vk as f64);
    ((fq + qf * qf) - (fv + vf * vf)) / (2.0 * qf - 2.0 * vf)
}

/// Squared distance, in pixel units, from each cell to the nearest `true` cell.
/// Columns processed this many at a time.
///
/// A column gather strides by `width`, so it touches one cache line per row and
/// uses eight of its sixty-four bytes. Doing eight columns together uses the
/// whole line, which cuts the pass's memory traffic eightfold -- and that
/// traffic, not the transform, is where the time went: a profile put the
/// gather/scatter wrapper at 18.6% of all samples against `transform_1d`'s
/// 9.8%, so the bookkeeping cost twice the algorithm it wraps.
///
/// Eight because a cache line holds eight `f64`. Sixteen would halve the passes
/// again but doubles the scratch and starts missing L1 on the transposed side.
const COLUMN_BLOCK: usize = 8;

/// Pixel columns processed together by the nearest-stone search. The same
/// chunking shape is used by the packed writer: it amortizes the grid-ring
/// walk without making the chunk's bound so wide that nearby stones are kept.
const NEAREST_CHUNK: usize = 128;

/// Below this many stones, the flat vectorizable scan is faster than building
/// and walking a spatial index.
const NEAREST_SEARCH_MINIMUM_STONES: usize = 96;

/// Buffers retained by a caller that renders many positions at one raster
/// size. Their contents are scratch, not part of a raster's meaning.
#[derive(Debug, Default)]
pub(crate) struct EdtScratch {
    legal: Vec<bool>,
    field: Vec<f64>,
    sources: Vec<f64>,
    results: Vec<f64>,
    vertices: Vec<usize>,
    boundaries: Vec<f64>,
    fine_column_xs: Vec<f64>,
    fine_row_ys: Vec<f64>,
    column_xs: Vec<f64>,
    fine_columns: Vec<usize>,
    nearest_squares: Vec<f64>,
}

impl EdtScratch {

    /// The squared sampled distance field, for a caller classifying `settled`
    /// a row at a time.
    pub(crate) fn field(&self) -> &[f64] {
        &self.field
    }

}

pub(crate) fn squared_distance_transform(mask: &[bool], width: usize, height: usize) -> Vec<f64> {
    let mut scratch = EdtScratch::default();
    squared_distance_transform_into(mask, width, height, &mut scratch);
    scratch.field
}

fn squared_distance_transform_into(
    mask: &[bool],
    width: usize,
    height: usize,
    scratch: &mut EdtScratch,
) {
    scratch.field.resize(mask.len(), 0.0);
    for (value, inside) in scratch.field.iter_mut().zip(mask) {
        *value = if *inside { 0.0 } else { ABSENT };
    }

    let longest = width.max(height);
    scratch.sources.resize(COLUMN_BLOCK * longest, 0.0);
    scratch.results.resize(COLUMN_BLOCK * longest, 0.0);
    scratch.vertices.resize(longest, 0);
    scratch.boundaries.resize(longest + 1, 0.0);

    // Columns, a cache line's worth at a time. `transform_1d` still sees one
    // contiguous line and is untouched; only the order of the gather changes,
    // and a minimum does not care in which order it is fed.
    for block in (0..width).step_by(COLUMN_BLOCK) {
        let columns = COLUMN_BLOCK.min(width - block);
        for row in 0..height {
            let base = row * width + block;
            for column in 0..columns {
                scratch.sources[column * longest + row] = scratch.field[base + column];
            }
        }
        for column in 0..columns {
            let from = column * longest;
            transform_1d(
                &scratch.sources[from..from + height],
                &mut scratch.results[from..from + height],
                &mut scratch.vertices,
                &mut scratch.boundaries,
            );
        }
        for row in 0..height {
            let base = row * width + block;
            for column in 0..columns {
                scratch.field[base + column] = scratch.results[column * longest + row];
            }
        }
    }

    // Rows are already contiguous, so they need none of that.
    for row in 0..height {
        let base = row * width;
        scratch.sources[..width].copy_from_slice(&scratch.field[base..base + width]);
        transform_1d(
            &scratch.sources[..width],
            &mut scratch.results[..width],
            &mut scratch.vertices,
            &mut scratch.boundaries,
        );
        scratch.field[base..base + width].copy_from_slice(&scratch.results[..width]);
    }
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

/// The legal set sampled onto a grid, built by stamping exclusion discs.
pub(crate) fn sampled_legal_set(position: &Position, fine_width: usize, fine_height: usize) -> Vec<bool> {
    let mut scratch = EdtScratch::default();
    sampled_legal_set_into(position, fine_width, fine_height, &[], &mut scratch);
    scratch.legal
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
    scratch: &mut EdtScratch,
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
            for column in low_column..=high_column {
                let x = scratch.fine_column_xs[column];
                let dx = x - stone.x;
                if dx.mul_add(dx, dy_squared) < exclusion_squared {
                    scratch.legal[base + column] = false;
                }
            }
        }
    }
    for vertex in vertices {
        let column = ((vertex.x * fine_width as f64) as usize).min(fine_width - 1);
        let row = ((vertex.y * fine_height as f64) as usize).min(fine_height - 1);
        scratch.legal[row * fine_width + column] = true;
    }
}

/// The legal set's vertices, bucketed so an undecided pixel can look for one
/// nearer than its own stone before paying for the exact test.
///
/// A vertex is an exact legal point. If one is strictly closer to a pixel than
/// the pixel's nearest stone, a new stone could be placed nearer than the owner
/// and the pixel is not settled -- no further geometry needed. The undecided
/// band is mostly pixels beside a thin legal gap, which are exactly the pixels
/// the vertex marks rescued from being wrongly called settled, and a gap's own
/// vertices are right there. Marks are few -- a handful per legal speck -- so
/// the buckets are coarse and mostly empty.
struct VertexGrid {
    cell: f64,
    side: usize,
    starts: Vec<u32>,
    points: Vec<Point>,
}

impl VertexGrid {
    fn build(marks: &[Point], cell: f64) -> Self {
        let side = ((1.0 / cell).ceil() as usize).max(1);
        let bucket = |p: &Point| {
            let column = ((p.x / cell) as usize).min(side - 1);
            let row = ((p.y / cell) as usize).min(side - 1);
            row * side + column
        };
        let mut counts = vec![0u32; side * side + 1];
        for mark in marks {
            counts[bucket(mark) + 1] += 1;
        }
        for index in 1..counts.len() {
            counts[index] += counts[index - 1];
        }
        let mut fill = counts.clone();
        let mut points = vec![Point::new(0.0, 0.0); marks.len()];
        for mark in marks {
            let slot = &mut fill[bucket(mark)];
            points[*slot as usize] = *mark;
            *slot += 1;
        }
        Self { cell, side, starts: counts, points }
    }

    /// Whether a vertex lies strictly closer than `distance` to `point`. Errs
    /// toward no at the boundary, which only sends a pixel to the exact test.
    fn any_closer(&self, point: Point, distance: f64) -> bool {
        if self.points.is_empty() {
            return false;
        }
        let reach = distance * (1.0 - 1e-9) - 1e-12;
        if reach <= 0.0 {
            return false;
        }
        let reach_squared = reach * reach;
        let low = |v: f64| (((v - reach) / self.cell).floor().max(0.0) as usize).min(self.side - 1);
        let high = |v: f64| (((v + reach) / self.cell).floor().max(0.0) as usize).min(self.side - 1);
        for row in low(point.y)..=high(point.y) {
            let base = row * self.side;
            let (first, last) = (
                self.starts[base + low(point.x)] as usize,
                self.starts[base + high(point.x) + 1] as usize,
            );
            for mark in &self.points[first..last] {
                let (dx, dy) = (mark.x - point.x, mark.y - point.y);
                if dx.mul_add(dx, dy * dy) < reach_squared {
                    return true;
                }
            }
        }
        false
    }
}

/// The settled mask, via distance transforms.
///
/// `oversample` multiplies the grid the legal-set distance is measured on; 1
/// samples it at the output resolution, 4 at four times that in each axis. The
/// output is always `config.pixels()` long.
#[must_use]
pub fn settled_mask_by_distance(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
) -> Vec<bool> {
    let pixels = config.pixels();
    let stones = position.stones();
    if stones.is_empty() {
        // No stone can own anything, and the comparison would be inf <= inf.
        return vec![false; pixels];
    }
    let scale = oversample.max(1);
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);

    let legal = sampled_legal_set(position, fine_width, fine_height);
    let squared = squared_distance_transform(&legal, fine_width, fine_height);

    let mut mask = vec![false; pixels];
    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        // Centre of the output cell in fine-grid coordinates.
        let fine_row = (row * scale + scale / 2).min(fine_height - 1);
        for column in 0..config.width {
            let x = (column as f64 + 0.5) / config.width as f64;
            let fine_column = (column * scale + scale / 2).min(fine_width - 1);

            let mut nearest = f64::INFINITY;
            for stone in stones {
                let (dx, dy) = (x - stone.x, y - stone.y);
                let distance = dx.mul_add(dx, dy * dy);
                if distance < nearest {
                    nearest = distance;
                }
            }
            let legal_squared = squared[fine_row * fine_width + fine_column];
            // Both sides are squared distances; the pixel-unit one is scaled
            // into normalised coordinates by the fine grid's spacing.
            let spacing = 1.0 / fine_width as f64;
            let legal_normalised = legal_squared * spacing * spacing;
            mask[row * config.width + column] = nearest <= legal_normalised;
        }
    }
    mask
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
/// exact continuous test — 55 of 16384 at oversample 1, so the cost is
/// dominated by the cheap path, which is the point: the exact test is what made
/// the direct formulation 42x too slow to use everywhere.
///
/// **The upper bound is not guaranteed.** It assumes every point of the legal
/// set has a sampled cell centre within half a diagonal, which fails where the
/// legal set is a sliver thinner than the grid — then `D_L_grid` overshoots by
/// more than the slack and a pixel can be called settled when it is not. The
/// lower direction (`D_S > D_L_grid` implies not settled) is always sound,
/// since every legal cell centre really is a point of the set.
///
/// In practice that costs at most one pixel of 16384 at the densest fixture
/// tested, against two for the shipping implementation, which walks a contour
/// at 1/128 tolerance and is not exact either. Oversampling to 3 removed it
/// entirely on every fixture.
///
/// Returns the mask and how many pixels needed the exact test, so callers can
/// see whether the band is staying small.
#[must_use]
pub fn settled_mask_by_bounded_distance(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
) -> (Vec<bool>, usize) {
    let mut scratch = EdtScratch::default();
    let mut mask = Vec::new();
    let tests = settled_by_bounded_distance_into(position, config, oversample, &mut scratch, &mut mask);
    (mask, tests)
}

pub(crate) fn settled_mask_by_bounded_distance_into(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
    scratch: &mut EdtScratch,
    mask: &mut Vec<bool>,
) {
    settled_by_bounded_distance_into(position, config, oversample, scratch, mask);
}

/// The bounded-distance `settled` test, split so a caller can supply the
/// nearest-stone field instead of having one computed for it.
///
/// `settled_by_bounded_distance_into` sweeps every stone against every pixel to
/// find the nearest one -- and the packed writer then sweeps them all again for
/// the ridge, which needs that same minimum. Measured at 240 stones the settled
/// sweep is 0.397 ms of a 1.56 ms raster, spent recomputing a field the caller
/// is already holding. This half does the row-invariant work; `classify_row`
/// does the rest against minima the caller passes in.
///
/// The two sweeps agree bit for bit only because every squared distance in the
/// raster is now `mul_add`. That is load-bearing: `settled` is a *bit* decided
/// by comparing this distance against a sampled one, so unlike the ridge -- which
/// quantises to f16 and swallows a last-ulp difference -- one ulp here is a
/// different pixel.
///
/// `oversample` is chosen by `settled_oversample`, the same as the reference.
pub(crate) struct SettledRows {
    scale: usize,
    fine_width: usize,
    fine_height: usize,
    half_diagonal: f64,
    spacing_squared: f64,
    slack: f64,
    index: Option<LegalSetIndex>,
    vertices: VertexGrid,
    pub(crate) exact_tests: usize,
}

pub(crate) fn prepare_settled_rows(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
    scratch: &mut EdtScratch,
) -> SettledRows {
    let scale = oversample.max(1) | 1;
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);
    let index = LegalSetIndex::build(position);
    sampled_legal_set_into(position, fine_width, fine_height, index.vertices(), scratch);
    let legal = std::mem::take(&mut scratch.legal);
    squared_distance_transform_into(&legal, fine_width, fine_height, scratch);
    scratch.legal = legal;
    let spacing = 1.0 / fine_width as f64;
    let vertices = VertexGrid::build(index.vertices(), (2.0 * position.radius()).max(spacing));
    SettledRows {
        vertices,
        scale,
        fine_width,
        fine_height,
        half_diagonal: 0.5 * spacing * std::f64::consts::SQRT_2,
        spacing_squared: spacing * spacing,
        slack: spacing * std::f64::consts::SQRT_2,
        index: Some(index),
        exact_tests: 0,
    }
}

impl SettledRows {
    /// Classify one row. `nearest_squares` is the caller's per-column minimum
    /// over every stone, `field` the squared sampled distance transform, and
    /// `out` the row's slice of the settled mask.
    ///
    /// The three cases and why rounding cannot break them are documented on the
    /// equivalent loop in `masks_by_bounded_distance_into`, which stays the
    /// reference implementation this is pinned against.
    pub(crate) fn classify_row(
        &mut self,
        position: &Position,
        row: usize,
        width: usize,
        y: f64,
        xs: &[f64],
        nearest_squares: &[f64],
        field: &[f64],
        out: &mut [bool],
    ) {
        let scale = self.scale;
        let fine_row = (row * scale + scale / 2).min(self.fine_height - 1);
        let base = fine_row * self.fine_width;
        for column in 0..width {
            let fine_column = (column * scale + scale / 2).min(self.fine_width - 1);
            let sampled_squared = field[base + fine_column] * self.spacing_squared;
            let sampled = sampled_squared.sqrt();
            let sampled_minus_slack = sampled - self.slack;
            out[column] = if sampled_minus_slack > 0.0
                && nearest_squares[column] <= sampled_minus_slack * sampled_minus_slack
            {
                true
            } else if nearest_squares[column]
                > (sampled + self.half_diagonal) * (sampled + self.half_diagonal)
            {
                false
            } else if self
                .vertices
                .any_closer(Point::new(xs[column], y), nearest_squares[column].sqrt())
            {
                false
            } else {
                self.exact_tests += 1;
                let known = self
                    .index
                    .get_or_insert_with(|| LegalSetIndex::build(position));
                let nearest = nearest_squares[column].sqrt();
                no_legal_point_closer_than_indexed(
                    position,
                    Point::new(xs[column], y),
                    nearest,
                    known,
                )
            };
        }
    }
}

fn settled_by_bounded_distance_into(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
    scratch: &mut EdtScratch,
    mask: &mut Vec<bool>,
) -> usize {
    let pixels = config.pixels();
    let stones = position.stones();
    mask.resize(pixels, false);
    mask.fill(false);
    // Nothing owns anything on a stoneless board.
    if stones.is_empty() {
        return 0;
    }
    let radius = position.radius();
    let scale = oversample.max(1) | 1;
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);
    let known_index = LegalSetIndex::build(position);
    sampled_legal_set_into(position, fine_width, fine_height, known_index.vertices(), scratch);
    let legal = std::mem::take(&mut scratch.legal);
    squared_distance_transform_into(&legal, fine_width, fine_height, scratch);
    scratch.legal = legal;
    let spacing = 1.0 / fine_width as f64;
    let spacing_squared = spacing * spacing;
    let slack = spacing * std::f64::consts::SQRT_2;
    let half_diagonal = 0.5 * slack;
    let vertex_grid = VertexGrid::build(known_index.vertices(), (2.0 * radius).max(spacing));
    // How far the sampled distance can overstate the true one.
    //
    // Half a cell diagonal is the tempting answer and it is wrong: it assumes
    // every point of the legal set has a *cell centre* within that distance,
    // which fails all along the set's boundary, where the cell containing a
    // legal point can easily have its centre outside. Measured, that cost two
    // wrong pixels at eight stones, where slivers cannot be the explanation.
    // A full diagonal covers the boundary case; nothing covers a sliver
    // narrower than a cell, which is why this function is not exact.
    let mut index: Option<LegalSetIndex> = Some(known_index);
    let mut exact_tests = 0usize;
    // The nearest stone, a row at a time, stones outside and pixels inside.
    //
    // This loop used to be 85% of `settled` and was documented as such. It is
    // 24% now, measured at 240 stones -- the classification below is the larger
    // half, and most of that is the exact test in the undecided band. Do not
    // take the old figure as a reason to optimise here first.
    //
    // The packed writer no longer runs this at all: it classifies each row
    // against the nearest-stone minimum its own sweep already computed, which
    // is what the comment below hinted at and never acted on. This path serves
    // the dense writer and stays the reference the fused one is pinned against.
    //
    // Sparse positions use the flat min below, over contiguous f64 that the
    // autovectorizer handles; dense positions use the exact chunked grid query,
    // which avoids work on distant stones.
    let row_width = config.width;
    scratch.column_xs.resize(row_width, 0.0);
    scratch.fine_columns.resize(row_width, 0);
    for (column, x) in scratch.column_xs.iter_mut().enumerate() {
        *x = (column as f64 + 0.5) / row_width as f64;
        scratch.fine_columns[column] = (column * scale + scale / 2).min(fine_width - 1);
    }
    scratch.nearest_squares.resize(row_width, f64::INFINITY);
    let nearest_grid = if stones.len() >= NEAREST_SEARCH_MINIMUM_STONES {
        let cell = (2.0 * radius).max(1.0 / (stones.len() as f64).sqrt());
        Some(StoneGrid::build(position, cell))
    } else {
        None
    };
    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        let fine_row = (row * scale + scale / 2).min(fine_height - 1);
        let output_base = row * config.width;
        let sampled_base = fine_row * fine_width;
        {
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
                    let stone_x = stone.x;
                    // Zipped rather than indexed, and `min` rather than a branch:
                    // both are what let this compile to a flat vector min with no
                    // bounds checks in the loop.
                    // `mul_add`, matching `nearest_row_chunked`. The two paths
                    // must compute the same distance or the dispatch threshold
                    // becomes a knob that silently re-renders positions near it;
                    // and it is the faster of the two here anyway, measured at
                    // 0.562 ms against 0.585 at 28 stones.
                    for (nearest, &x) in scratch
                        .nearest_squares
                        .iter_mut()
                        .zip(scratch.column_xs.iter())
                    {
                        let dx = x - stone_x;
                        *nearest = dx.mul_add(dx, dy_square).min(*nearest);
                    }
                }
            }
        }
        for column in 0..config.width {
            let x = scratch.column_xs[column];
            let sampled_squared =
                scratch.field[sampled_base + scratch.fine_columns[column]] * spacing_squared;
            // Keep the cheap cases in squared-distance space. The old form
            // took two square roots for every pixel, although only the narrow
            // undecided band needs the exact nearest distance. This leaves one
            // root for the common settled case and defers the other until the
            // fallback is actually entered.
            //
            // This is not the same arithmetic: `sqrt(s) * spacing` and
            // `sqrt(s * spacing^2)` round differently, and squaring a
            // comparison moves its boundary by a few ulp. That is safe here for
            // a reason worth writing down, because it is not obvious.
            //
            // The three cases are disjoint and ordered -- `slack > 0` keeps
            // `sampled - slack` strictly below `sampled` -- so a rounding flip
            // can only move a pixel between a cheap case and the undecided
            // band, never from settled straight to unsettled. A pixel that
            // drifts into the band gets the exact test, which is authoritative
            // and agrees with whichever cheap case it came from, since both
            // cheap tests are sound implications rather than approximations.
            // So the mask is unchanged and only `exact_tests` moves.
            let sampled = sampled_squared.sqrt();
            let sampled_minus_slack = sampled - slack;
            mask[output_base + column] = if sampled_minus_slack > 0.0
                && scratch.nearest_squares[column] <= sampled_minus_slack * sampled_minus_slack
            {
                true
            } else if scratch.nearest_squares[column]
                > (sampled + half_diagonal) * (sampled + half_diagonal)
            {
                false
            } else if vertex_grid.any_closer(Point::new(x, y), scratch.nearest_squares[column].sqrt()) {
                false
            } else {
                exact_tests += 1;
                let known = index.get_or_insert_with(|| LegalSetIndex::build(position));
                let nearest = scratch.nearest_squares[column].sqrt();
                no_legal_point_closer_than_indexed(position, Point::new(x, y), nearest, known)
            };
        }
    }
    exact_tests
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Stone, distance_to_legal_set, legal_set_vertices};

    use super::*;
    use crate::{RasterKind, settled_mask};

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
                    let colour = if stones.len() % 2 == 0 { Color::Black } else { Color::White };
                    stones.push(Stone::new(x, y, colour));
                }
            }
            Position::new(radius, stones, Color::Black)
        }
        for (size, radius, target) in [(128usize, 1.0 / 38.0, 250usize), (128, 1.0 / 38.0, 400), (96, 1.0 / 18.0, 90)] {
            let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
            let position = packing(radius, target, target as u64);
            let scale = crate::settled_oversample(&position, config).expect("distance path");
            let vertices = legal_set_vertices(&position);
            let (mask, _) = settled_mask_by_bounded_distance(&position, config, scale);
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
            assert_eq!(wrong, 0, "{size}px, {target} stones: {wrong} pixels disagree");
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
            let scale = crate::settled_oversample(&position, config)
                .expect("a coarse raster must take the distance path");
            let vertices = legal_set_vertices(&position);
            let (mask, _) = settled_mask_by_bounded_distance(&position, config, scale);
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
            assert_eq!(wrong, 0, "{count} stones: {wrong} pixels disagree with the definition");
        }
    }

    /// The bounded form must agree with the definition, not merely with the
    /// other implementation -- which walks a contour at 1/128 and is itself
    /// wrong on a pixel or two at high stone counts.
    #[test]
    fn bounded_distance_agrees_with_the_definition() {
        let radius = 0.055_714_285_714_285_716;
        let config = RasterConfig::square_of(128, RasterKind::Compact);
        for count in [8usize, 28, 52] {
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
        assert_eq!(settled_mask(&position, config), mask);
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
                            if placed % 2 == 0 {
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
