//! The raster in the form inference wants to receive it.
//!
//! Staging is what limits inference, not arithmetic. The two ONNX lanes
//! measured 87% CPU each while thirty-four actor threads sat at 39%, and the
//! card drew 120 W of its ~300: the host was busy copying and the GPU waited.
//!
//! Most of those bytes carry nothing. Of the seven `compact-radius` planes,
//! three hold one value repeated across every pixel and three more are
//! strictly binary, so at 256x256 a float32 raster spends 1792 KB to say what
//! fits in 152 KB.
//!
//! Writing this form is *cheaper* than writing the dense one, not merely
//! cheaper to send: the same loop stores twelve times less memory, and the
//! constant planes collapse from two `fill`s over 65,536 floats into two
//! stores. The graph expands it again on the device -- see
//! `training/vgo_training/packed_input.py`, whose `compress` is the reference
//! this has to agree with, and `bit_expansion_table` for the bit order.

use std::ops::Range;

use half::f16;
use vgo_core::{Color, Point, Position};

use crate::edt::{EdtScratch, settled_mask_by_incremental_append_into};
use crate::{RasterConfig, RasterKind, settled_for_raster_into};

/// Which planes of a layout are binary, continuous, and constant.
///
/// Mirrors `_LAYOUTS` in `vgo_training/packed_states.py`. The two are separate
/// statements of the same fact and have to agree; the round-trip test in
/// `tests` and `test_packed_input.py` each check one direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedLayout {
    /// Channels holding only 0.0 or 1.0, stored one bit per pixel.
    pub binary: &'static [usize],
    /// Channels needing full precision, stored fp16 per pixel.
    pub continuous: &'static [usize],
    /// Channels constant across the plane, stored as one fp16 each.
    pub scalar: &'static [usize],
}

impl PackedLayout {
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.binary.len() + self.continuous.len() + self.scalar.len()
    }
}

/// `compact-radius`: current_stones, opponent_stones, voronoi_ridge, settled,
/// komi, previous_pass, radius.
pub const COMPACT_RADIUS_PACKED: PackedLayout = PackedLayout {
    binary: &[0, 1, 3],
    continuous: &[2],
    scalar: &[4, 5, 6],
};

/// The layout for a raster kind, or `None` where one is not defined.
#[must_use]
pub const fn packed_layout_for(kind: RasterKind) -> Option<PackedLayout> {
    match kind {
        RasterKind::CompactRadius => Some(COMPACT_RADIUS_PACKED),
        _ => None,
    }
}

/// Bytes a bit plane occupies, rounding the final partial byte up.
#[must_use]
pub const fn bit_plane_bytes(pixels: usize) -> usize {
    pixels.div_ceil(8)
}

/// One position, in the three tensors the packed graph takes.
// No `Eq`: the fp16 planes are floats, and a raster comparing equal to
// itself is not something anything here relies on.
#[derive(Debug)]
pub struct PackedRaster {
    config: RasterConfig,
    layout: PackedLayout,
    bits: Vec<u8>,
    dense: Vec<f16>,
    scalars: Vec<f16>,
    scratch: PackedScratch,
}

#[derive(Debug, Default)]
struct PackedScratch {
    edt: EdtScratch,
    settled: Vec<bool>,
    stone_xs: Vec<f64>,
    stone_ys: Vec<f64>,
    stone_is_current: Vec<bool>,
}

impl Clone for PackedRaster {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            layout: self.layout,
            bits: self.bits.clone(),
            dense: self.dense.clone(),
            scalars: self.scalars.clone(),
            // Scratch is deliberately not cloned: cloning an inference input
            // should copy the raster, not duplicate its retained work buffers.
            scratch: PackedScratch::default(),
        }
    }
}

impl PartialEq for PackedRaster {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.layout == other.layout
            && self.bits == other.bits
            && self.dense == other.dense
            && self.scalars == other.scalars
    }
}

impl PackedRaster {
    /// Allocates for `config`, zeroed.
    ///
    /// # Panics
    /// If `config.kind` has no packed layout.
    #[must_use]
    pub fn new(config: RasterConfig) -> Self {
        let layout = packed_layout_for(config.kind)
            .expect("raster kind has no packed layout; see packed_layout_for");
        let pixels = config.pixels();
        Self {
            config,
            layout,
            bits: vec![0; layout.binary.len() * bit_plane_bytes(pixels)],
            dense: vec![f16::ZERO; layout.continuous.len() * pixels],
            scalars: vec![f16::ZERO; layout.scalar.len()],
            scratch: PackedScratch::default(),
        }
    }

    #[must_use]
    pub const fn config(&self) -> RasterConfig {
        self.config
    }

    #[must_use]
    pub const fn layout(&self) -> PackedLayout {
        self.layout
    }

    /// The binary planes, `bit_plane_bytes(pixels)` bytes each, in
    /// `layout.binary` order. Bit `i` of byte `b` is pixel `8 * b + i`, so the
    /// lowest bit comes first -- the order `numpy.packbits(bitorder="little")`
    /// produces and the graph's expansion table assumes.
    #[must_use]
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// The continuous planes at fp16, `pixels` values each, in
    /// `layout.continuous` order.
    #[must_use]
    pub fn dense(&self) -> &[f16] {
        &self.dense
    }

    /// One value per constant plane, in `layout.scalar` order.
    #[must_use]
    pub fn scalars(&self) -> &[f16] {
        &self.scalars
    }

    /// Bytes this occupies, against the dense float32 raster it replaces.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bits.len() + 2 * self.dense.len() + 2 * self.scalars.len()
    }

    fn bit_plane(&self, slot: usize) -> Range<usize> {
        let stride = bit_plane_bytes(self.config.pixels());
        slot * stride..(slot + 1) * stride
    }

    /// Reads one pixel of a binary plane. For tests and for debugging; the
    /// production path never reads back what it wrote.
    #[must_use]
    pub fn binary_pixel(&self, slot: usize, pixel: usize) -> bool {
        let plane = self.bit_plane(slot);
        self.bits[plane.start + pixel / 8] & (1 << (pixel % 8)) != 0
    }
}

/// Experimental append-only raster state.
///
/// This is a prototype for game positions that are a direct child of the
/// previous position and did not capture. It keeps the exact nearest and
/// second-nearest stone labels so the stone-derived planes can be updated for
/// one appended stone. The legal set is updated by clearing only the newly
/// occupied exclusion disc; its exact distance transform is then recomputed.
/// Captures deliberately return `false` so callers can use the normal full
/// raster path.
#[doc(hidden)]
pub struct IncrementalPackedRaster {
    raster: PackedRaster,
    nearest_squares: Vec<f64>,
    settled_nearest_squares: Vec<f64>,
    second_squares: Vec<f64>,
    nearest_stones: Vec<usize>,
    second_stones: Vec<usize>,
    candidate_bounds: Vec<f64>,
}

const CANDIDATE_TILE: usize = 16;

impl IncrementalPackedRaster {
    #[must_use]
    pub fn new(position: &Position, config: RasterConfig) -> Self {
        let raster = rasterize_packed(position, config);
        let mut this = Self {
            raster,
            nearest_squares: vec![f64::INFINITY; config.pixels()],
            settled_nearest_squares: vec![f64::INFINITY; config.pixels()],
            second_squares: vec![f64::INFINITY; config.pixels()],
            nearest_stones: vec![usize::MAX; config.pixels()],
            second_stones: vec![usize::MAX; config.pixels()],
            candidate_bounds: vec![
                f64::INFINITY;
                config.width.div_ceil(CANDIDATE_TILE)
                    * config.height.div_ceil(CANDIDATE_TILE)
            ],
        };
        this.rebuild_stone_fields(position);
        this
    }

    #[must_use]
    pub fn raster(&self) -> &PackedRaster {
        &self.raster
    }

    /// Replace the cached position after a capture or another non-append
    /// transition. The next append can then use [`Self::update_no_capture`]
    /// again.
    pub fn replace(&mut self, position: &Position) {
        self.raster.scratch.edt.invalidate_legal();
        rasterize_compact_radius_packed_into(position, self.raster.config, &mut self.raster);
        self.rebuild_stone_fields(position);
    }

    fn rebuild_stone_fields(&mut self, position: &Position) {
        let width = self.raster.config.width;
        let height = self.raster.config.height;
        self.nearest_squares.fill(f64::INFINITY);
        self.settled_nearest_squares.fill(f64::INFINITY);
        self.second_squares.fill(f64::INFINITY);
        self.nearest_stones.fill(usize::MAX);
        self.second_stones.fill(usize::MAX);
        for row in 0..height {
            let y = (row as f64 + 0.5) / height as f64;
            for column in 0..width {
                let x = (column as f64 + 0.5) / width as f64;
                let pixel = row * width + column;
                for (stone_index, stone) in position.stones().iter().enumerate() {
                    let dx = x - stone.x;
                    let dy = y - stone.y;
                    let square = dx * dx + dy * dy;
                    let settled_square = dx.mul_add(dx, dy * dy);
                    let settled_nearest = &mut self.settled_nearest_squares[pixel];
                    if settled_square < *settled_nearest {
                        *settled_nearest = settled_square;
                    }
                    if square < self.nearest_squares[pixel] {
                        self.second_squares[pixel] = self.nearest_squares[pixel];
                        self.second_stones[pixel] = self.nearest_stones[pixel];
                        self.nearest_squares[pixel] = square;
                        self.nearest_stones[pixel] = stone_index;
                    } else if square < self.second_squares[pixel] {
                        self.second_squares[pixel] = square;
                        self.second_stones[pixel] = stone_index;
                    }
                }
            }
        }
        self.rebuild_candidate_bounds();
    }

    fn rebuild_candidate_bounds(&mut self) {
        let width = self.raster.config.width;
        let height = self.raster.config.height;
        let tiles_width = width.div_ceil(CANDIDATE_TILE);
        self.candidate_bounds.fill(0.0);
        for row in 0..height {
            let tile_row = row / CANDIDATE_TILE;
            for column in 0..width {
                let pixel = row * width + column;
                let tile = tile_row * tiles_width + column / CANDIDATE_TILE;
                self.candidate_bounds[tile] = self.candidate_bounds[tile]
                    .max(self.second_squares[pixel])
                    .max(self.settled_nearest_squares[pixel]);
            }
        }
    }

    /// Apply an append-only, no-capture transition.
    ///
    /// Returns `true` when the transition was handled incrementally. A false
    /// result leaves this raster unchanged. Call [`Self::replace`] to
    /// resynchronize it through the full raster path.
    pub fn update_no_capture(&mut self, previous: &Position, next: &Position) -> bool {
        let previous_stones = previous.stones();
        let next_stones = next.stones();
        if next.radius() != previous.radius()
            || next_stones.len() != previous_stones.len() + 1
            || next_stones[..previous_stones.len()] != *previous_stones
        {
            return false;
        }
        let width = self.raster.config.width;
        let height = self.raster.config.height;
        let pixels = self.raster.config.pixels();
        let new_index = previous_stones.len();
        let new_stone = next_stones[new_index];
        let radius = next.radius();
        let radius_square = radius * radius;

        // The full writer only leaves a reusable legal mask on its EDT path.
        // If the previous position used the cheap small-board fallback, let
        // the caller do one full render to establish that state.
        if self.raster.scratch.edt.legal_len() != pixels {
            return false;
        }

        // A new stone can change a pixel only when it beats that pixel's
        // current second-nearest stone. The tile bounds are conservative upper
        // bounds, so skipping a tile cannot skip a change. They also include
        // the FMA nearest field used by settled classification.
        let tiles_width = width.div_ceil(CANDIDATE_TILE);
        let tiles_height = height.div_ceil(CANDIDATE_TILE);
        for tile_row in 0..tiles_height {
            let low_row = tile_row * CANDIDATE_TILE;
            let high_row = ((tile_row + 1) * CANDIDATE_TILE).min(height);
            for tile_column in 0..tiles_width {
                let tile = tile_row * tiles_width + tile_column;
                let low_column = tile_column * CANDIDATE_TILE;
                let high_column = ((tile_column + 1) * CANDIDATE_TILE).min(width);
                let low_x = (low_column as f64 + 0.5) / width as f64;
                let high_x = ((high_column - 1) as f64 + 0.5) / width as f64;
                let low_y = (low_row as f64 + 0.5) / height as f64;
                let high_y = ((high_row - 1) as f64 + 0.5) / height as f64;
                let dx = if new_stone.x < low_x {
                    low_x - new_stone.x
                } else if new_stone.x > high_x {
                    new_stone.x - high_x
                } else {
                    0.0
                };
                let dy = if new_stone.y < low_y {
                    low_y - new_stone.y
                } else if new_stone.y > high_y {
                    new_stone.y - high_y
                } else {
                    0.0
                };
                if dx.mul_add(dx, dy * dy) >= self.candidate_bounds[tile] {
                    continue;
                }
                for row in low_row..high_row {
                    let y = (row as f64 + 0.5) / height as f64;
                    for column in low_column..high_column {
                        let pixel = row * width + column;
                        let x = (column as f64 + 0.5) / width as f64;
                        let dx = x - new_stone.x;
                        let dy = y - new_stone.y;
                        let settled_square = dx.mul_add(dx, dy * dy);
                        if settled_square < self.settled_nearest_squares[pixel] {
                            self.settled_nearest_squares[pixel] = settled_square;
                        }
                        let square = dx * dx + dy * dy;
                        let changed = if square < self.nearest_squares[pixel] {
                            self.second_squares[pixel] = self.nearest_squares[pixel];
                            self.second_stones[pixel] = self.nearest_stones[pixel];
                            self.nearest_squares[pixel] = square;
                            self.nearest_stones[pixel] = new_index;
                            true
                        } else if square < self.second_squares[pixel] {
                            self.second_squares[pixel] = square;
                            self.second_stones[pixel] = new_index;
                            true
                        } else {
                            false
                        };
                        if changed {
                            self.raster.dense[pixel] = f16::from_f32(ridge_at(
                                self.nearest_squares[pixel],
                                self.second_squares[pixel],
                                radius,
                            ));
                        }
                    }
                }
            }
        }

        assert!(settled_mask_by_incremental_append_into(
            next,
            self.raster.config,
            Point::new(new_stone.x, new_stone.y),
            &self.settled_nearest_squares,
            &mut self.raster.scratch.edt,
            &mut self.raster.scratch.settled,
        ));

        let stride = bit_plane_bytes(pixels);
        let (current_plane, rest) = self.raster.bits.split_at_mut(stride);
        let (opponent_plane, settled_plane) = rest.split_at_mut(stride);
        settled_plane.fill(0);
        for pixel in 0..pixels {
            set_bit(settled_plane, pixel, self.raster.scratch.settled[pixel]);
        }

        if next.to_move() != previous.to_move() {
            for byte in 0..stride {
                std::mem::swap(&mut current_plane[byte], &mut opponent_plane[byte]);
            }
        }
        let target_plane = if new_stone.color == next.to_move() {
            current_plane
        } else {
            opponent_plane
        };
        let low_row = (((new_stone.y - radius) * height as f64 - 0.5).floor()).max(0.0) as usize;
        let high_row = ((((new_stone.y + radius) * height as f64 - 0.5).ceil()) as usize)
            .min(height - 1);
        let low_column =
            (((new_stone.x - radius) * width as f64 - 0.5).floor()).max(0.0) as usize;
        let high_column = ((((new_stone.x + radius) * width as f64 - 0.5).ceil()) as usize)
            .min(width - 1);
        for row in low_row..=high_row {
            let y = (row as f64 + 0.5) / height as f64;
            let dy = y - new_stone.y;
            let dy_square = dy * dy;
            for column in low_column..=high_column {
                let x = (column as f64 + 0.5) / width as f64;
                let dx = x - new_stone.x;
                if dx * dx + dy_square <= radius_square {
                    set_bit(target_plane, row * width + column, true);
                }
            }
        }

        self.raster.scalars[0] = f16::from_f32(match next.to_move() {
            Color::Black => next.komi() as f32,
            Color::White => -next.komi() as f32,
        });
        self.raster.scalars[1] = f16::from_f32(f32::from(next.consecutive_passes() > 0));
        self.raster.scalars[2] = f16::from_f32((2.0 * radius) as f32);
        true
    }
}


/// Stones bucketed by cell, so a chunk of pixels can find the few that matter.
///
/// The sweep it replaces is O(pixels x stones): every stone is tested against
/// every pixel, and on a 38-unit board running to 312 plies that is 150+ stones
/// against 65,536 pixels, roughly ten million distance computations per raster.
/// A profile put the sweeps at 29% of the generator's CPU.
///
/// Cell size is one exclusion diameter, so a cell holds *at most two* stones --
/// not one, as the tempting argument goes: two stones at opposite corners of a
/// `2r` cell are `2r*sqrt(2)` apart, comfortably above the `2r` minimum
/// separation. Hence buckets rather than a single slot per cell.
///
/// CSR rather than `Vec<Vec<_>>`: the grid is rebuilt for every rasterization,
/// and a hundred small allocations per raster would cost more than the scan it
/// is replacing.
struct StoneGrid {
    cell: f64,
    width: usize,
    height: usize,
    starts: Vec<u32>,
    indices: Vec<u32>,
}

impl StoneGrid {
    fn build(xs: &[f64], ys: &[f64], cell: f64) -> Self {
        debug_assert_eq!(xs.len(), ys.len());
        let width = (1.0 / cell).ceil() as usize + 1;
        let height = width;
        let cells = width * height;
        let mut counts = vec![0u32; cells + 1];
        let index_of = |x: f64, y: f64| {
            let cx = ((x / cell) as usize).min(width - 1);
            let cy = ((y / cell) as usize).min(height - 1);
            cy * width + cx
        };
        for (x, y) in xs.iter().zip(ys) {
            counts[index_of(*x, *y) + 1] += 1;
        }
        for i in 0..cells {
            counts[i + 1] += counts[i];
        }
        let mut cursor = counts.clone();
        let mut indices = vec![0u32; xs.len()];
        for (stone, (x, y)) in xs.iter().zip(ys).enumerate() {
            let cell_index = index_of(*x, *y);
            indices[cursor[cell_index] as usize] = stone as u32;
            cursor[cell_index] += 1;
        }
        Self { cell, width, height, starts: counts, indices }
    }

    fn empty(cell: f64) -> Self {
        Self { cell, width: 1, height: 1, starts: vec![0, 0], indices: Vec::new() }
    }

    #[inline]
    fn bucket(&self, cx: usize, cy: usize) -> &[u32] {
        let cell = cy * self.width + cx;
        let (start, end) = (self.starts[cell] as usize, self.starts[cell + 1] as usize);
        &self.indices[start..end]
    }
}

/// Pixels per chunk, once chunking is worth doing.
///
/// The box a chunk needs grows with its width, so a wider chunk gathers more
/// stones; a narrower one amortizes the ring walk over fewer pixels. Measured
/// at radius 1/38, 256x256, sweep time against the flat scan:
///
/// ```text
/// CHUNK    28 stones   60    120    240
///    16        0.28x  0.53  1.30   3.62
///    32        0.39   0.78  1.78   4.48
///    64        0.51   0.95  2.05   5.16
///   128        0.54   1.05  1.99   4.49
///   256        0.53   1.18  1.79   3.87
/// ```
///
/// 64 wins where it matters. Half of generated games are 38-unit boards running
/// to 312 plies, so by sample count roughly three quarters of positions come
/// from boards carrying over a hundred stones.
const CHUNK: usize = 64;

/// Below this many stones, scan every stone instead.
///
/// The search only pays when the bound actually prunes, and with few stones
/// every stone is inside it anyway -- so the ring walk, the grid indirection
/// and the per-ring bound scan are pure overhead. Setting the chunk to the full
/// row is not enough to recover it, because that machinery still runs; the flat
/// scan has to be a separate path.
///
/// The measured break-even is around 60 stones; keeping the threshold there
/// avoids paying the grid setup on early positions while helping late ones.
const SEARCH_MINIMUM_STONES: usize = 60;

/// Every stone against every pixel, one stone at a time.
///
/// What the dense writer does, and what wins below `SEARCH_MINIMUM_STONES`.
/// Stones outside and pixels inside, so the inner loop is a flat sequence of
/// independent minima over contiguous f64 -- the shape the autovectorizer
/// handles.
#[allow(clippy::too_many_arguments)]
fn sweep_row_flat(
    stone_xs: &[f64],
    stone_ys: &[f64],
    stone_is_current: &[bool],
    y: f64,
    xs: &[f64],
    current_squares: &mut [f64],
    opponent_squares: &mut [f64],
    nearest_squares: &mut [f64],
    second_squares: &mut [f64],
) {
    for stone in 0..stone_xs.len() {
        let (sx, sy) = (stone_xs[stone], stone_ys[stone]);
        let dy = y - sy;
        let dy_square = dy * dy;
        let target = if stone_is_current[stone] {
            &mut *current_squares
        } else {
            &mut *opponent_squares
        };
        for column in 0..xs.len() {
            let dx = xs[column] - sx;
            let square = dx * dx + dy_square;
            if square < target[column] {
                target[column] = square;
            }
            if square < nearest_squares[column] {
                second_squares[column] = nearest_squares[column];
                nearest_squares[column] = square;
            } else if square < second_squares[column] {
                second_squares[column] = square;
            }
        }
    }
}

/// Nearest and second-nearest squared distances over all stones, plus the
/// nearest over each colour, for one row -- searching outward from each chunk
/// instead of scanning every stone.
///
/// The bound: a stone whose closest possible approach to the chunk already
/// exceeds every pixel's current second-nearest distance cannot change any of
/// them, and if it also exceeds `radius^2` it cannot set a stone-disc bit
/// either. Rings are walked outward so the near stones land first and pull that
/// bound down quickly; ring `k` is at least `(k-1) * cell` away, so once that
/// passes the bound every later ring does too.
///
/// Writes exactly what the all-stones scan writes. `min` and the two-smallest
/// update are order-independent for the values they produce, so visiting stones
/// in a different order changes nothing -- which is what lets this be checked
/// bit-for-bit against the dense writer.
#[allow(clippy::too_many_arguments)]
fn sweep_row_chunked(
    grid: &StoneGrid,
    stone_xs: &[f64],
    stone_ys: &[f64],
    stone_is_current: &[bool],
    y: f64,
    xs: &[f64],
    radius_square: f64,
    current_squares: &mut [f64],
    opponent_squares: &mut [f64],
    nearest_squares: &mut [f64],
    second_squares: &mut [f64],
) {
    let width = xs.len();
    let cell = grid.cell;
    let row_cell = ((y / cell) as usize).min(grid.height - 1);
    for start in (0..width).step_by(CHUNK) {
        let end = (start + CHUNK).min(width);
        let (low_x, high_x) = (xs[start], xs[end - 1]);
        let cx0 = ((low_x / cell) as usize).min(grid.width - 1);
        let cx1 = ((high_x / cell) as usize).min(grid.width - 1);

        let mut bound = f64::INFINITY;
        let longest = grid.width.max(grid.height);
        for ring in 0..=longest {
            if ring > 0 {
                let reach = (ring - 1) as f64 * cell;
                let reach_square = reach * reach;
                if reach_square >= bound && reach_square > radius_square {
                    break;
                }
            }
            let mut touched = false;
            for cy in row_cell.saturating_sub(ring)..=(row_cell + ring).min(grid.height - 1) {
                let edge_y = cy.abs_diff(row_cell) == ring;
                let from = cx0.saturating_sub(ring);
                let to = (cx1 + ring).min(grid.width - 1);
                for cx in from..=to {
                    // Only the ring itself, not the filled block: inner cells
                    // were visited by an earlier, closer ring.
                    let edge_x = cx + ring == cx0 || cx == cx1 + ring;
                    if !(edge_y || edge_x) {
                        continue;
                    }
                    touched = true;
                    for &stone in grid.bucket(cx, cy) {
                        let stone = stone as usize;
                        let (sx, sy) = (stone_xs[stone], stone_ys[stone]);
                        let dy = y - sy;
                        let dy_square = dy * dy;
                        // Closest the chunk can come to this stone.
                        let gap = if sx < low_x {
                            low_x - sx
                        } else if sx > high_x {
                            sx - high_x
                        } else {
                            0.0
                        };
                        let floor_square = gap.mul_add(gap, dy_square);
                        if floor_square >= bound && floor_square > radius_square {
                            continue;
                        }
                        let target = if stone_is_current[stone] {
                            &mut *current_squares
                        } else {
                            &mut *opponent_squares
                        };
                        for column in start..end {
                            let dx = xs[column] - sx;
                            let square = dx.mul_add(dx, dy_square);
                            if square < target[column] {
                                target[column] = square;
                            }
                            if square < nearest_squares[column] {
                                second_squares[column] = nearest_squares[column];
                                nearest_squares[column] = square;
                            } else if square < second_squares[column] {
                                second_squares[column] = square;
                            }
                        }
                    }
                }
            }
            // Recompute the bound once per ring, not once per stone: it only
            // ever decreases, so a stale value prunes less but never wrongly,
            // and a per-stone reduction over the chunk would cost as much as
            // the scan being avoided.
            let mut worst: f64 = 0.0;
            for column in start..end {
                if second_squares[column] > worst {
                    worst = second_squares[column];
                }
            }
            bound = worst;
            if !touched && ring > 0 && bound.is_finite() {
                // Nothing left in this ring and the bound is real; the next
                // rings are further still.
                let reach = ring as f64 * cell;
                if reach * reach >= bound && reach * reach > radius_square {
                    break;
                }
            }
        }
    }
}

/// The ridge value for one pixel, from the nearest and second-nearest squared
/// distances. Matches the dense writer's expression exactly, including doing
/// the square roots before the subtraction.
#[inline]
fn ridge_at(nearest_square: f64, second_square: f64, radius: f64) -> f32 {
    let nearest = nearest_square.sqrt();
    let second = second_square.sqrt();
    // No `is_finite` guard, so the two roots vectorize into `vsqrtpd`: with the
    // branch in place a profile put this function at 22.7% of all samples,
    // about what 131,072 scalar roots per raster costs.
    //
    // `max` then `min` rather than `clamp`, and that is the whole subtlety.
    // With one stone the subtraction is `inf`, the expression is `-inf`, and
    // either spelling gives 0.0. With *no* stones both distances are infinite,
    // `inf - inf` is NaN, and `clamp` propagates NaN where `max` returns the
    // other operand. Every game starts from an empty board, so the NaN version
    // reached the model on the first position of the first game and the run
    // died with "invalid inference value".
    (1.0 - (second - nearest) / radius).max(0.0).min(1.0) as f32
}

#[inline]
fn set_bit(plane: &mut [u8], pixel: usize, value: bool) {
    // Branchless: the store happens either way, so a mispredicted branch costs
    // more than the OR of a zero.
    plane[pixel / 8] |= u8::from(value) << (pixel % 8);
}

/// Allocates and fills a packed raster, mirroring `rasterize`.
///
/// The reusing form is `rasterize_compact_radius_packed_into`; prefer it on the
/// hot path, where one buffer per actor outlives every position it renders.
#[must_use]
pub fn rasterize_packed(position: &Position, config: RasterConfig) -> PackedRaster {
    let mut out = PackedRaster::new(config);
    rasterize_compact_radius_packed_into(position, config, &mut out);
    out
}

/// Writes `position` into `out` in packed form.
///
/// Produces exactly what `rasterize_compact_radius_into` produces, then
/// compressed -- the tests assert that against the dense writer rather than
/// against a second implementation of the same arithmetic.
///
/// # Panics
/// If `out` was not built for a `compact-radius` config matching `config`.
pub fn rasterize_compact_radius_packed_into(
    position: &Position,
    config: RasterConfig,
    out: &mut PackedRaster,
) {
    assert_eq!(config.kind, RasterKind::CompactRadius);
    assert_eq!(out.config, config);
    assert!(config.width > 0 && config.height > 0);
    // Caller invariant, not this function's business, and an O(n^2)
    // sweep per rasterization if checked in release. See `game::place`.
    debug_assert!(position.validate().is_playable());

    let pixels = config.pixels();
    let width = config.width;
    let radius = position.radius();
    let radius_square = radius * radius;
    let to_move = position.to_move();

    // The three constant planes, as three stores rather than three fills over
    // 65,536 floats each.
    let mover_komi = match to_move {
        Color::Black => position.komi() as f32,
        Color::White => -position.komi() as f32,
    };
    out.scalars[0] = f16::from_f32(mover_komi);
    out.scalars[1] = f16::from_f32(f32::from(position.consecutive_passes() > 0));
    out.scalars[2] = f16::from_f32((2.0 * radius) as f32);


    // Destructured once, so the loops below work through plain slices. Indexing
    // `out.dense` while `out.bits` is separately borrowed leaves the compiler
    // unable to prove the Vec's data pointer is loop-invariant, and it reloads
    // it every pixel: measured 0.49 ms against 0.35 ms for the same arithmetic
    // through a hoisted slice.
    let PackedRaster {
        bits,
        dense: dense_plane,
        scratch,
        ..
    } = out;
    let PackedScratch {
        edt,
        settled,
        stone_xs,
        stone_ys,
        stone_is_current,
    } = scratch;
    for byte in bits.iter_mut() {
        *byte = 0;
    }

    // There is no geometry to sweep on the first position of every game. Clear
    // the reusable continuous plane and return before allocating scratch or
    // taking 65,536 `sqrt(inf - inf)` paths for a ridge that is known to be 0.
    if position.stones().is_empty() {
        dense_plane.fill(f16::ZERO);
        return;
    }

    settled_for_raster_into(position, config, edt, settled);
    assert_eq!(settled.len(), pixels);
    let settled = settled.as_slice();

    let stride = bit_plane_bytes(pixels);
    let (current_plane, rest) = bits.split_at_mut(stride);
    let (opponent_plane, settled_plane) = rest.split_at_mut(stride);
    let dense_plane = dense_plane.as_mut_slice();

    // Stones flattened with a colour flag, so one grid serves all three minima.
    // The dense writer keeps two lists and scans each in full; the grid decides
    // which stones a chunk of pixels can possibly need.
    let stones = position.stones();
    stone_xs.clear();
    stone_ys.clear();
    stone_is_current.clear();
    stone_xs.reserve(stones.len());
    stone_ys.reserve(stones.len());
    stone_is_current.reserve(stones.len());
    for stone in stones {
        stone_xs.push(stone.x);
        stone_ys.push(stone.y);
        stone_is_current.push(stone.color == to_move);
    }
    let stone_xs = stone_xs.as_slice();
    let stone_ys = stone_ys.as_slice();
    let stone_is_current = stone_is_current.as_slice();
    // Cell size from the stone count, not from the radius. `2r` is the natural
    // geometric choice -- it caps a cell at two stones -- but on a sparse board
    // it leaves the grid 93% empty, and the ring walk then spends its time
    // stepping over cells that hold nothing. One stone per cell on average
    // keeps the walk proportional to the stones it finds. Never finer than 2r,
    // which is where the bucket bound comes from.
    let searching = stone_xs.len() >= SEARCH_MINIMUM_STONES;
    let spacing = if stone_xs.is_empty() {
        2.0 * radius
    } else {
        (2.0 * radius).max(1.0 / (stone_xs.len() as f64).sqrt())
    };
    let grid = if searching {
        StoneGrid::build(&stone_xs, &stone_ys, spacing)
    } else {
        StoneGrid::empty(spacing)
    };

    let mut row_storage = vec![f64::INFINITY; 5 * width];
    let (xs, row_storage) = row_storage.split_at_mut(width);
    for (column, x) in xs.iter_mut().enumerate() {
        *x = (column as f64 + 0.5) / width as f64;
    }
    let (current_squares, row_storage) = row_storage.split_at_mut(width);
    let (opponent_squares, row_storage) = row_storage.split_at_mut(width);
    let (nearest_squares, second_squares) = row_storage.split_at_mut(width);

    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        current_squares.fill(f64::INFINITY);
        opponent_squares.fill(f64::INFINITY);
        nearest_squares.fill(f64::INFINITY);
        second_squares.fill(f64::INFINITY);

        if searching {
            sweep_row_chunked(
                &grid,
                &stone_xs,
                &stone_ys,
                &stone_is_current,
                y,
                xs,
                radius_square,
                current_squares,
                opponent_squares,
                nearest_squares,
                second_squares,
            );
        } else {
            sweep_row_flat(
                &stone_xs,
                &stone_ys,
                &stone_is_current,
                y,
                xs,
                current_squares,
                opponent_squares,
                nearest_squares,
                second_squares,
            );
        }

        // Build each output byte in a register and store it once. Setting bits
        // in place instead costs a read-modify-write on the same byte for eight
        // consecutive pixels, and that dependency chain stops the loop
        // vectorizing -- measured 2.9x the dense writer's time, against 1.2x
        // this way.
        //
        // Groups of eight align with byte boundaries only because `width` is a
        // multiple of eight, which makes `row * width` one too. The general
        // path below carries a running bit position across rows instead.
        if width % 8 == 0 {
            for group in 0..width / 8 {
                let base = row * width + group * 8;
                let (mut current, mut opponent, mut settled_bits) = (0u8, 0u8, 0u8);
                for bit in 0..8 {
                    let column = group * 8 + bit;
                    let pixel = base + bit;
                    current |= u8::from(current_squares[column] <= radius_square) << bit;
                    opponent |= u8::from(opponent_squares[column] <= radius_square) << bit;
                    settled_bits |= u8::from(settled[pixel]) << bit;
                    dense_plane[pixel] = f16::from_f32(ridge_at(
                        nearest_squares[column],
                        second_squares[column],
                        radius,
                    ));
                }
                let byte = base / 8;
                current_plane[byte] = current;
                opponent_plane[byte] = opponent;
                settled_plane[byte] = settled_bits;
            }
        } else {
            for column in 0..width {
                let pixel = row * width + column;
                set_bit(current_plane, pixel, current_squares[column] <= radius_square);
                set_bit(opponent_plane, pixel, opponent_squares[column] <= radius_square);
                set_bit(settled_plane, pixel, settled[pixel]);
                dense_plane[pixel] = f16::from_f32(ridge_at(
                    nearest_squares[column],
                    second_squares[column],
                    radius,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position, Stone};

    use super::*;
    use crate::rasterize_compact_radius_into;

    fn position(komi: f64, passes: u32) -> Position {
        Position::new(
            1.0 / 18.0,
            vec![
                Stone::new(0.30, 0.30, Color::Black),
                Stone::new(0.62, 0.44, Color::White),
                Stone::new(0.45, 0.70, Color::Black),
                Stone::new(0.18, 0.81, Color::White),
            ],
            Color::White,
        )
        .with_komi(komi)
        .with_passes(passes)
    }

    /// A board dense enough that the grid actually prunes.
    ///
    /// The four-stone fixture below exercises none of the search: with that few
    /// stones every ring is visited and nothing is skipped, so it would pass
    /// even if the bound were wrong. This is the case that can fail.
    #[test]
    fn the_chunked_sweep_matches_at_real_stone_counts() {
        // Both sides of `CHUNKED_SWEEP_MINIMUM_STONES`, and a radius small
        // enough to fit counts above it: at 1/18 the board holds ~128 stones,
        // so a threshold of 100 would barely be crossed. Without the high
        // counts the chunked path is not exercised at all and the dispatch
        // hides it.
        let radius = 1.0 / 38.0;
        let step = 2.2 * radius;
        for count in [0usize, 1, 2, 9, 28, 60, 99, 100, 140, 240] {
            let mut stones = Vec::new();
            let mut index = 0;
            // `count == 0` is an empty board -- the first position of every
            // game, and where `inf - inf` becomes NaN. The loop below tests
            // `index == count` after incrementing, so zero would fill it.
            let wanted = count;
            'outer: for row in 0..24 {
                if wanted == 0 {
                    break;
                }
                for column in 0..24 {
                    let x = 0.04 + step * f64::from(column);
                    let y = 0.04 + step * f64::from(row);
                    if x > 0.97 || y > 0.97 {
                        continue;
                    }
                    stones.push(Stone::new(
                        x,
                        y,
                        if index % 2 == 0 { Color::Black } else { Color::White },
                    ));
                    index += 1;
                    if index == count {
                        break 'outer;
                    }
                }
            }
            if stones.len() < count {
                continue;
            }
            // Zero stones is not a corner case: it is the first position of
            // every game, and it is where `inf - inf` becomes NaN.
            let position = Position::new(radius, stones, Color::White).with_komi(0.024);
            for size in [64usize, 128, 256] {
                let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
                let pixels = config.pixels();
                let mut dense = vec![0.0f32; config.channels() * pixels];
                rasterize_compact_radius_into(&position, config, &mut dense);
                let mut packed = PackedRaster::new(config);
                rasterize_compact_radius_packed_into(&position, config, &mut packed);
                let layout = packed.layout();
                for (slot, &channel) in layout.binary.iter().enumerate() {
                    for pixel in 0..pixels {
                        assert_eq!(
                            packed.binary_pixel(slot, pixel),
                            dense[channel * pixels + pixel] != 0.0,
                            "{count} stones at {size}: binary channel {channel}, pixel {pixel}"
                        );
                    }
                }
                for pixel in 0..pixels {
                    assert_eq!(
                        packed.dense()[pixel],
                        f16::from_f32(dense[2 * pixels + pixel]),
                        "{count} stones at {size}: ridge, pixel {pixel}"
                    );
                }
            }
        }
    }

    /// The packed form has to be the dense form compressed, not a second
    /// implementation that agrees approximately. Every plane is checked against
    /// the dense writer's own output rather than against recomputed arithmetic.
    #[test]
    fn the_packed_writer_matches_the_dense_one() {
        // Widths both divisible and not divisible by eight: the writer takes a
        // different path for each, and only the first is the production size.
        for (size, komi, passes) in [
            (64, 0.104, 0),
            (64, -0.081, 1),
            (45, 0.104, 0),
            (45, -0.081, 1),
            (13, 0.0, 0),
        ] {
            let position = position(komi, passes);
            let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
            let pixels = config.pixels();

            let mut dense = vec![0.0f32; config.channels() * pixels];
            rasterize_compact_radius_into(&position, config, &mut dense);

            let mut packed = PackedRaster::new(config);
            rasterize_compact_radius_packed_into(&position, config, &mut packed);

            let layout = packed.layout();
            for (slot, &channel) in layout.binary.iter().enumerate() {
                for pixel in 0..pixels {
                    let expected = dense[channel * pixels + pixel] != 0.0;
                    assert_eq!(
                        packed.binary_pixel(slot, pixel),
                        expected,
                        "binary channel {channel}, pixel {pixel}"
                    );
                }
            }
            for (slot, &channel) in layout.continuous.iter().enumerate() {
                for pixel in 0..pixels {
                    let expected = f16::from_f32(dense[channel * pixels + pixel]);
                    assert_eq!(
                        packed.dense()[slot * pixels + pixel],
                        expected,
                        "continuous channel {channel}, pixel {pixel}"
                    );
                }
            }
            for (slot, &channel) in layout.scalar.iter().enumerate() {
                let expected = f16::from_f32(dense[channel * pixels]);
                assert_eq!(packed.scalars()[slot], expected, "scalar channel {channel}");
            }
        }
    }

    /// A hundred chained updates, each checked against a full render.
    ///
    /// Two updates cannot show drift, and drift is the entire risk of carrying
    /// state: each step is exact, and the hundredth is wrong because the
    /// ninety-nine before it each moved something by an ulp. A real game plays
    /// 312 plies on a 38-unit board, so the incremental path is asked to stay
    /// exact over hundreds of appends, not two.
    ///
    /// Compares the whole `PackedRaster` -- every bit plane, every fp16 ridge
    /// value, every scalar -- at every step, so a divergence is caught at the
    /// step it appears rather than at the end.
    #[test]
    fn incremental_stays_exact_over_a_long_chain() {
        let radius = 1.0 / 38.0;
        let step = 2.2 * radius;
        let place = |index: usize| {
            Stone::new(
                0.04 + step * f64::from((index % 15) as u32),
                0.04 + step * f64::from((index / 15) as u32),
                if index % 2 == 0 { Color::Black } else { Color::White },
            )
        };
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);

        // Start above DISTANCE_SETTLED_MINIMUM_STONES so the first position is
        // already on the EDT path; below it the writer leaves no reusable legal
        // mask and the incremental path correctly refuses.
        let start = 30usize;
        let mut stones: Vec<Stone> = (0..start).map(place).collect();
        let mut previous = Position::new(radius, stones.clone(), Color::White).with_komi(0.104);
        let mut state = IncrementalPackedRaster::new(&previous, config);
        let mut expected = PackedRaster::new(config);

        let mut taken = 0usize;
        for index in start..start + 100 {
            stones.push(place(index));
            let to_move = if index % 2 == 0 { Color::White } else { Color::Black };
            let next = Position::new(radius, stones.clone(), to_move).with_komi(0.104);
            if !next.validate().is_playable() {
                break;
            }
            if state.update_no_capture(&previous, &next) {
                taken += 1;
                rasterize_compact_radius_packed_into(&next, config, &mut expected);
                assert_eq!(
                    state.raster(),
                    &expected,
                    "diverged at append {} ({} stones)",
                    index - start,
                    next.stones().len()
                );
            } else {
                state.replace(&next);
            }
            previous = next;
        }
        assert!(taken >= 50, "only {taken} appends took the incremental path");
    }

#[test]
    fn incremental_no_capture_matches_a_full_raster() {
        let radius = 1.0 / 38.0;
        let mut previous_stones = Vec::new();
        for index in 0..60 {
            let row = index / 16;
            let column = index % 16;
            previous_stones.push(Stone::new(
                0.04 + 2.2 * radius * f64::from(column),
                0.04 + 2.2 * radius * f64::from(row),
                if index % 2 == 0 { Color::Black } else { Color::White },
            ));
        }
        let mut next_stones = previous_stones.clone();
        next_stones.push(Stone::new(
            0.04 + 2.2 * radius * f64::from(60 % 16),
            0.04 + 2.2 * radius * f64::from(60 / 16),
            Color::Black,
        ));
        let previous = Position::new(radius, previous_stones, Color::White).with_komi(0.104);
        let next = Position::new(radius, next_stones, Color::Black).with_komi(0.104);
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);

        let mut incremental = IncrementalPackedRaster::new(&previous, config);
        assert!(incremental.update_no_capture(&previous, &next));

        let mut expected = PackedRaster::new(config);
        rasterize_compact_radius_packed_into(&next, config, &mut expected);
        assert_eq!(incremental.raster(), &expected);

        let mut next2_stones = next.stones().to_vec();
        next2_stones.push(Stone::new(
            0.04 + 2.2 * radius * f64::from(61 % 16),
            0.04 + 2.2 * radius * f64::from(61 / 16),
            Color::White,
        ));
        let next2 = Position::new(radius, next2_stones, Color::White).with_komi(0.104);
        assert!(incremental.update_no_capture(&next, &next2));
        rasterize_compact_radius_packed_into(&next2, config, &mut expected);
        assert_eq!(incremental.raster(), &expected);
    }

    #[test]
    #[ignore]
    fn incremental_no_capture_timing_probe() {
        use std::hint::black_box;
        use std::time::Instant;

        let radius = 1.0 / 38.0;
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let mut stones = Vec::new();
        let mut positions = Vec::new();
        for count in 0..=240 {
            positions.push(Position::new(radius, stones.clone(), Color::White));
            if count == 240 {
                break;
            }
            let index = count;
            let row = index / 16;
            let column = index % 16;
            stones.push(Stone::new(
                0.04 + 2.2 * radius * f64::from(column),
                0.04 + 2.2 * radius * f64::from(row),
                if index % 2 == 0 { Color::Black } else { Color::White },
            ));
        }
        let rounds = 3;
        let started = Instant::now();
        for _ in 0..rounds {
            let mut incremental = IncrementalPackedRaster::new(&positions[0], config);
            for pair in positions.windows(2) {
                if !incremental.update_no_capture(&pair[0], &pair[1]) {
                    incremental.replace(&pair[1]);
                }
            }
            black_box(incremental.raster());
        }
        let incremental_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);

        let started = Instant::now();
        for _ in 0..rounds {
            let mut full = PackedRaster::new(config);
            for position in &positions {
                rasterize_compact_radius_packed_into(position, config, &mut full);
            }
            black_box(&full);
        }
        let full_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);
        println!(
            "incremental {incremental_ms:.2} ms vs full {full_ms:.2} ms for {} appends",
            positions.len() - 1
        );
    }

    /// The dense planes really are constant, which is the premise of storing
    /// one value for each. If a future layout change broke that, the packed
    /// form would quietly carry the first pixel's value across the board.
    #[test]
    fn the_scalar_planes_are_constant_in_the_dense_writer() {
        let position = position(0.104, 1);
        let config = RasterConfig::square_of(48, RasterKind::CompactRadius);
        let pixels = config.pixels();
        let mut dense = vec![0.0f32; config.channels() * pixels];
        rasterize_compact_radius_into(&position, config, &mut dense);
        for &channel in COMPACT_RADIUS_PACKED.scalar {
            let plane = &dense[channel * pixels..(channel + 1) * pixels];
            assert!(
                plane.iter().all(|value| *value == plane[0]),
                "channel {channel} is not constant"
            );
        }
    }

    /// Likewise the binary planes: a bit cannot carry 0.5.
    #[test]
    fn the_binary_planes_are_binary_in_the_dense_writer() {
        let position = position(0.0, 0);
        let config = RasterConfig::square_of(48, RasterKind::CompactRadius);
        let pixels = config.pixels();
        let mut dense = vec![0.0f32; config.channels() * pixels];
        rasterize_compact_radius_into(&position, config, &mut dense);
        for &channel in COMPACT_RADIUS_PACKED.binary {
            let plane = &dense[channel * pixels..(channel + 1) * pixels];
            assert!(
                plane.iter().all(|value| *value == 0.0 || *value == 1.0),
                "channel {channel} is not binary"
            );
        }
    }

    /// Bit `i` of byte `b` is pixel `8b + i`. This is the order
    /// `numpy.packbits(bitorder="little")` produces, which the graph's
    /// expansion table assumes; the two are separate statements of one fact.
    #[test]
    fn the_lowest_bit_holds_the_first_pixel() {
        let mut plane = vec![0u8; 2];
        set_bit(&mut plane, 0, true);
        set_bit(&mut plane, 3, true);
        set_bit(&mut plane, 8, true);
        assert_eq!(plane[0], 0b0000_1001);
        assert_eq!(plane[1], 0b0000_0001);
    }

    /// A pixel count that is not a multiple of eight leaves spare bits in the
    /// last byte, which must stay zero rather than aliasing a neighbour.
    #[test]
    fn a_partial_final_byte_is_padded() {
        assert_eq!(bit_plane_bytes(25), 4);
        assert_eq!(bit_plane_bytes(64), 8);
        assert_eq!(bit_plane_bytes(0), 0);
    }

    /// Writing the packed form should also be *faster* than writing the dense
    /// one -- it stores twelve times less and collapses two full-plane fills
    /// into two stores. Ignored by default because it is a measurement, not an
    /// assertion: run with
    /// `cargo test --release -p vgo-raster packed_is_cheaper -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn the_packed_write_is_cheaper() {
        use std::time::Instant;
        // Twenty-eight stones, the median of real shards. The four-stone
        // fixture the equivalence tests use makes `settled` almost free and so
        // overstates every per-pixel cost relative to production. Placed on a
        // lattice coarser than a diameter, because overlapping stones are not a
        // playable position.
        let radius = 1.0 / 18.0;
        let step = 2.5 * radius;
        let mut stones = Vec::new();
        let mut index = 0;
        'outer: for row in 0..8 {
            for column in 0..8 {
                let x = 0.08 + step * f64::from(column);
                let y = 0.08 + step * f64::from(row);
                if x > 0.95 || y > 0.95 {
                    continue;
                }
                let colour = if index % 2 == 0 { Color::Black } else { Color::White };
                stones.push(Stone::new(x, y, colour));
                index += 1;
                if index == 28 {
                    break 'outer;
                }
            }
        }
        let position = Position::new(radius, stones, Color::White).with_komi(0.104);
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let pixels = config.pixels();
        let rounds = 30;

        let mut dense = vec![0.0f32; config.channels() * pixels];
        let mut packed = PackedRaster::new(config);
        // Warm both paths so the first allocation is not in either timing.
        rasterize_compact_radius_into(&position, config, &mut dense);
        rasterize_compact_radius_packed_into(&position, config, &mut packed);

        // `settled` is the same call in both writers, so timing it alone says
        // how much of each total is shared and how much is actually different.
        let started = Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(crate::settled_for_raster(&position, config));
        }
        let settled_elapsed = started.elapsed();

        let started = Instant::now();
        for _ in 0..rounds {
            rasterize_compact_radius_into(&position, config, &mut dense);
        }
        let dense_elapsed = started.elapsed();

        let started = Instant::now();
        for _ in 0..rounds {
            rasterize_compact_radius_packed_into(&position, config, &mut packed);
        }
        let packed_elapsed = started.elapsed();

        let dense_bytes = config.channels() * pixels * 4;
        let per = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / f64::from(rounds);
        println!(
            "  settled (shared by both)  {:>7.2} ms\n  dense minus settled       {:>7.2} ms\n  packed minus settled      {:>7.2} ms",
            per(settled_elapsed),
            per(dense_elapsed) - per(settled_elapsed),
            per(packed_elapsed) - per(settled_elapsed),
        );
        println!(
            "  dense  {:>7.2} ms/raster  {:>6} KB\n  packed {:>7.2} ms/raster  {:>6} KB\n  {:.2}x time, {:.1}x bytes",
            dense_elapsed.as_secs_f64() * 1000.0 / f64::from(rounds),
            dense_bytes / 1024,
            packed_elapsed.as_secs_f64() * 1000.0 / f64::from(rounds),
            packed.bytes() / 1024,
            packed_elapsed.as_secs_f64() / dense_elapsed.as_secs_f64(),
            dense_bytes as f64 / packed.bytes() as f64,
        );
    }

    /// The sweep across the stone counts the board mix actually produces.
    ///
    /// Half of generated games are 38-unit boards running to 312 plies, so the
    /// expensive positions carry hundreds of stones -- not the 28 the benchmark
    /// above uses, which is a mini-board figure and has misled every estimate
    /// made from it tonight.
    ///
    /// `cargo test --release -p vgo-raster stone_count_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn the_stone_count_sweep() {
        use std::time::Instant;
        let radius = 1.0 / 38.0;
        let step = 2.2 * radius;
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let pixels = config.pixels();
        println!("  radius 1/38, 256x256, dense vs chunked");
        for count in [28usize, 60, 120, 240] {
            let mut stones = Vec::new();
            let mut index = 0;
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
                        if index % 2 == 0 { Color::Black } else { Color::White },
                    ));
                    index += 1;
                    if index == count {
                        break 'outer;
                    }
                }
            }
            if stones.len() < count {
                continue;
            }
            let position = Position::new(radius, stones, Color::White).with_komi(0.024);
            let rounds = 24;
            let mut dense = vec![0.0f32; config.channels() * pixels];
            let mut packed = PackedRaster::new(config);
            let settled = crate::settled_for_raster(&position, config);
            std::hint::black_box(&settled);

            rasterize_compact_radius_into(&position, config, &mut dense);
            let started = Instant::now();
            for _ in 0..rounds {
                rasterize_compact_radius_into(&position, config, &mut dense);
            }
            let dense_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);

            rasterize_compact_radius_packed_into(&position, config, &mut packed);
            let started = Instant::now();
            for _ in 0..rounds {
                rasterize_compact_radius_packed_into(&position, config, &mut packed);
            }
            let packed_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);

            let started = Instant::now();
            for _ in 0..rounds {
                std::hint::black_box(crate::settled_for_raster(&position, config));
            }
            let settled_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);

            println!(
                "  {count:>3} stones: settled {settled_ms:>5.2}  dense-sweep {:>5.2}  chunked-sweep {:>5.2}  ({:.2}x)",
                dense_ms - settled_ms,
                packed_ms - settled_ms,
                (dense_ms - settled_ms) / (packed_ms - settled_ms).max(1e-9)
            );
        }
    }

    #[test]
    fn the_packed_form_is_far_smaller() {
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let packed = PackedRaster::new(config);
        let dense = config.channels() * config.pixels() * 4;
        assert!(
            packed.bytes() * 10 < dense,
            "packed {} vs dense {dense}",
            packed.bytes()
        );
    }
}
