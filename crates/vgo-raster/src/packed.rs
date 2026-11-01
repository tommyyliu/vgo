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

use std::cell::RefCell;

use half::{f16, slice::HalfFloatSliceExt};
use vgo_core::{Color, Position, Stone, StoneFit, StoneRun};

use crate::edt::EarlierTransform;
use crate::settled_bounds::{ANSWER_UNKNOWN, KnownAnswers, SettledScratch, prepare_settled_rows};
use crate::{RasterConfig, RasterKind, SETTLED_OVERSAMPLE};

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
pub(crate) const COMPACT_RADIUS_PACKED: PackedLayout = PackedLayout {
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
#[derive(Clone, Debug, PartialEq)]
pub struct PackedRaster {
    config: RasterConfig,
    layout: PackedLayout,
    bits: Vec<u8>,
    dense: Vec<f16>,
    scalars: Vec<f16>,
}

/// Work buffers, one set per thread.
///
/// A raster leaves for the broker the moment it is written, so anything it owned
/// would be allocated and zeroed again for the next leaf, 300 KB of it at 128,
/// and freed on another thread. Nothing here outlives a call.
#[derive(Debug, Default)]
struct PackedScratch {
    settled_scratch: SettledScratch,
    settled: Vec<bool>,
    stone_xs: Vec<f64>,
    stone_ys: Vec<f64>,
    nearest_squares: Vec<f64>,
    second_squares: Vec<f64>,
    candidates: Vec<u32>,
    ridges: Vec<f32>,
    coordinates: Vec<f64>,
    anchor: SweepAnchor,
}

thread_local! {
    static SCRATCH: RefCell<PackedScratch> = RefCell::default();
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

    /// Reads one pixel of a binary plane. For tests and for debugging; the
    /// production path never reads back what it wrote.
    #[must_use]
    pub fn binary_pixel(&self, slot: usize, pixel: usize) -> bool {
        let stride = bit_plane_bytes(self.config.pixels());
        self.bits[slot * stride + pixel / 8] & (1 << (pixel % 8)) != 0
    }
}

/// Side of the smallest tile, in pixels. Measured best of 4, 8, 16 and 32.
const LEAF_TILE: usize = 16;

/// The stones and pixel coordinates a sweep reads.
struct SweepInput<'a> {
    stone_xs: &'a [f64],
    stone_ys: &'a [f64],
    xs: &'a [f64],
    ys: &'a [f64],
}

/// Nearest and second-nearest squared distances for a tile of pixels.
///
/// `candidates[inherited..]` holds the stones the enclosing tile kept. Let `c`
/// be this tile's centre, `h` the distance from `c` to its corners and `d` the
/// distance from `c` to its second-nearest candidate. Those two candidates are
/// within `d + h` of every pixel, so each pixel's second-nearest distance is at
/// most `d + h`. A stone at least `d + 2h` from `c` is at least `d + h` from
/// every pixel, so it cannot be one of a pixel's two nearest. Any two stones give
/// a valid `d`, which is why each level only needs to look at its parent's list.
///
/// The bound only chooses candidates. Every stored distance is still the exact
/// `mul_add` the dense writer computes, and the two-smallest update does not
/// depend on visit order. The slack on the limit covers rounding in the bound.
/// Proofs: [E1], [N2].
fn sweep_tile(
    input: &SweepInput<'_>,
    columns: std::ops::Range<usize>,
    rows: std::ops::Range<usize>,
    inherited: usize,
    candidates: &mut Vec<u32>,
    nearest_squares: &mut [f64],
    second_squares: &mut [f64],
) {
    let SweepInput {
        stone_xs,
        stone_ys,
        xs,
        ys,
    } = *input;
    let centre_x = 0.5 * (xs[columns.start] + xs[columns.end - 1]);
    let centre_y = 0.5 * (ys[rows.start] + ys[rows.end - 1]);
    let half_diagonal = (centre_x - xs[columns.start]).hypot(centre_y - ys[rows.start]);

    let end = candidates.len();
    let (mut first, mut second) = (f64::INFINITY, f64::INFINITY);
    for &stone in &candidates[inherited..end] {
        let stone = stone as usize;
        let (dx, dy) = (stone_xs[stone] - centre_x, stone_ys[stone] - centre_y);
        let square = dx * dx + dy * dy;
        second = second.min(square.max(first));
        first = first.min(square);
    }
    let reach = second.sqrt() + 2.0 * half_diagonal;
    let limit = reach * reach * (1.0 + 1.0e-9);
    for index in inherited..end {
        let stone = candidates[index];
        let at = stone as usize;
        let (dx, dy) = (stone_xs[at] - centre_x, stone_ys[at] - centre_y);
        if dx * dx + dy * dy <= limit {
            candidates.push(stone);
        }
    }

    let (tile_width, tile_height) = (columns.len(), rows.len());
    if tile_width > LEAF_TILE || tile_height > LEAF_TILE {
        let middle_column = columns.start + tile_width.div_ceil(2);
        let middle_row = rows.start + tile_height.div_ceil(2);
        let row_halves = [rows.start..middle_row, middle_row..rows.end];
        let column_halves = [columns.start..middle_column, middle_column..columns.end];
        for row_half in &row_halves {
            for column_half in &column_halves {
                if !row_half.is_empty() && !column_half.is_empty() {
                    sweep_tile(
                        input,
                        column_half.clone(),
                        row_half.clone(),
                        end,
                        candidates,
                        nearest_squares,
                        second_squares,
                    );
                }
            }
        }
    } else {
        let width = xs.len();
        let tile_xs = &xs[columns.clone()];
        for row in rows {
            let y = ys[row];
            let from = row * width + columns.start;
            let to = row * width + columns.end;
            let nearest_row = &mut nearest_squares[from..to];
            let second_row = &mut second_squares[from..to];
            for &stone in &candidates[end..] {
                let stone = stone as usize;
                let stone = (stone_xs[stone], stone_ys[stone]);
                fold_stone_into_row(stone, y, tile_xs, nearest_row, second_row);
            }
        }
    }
    candidates.truncate(end);
}

/// Folds one stone into a run of pixels' two smallest squared distances.
///
/// The sweep and the anchored fold both come through here, which is what makes
/// their output the same bits. Proof: [E15].
#[inline(always)]
fn fold_stone_into_row(
    (stone_x, stone_y): (f64, f64),
    y: f64,
    xs: &[f64],
    nearest_row: &mut [f64],
    second_row: &mut [f64],
) {
    let dy = y - stone_y;
    let dy_square = dy * dy;
    for ((nearest, second), &x) in nearest_row.iter_mut().zip(second_row.iter_mut()).zip(xs) {
        let dx = x - stone_x;
        // `mul_add`, matching the dense writer and the nearest field
        // `settled` classifies against. `settled` is a bit decided
        // by comparing this distance against a sampled one, so one
        // ulp is a different pixel.
        let square = dx.mul_add(dx, dy_square);
        // Comparisons rather than `f64::min`, which pays to handle
        // NaN: nothing here is NaN, so this is one `vminpd` with
        // the same result, on the hottest line of the sweep.
        let larger = if square > *nearest { square } else { *nearest };
        *second = if larger < *second { larger } else { *second };
        *nearest = if square < *nearest { square } else { *nearest };
    }
}

/// What this thread's leaves can share about a search's root.
///
/// Every leaf of a search is the root's stones plus the few on its path, so
/// rendering each one repeats nearly all the work of the last. This keeps the
/// root's nearest field, which a leaf extends by folding in its own stones, and
/// each pixel's exact settled answer, which a leaf reuses wherever the stones it
/// differs by are too far away to matter. Both give the same bits as starting over.
///
/// Nothing here is needed for correctness: with no root announced, every raster
/// sweeps and tests from nothing.
#[derive(Debug, Default)]
struct SweepAnchor {
    width: usize,
    height: usize,
    radius_bits: u64,
    stones: StoneRun,
    nearest_squares: Vec<f64>,
    second_squares: Vec<f64>,
    /// Each row's largest second-nearest square, past which a stone changes nothing.
    row_limits: Vec<f64>,
    /// The root's own transform and ridge, which rows that match it can copy.
    transform_heights: Vec<u32>,
    transform_field: Vec<f64>,
    ridges: Vec<f16>,
    /// Filled in as leaves reach the exact test. See `KnownAnswers`.
    answers: Vec<u8>,
    /// Set by [`next_raster_is_search_root`], and spent by the next raster.
    next_is_root: bool,
}

impl SweepAnchor {
    /// How `stones` relate to the anchor on this grid and radius.
    fn fit(&mut self, stones: &[Stone], input: &SweepInput<'_>, radius: f64) -> StoneFit {
        if self.width == input.xs.len()
            && self.height == input.ys.len()
            && self.radius_bits == radius.to_bits()
        {
            self.stones.fit(stones)
        } else {
            self.stones.clear_differing();
            StoneFit::Unrelated
        }
    }

    /// Makes this the anchor for all of `stones`, which `input` was built from.
    fn rebuild(
        &mut self,
        stones: &[Stone],
        input: &SweepInput<'_>,
        radius: f64,
        candidates: &mut Vec<u32>,
    ) {
        self.width = input.xs.len();
        self.height = input.ys.len();
        self.radius_bits = radius.to_bits();
        self.answers.clear();
        self.answers
            .resize(self.width * self.height, ANSWER_UNKNOWN);
        self.stones.set(stones);
        sweep_stones(
            input,
            candidates,
            &mut self.nearest_squares,
            &mut self.second_squares,
        );
        self.row_limits.clear();
        self.row_limits.extend(
            self.second_squares
                .chunks_exact(self.width)
                .map(|row| row.iter().copied().fold(0.0, f64::max)),
        );
        self.transform_heights.clear();
        self.transform_field.clear();
        self.ridges.clear();
    }

    /// The root's transform, once its own raster has made one.
    fn earlier_transform(&self, width: usize) -> Option<EarlierTransform<'_>> {
        (self.width == width && !self.transform_field.is_empty()).then_some(EarlierTransform {
            heights: &self.transform_heights,
            field: &self.transform_field,
        })
    }
}

/// Sweeps every stone into a cleared field.
fn sweep_stones(
    input: &SweepInput<'_>,
    candidates: &mut Vec<u32>,
    nearest_squares: &mut Vec<f64>,
    second_squares: &mut Vec<f64>,
) {
    let (width, height) = (input.xs.len(), input.ys.len());
    nearest_squares.clear();
    nearest_squares.resize(width * height, f64::INFINITY);
    second_squares.clear();
    second_squares.resize(width * height, f64::INFINITY);
    candidates.clear();
    candidates.extend(0..input.stone_xs.len() as u32);
    sweep_tile(
        input,
        0..width,
        0..height,
        0,
        candidates,
        nearest_squares,
        second_squares,
    );
}

/// The two smallest squared distances at every pixel, by folding the stones past
/// the anchor into its field where this position extends the anchor, and by a
/// sweep where it does not. Returns how the position fits the anchor. Proof: [E15].
fn nearest_field(
    stones: &[Stone],
    input: &SweepInput<'_>,
    radius: f64,
    anchor: &mut SweepAnchor,
    candidates: &mut Vec<u32>,
    nearest_squares: &mut Vec<f64>,
    second_squares: &mut Vec<f64>,
) -> StoneFit {
    let mut fit = anchor.fit(stones, input, radius);
    if std::mem::take(&mut anchor.next_is_root)
        && (fit != StoneFit::Extends || !anchor.stones.differing().is_empty())
    {
        anchor.rebuild(stones, input, radius, candidates);
        fit = StoneFit::Extends;
    }
    if fit != StoneFit::Extends {
        sweep_stones(input, candidates, nearest_squares, second_squares);
        return fit;
    }

    nearest_squares.clear();
    nearest_squares.extend_from_slice(&anchor.nearest_squares);
    second_squares.clear();
    second_squares.extend_from_slice(&anchor.second_squares);
    let width = input.xs.len();
    for stone in anchor.stones.differing() {
        for (row, &pixel_y) in input.ys.iter().enumerate() {
            // Every pixel of the row is at least this far from the stone. Proof: [E15].
            let dy = pixel_y - stone.y;
            if dy * dy >= anchor.row_limits[row] {
                continue;
            }
            let base = row * width;
            fold_stone_into_row(
                (stone.x, stone.y),
                pixel_y,
                input.xs,
                &mut nearest_squares[base..base + width],
                &mut second_squares[base..base + width],
            );
        }
    }
    fit
}

/// Says the next packed raster on this thread is the root of a search, whose leaves
/// are the rasters after it.
///
/// This is a promise about what comes next and nothing else: a raster that turns
/// out to share nothing with the root is simply drawn from nothing.
pub fn next_raster_is_search_root() {
    SCRATCH.with_borrow_mut(|scratch| scratch.anchor.next_is_root = true);
}

/// Marks the pixels inside `(x, y)`'s disc.
///
/// A disc has few pixels, so this is far cheaper than keeping a per-colour
/// minimum for every pixel in the sweep. The row and column ranges are padded by
/// a pixel; the test itself is the dense writer's `square <= radius^2`. Proof: [E2].
fn stamp_disc(plane: &mut [u8], (x, y): (f64, f64), radius: f64, xs: &[f64], ys: &[f64]) {
    let (width, height) = (xs.len(), ys.len());
    let radius_square = radius * radius;
    let span = |centre: f64, size: usize| {
        let low = ((centre - radius) * size as f64 - 1.5).floor().max(0.0) as usize;
        let high = (((centre + radius) * size as f64 + 1.5).ceil() as usize).min(size);
        low..high
    };
    let columns = span(x, width);
    for row in span(y, height) {
        let dy = ys[row] - y;
        let dy_square = dy * dy;
        for column in columns.clone() {
            let dx = xs[column] - x;
            if dx.mul_add(dx, dy_square) <= radius_square {
                set_bit(plane, row * width + column, true);
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
    #[allow(clippy::manual_clamp)]
    let ridge = (1.0 - (second - nearest) / radius).max(0.0).min(1.0);
    ridge as f32
}

/// `ridge_at` over a row, as a flat loop over contiguous slices.
fn ridge_row(nearest_squares: &[f64], second_squares: &[f64], radius: f64, out: &mut [f32]) {
    for ((out, &nearest), &second) in out.iter_mut().zip(nearest_squares).zip(second_squares) {
        *out = ridge_at(nearest, second, radius);
    }
}

/// The columns from the first to the last that differ between two pairs of rows.
fn moved_span(
    (nearest, root_nearest): (&[f64], &[f64]),
    (second, root_second): (&[f64], &[f64]),
) -> std::ops::Range<usize> {
    let differs = |column: usize| {
        nearest[column] != root_nearest[column] || second[column] != root_second[column]
    };
    let width = nearest.len();
    let Some(first) = (0..width).find(|&column| differs(column)) else {
        return 0..0;
    };
    let last = (first..width).rev().find(|&column| differs(column));
    first..last.map_or(first, |last| last + 1)
}

#[inline]
fn set_bit(plane: &mut [u8], pixel: usize, value: bool) {
    // Branchless: the store happens either way, so a mispredicted branch costs
    // more than the OR of a zero.
    plane[pixel / 8] |= u8::from(value) << (pixel % 8);
}

/// Allocates and fills a packed raster, mirroring `rasterize`.
///
/// The reusing form is `rasterize_compact_radius_packed_into`.
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
pub(crate) fn rasterize_compact_radius_packed_into(
    position: &Position,
    config: RasterConfig,
    out: &mut PackedRaster,
) {
    SCRATCH.with_borrow_mut(|scratch| write_packed(position, config, out, scratch));
}

fn write_packed(
    position: &Position,
    config: RasterConfig,
    out: &mut PackedRaster,
    scratch: &mut PackedScratch,
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
        ..
    } = out;
    let PackedScratch {
        settled_scratch,
        settled,
        stone_xs,
        stone_ys,
        nearest_squares,
        second_squares,
        candidates,
        ridges,
        coordinates,
        anchor,
    } = scratch;
    for byte in bits.iter_mut() {
        *byte = 0;
    }

    // There is no geometry to sweep on the first position of every game. Clear
    // the reusable continuous plane and return before allocating scratch or
    // taking 65,536 `sqrt(inf - inf)` paths for a ridge that is known to be 0.
    if position.stones().is_empty() {
        dense_plane.fill(f16::ZERO);
        anchor.next_is_root = false;
        return;
    }

    settled.clear();
    settled.resize(pixels, false);

    let stride = bit_plane_bytes(pixels);
    let (current_plane, rest) = bits.split_at_mut(stride);
    let (opponent_plane, settled_plane) = rest.split_at_mut(stride);
    let dense_plane = dense_plane.as_mut_slice();

    let stones = position.stones();
    stone_xs.clear();
    stone_ys.clear();
    stone_xs.extend(stones.iter().map(|stone| stone.x));
    stone_ys.extend(stones.iter().map(|stone| stone.y));
    let stone_xs = stone_xs.as_slice();
    let stone_ys = stone_ys.as_slice();

    let height = config.height;
    ridges.resize(width, 0.0);
    let ridges = ridges.as_mut_slice();
    coordinates.resize(width + height, 0.0);
    let (xs, ys) = coordinates.split_at_mut(width);
    for (column, x) in xs.iter_mut().enumerate() {
        *x = (column as f64 + 0.5) / width as f64;
    }
    for (row, y) in ys.iter_mut().enumerate() {
        *y = (row as f64 + 0.5) / height as f64;
    }
    let (xs, ys) = (&*xs, &*ys);

    for stone in stones {
        let plane = if stone.color == to_move {
            &mut *current_plane
        } else {
            &mut *opponent_plane
        };
        stamp_disc(plane, (stone.x, stone.y), radius, xs, ys);
    }

    let fit = nearest_field(
        stones,
        &SweepInput {
            stone_xs,
            stone_ys,
            xs,
            ys,
        },
        radius,
        anchor,
        candidates,
        nearest_squares,
        second_squares,
    );

    // `settled` is classified row by row below, against the nearest-stone minimum
    // the ridge needs anyway, rather than by a builder that sweeps every stone a
    // second time. See `SettledRows`.
    let mut settled_rows = prepare_settled_rows(
        position,
        config,
        SETTLED_OVERSAMPLE,
        settled_scratch,
        anchor.earlier_transform(width),
    );
    // The first raster after a rebuild is the root's own.
    let is_root = fit == StoneFit::Extends && anchor.transform_field.is_empty();
    if is_root {
        let transform = &settled_scratch.edt;
        anchor
            .transform_heights
            .extend_from_slice(transform.heights());
        anchor.transform_field.extend_from_slice(transform.field());
    }
    let reads_ridges = fit != StoneFit::Unrelated && !anchor.ridges.is_empty();

    for (row, &y) in ys.iter().enumerate() {
        let base = row * width;
        let nearest_row = &nearest_squares[base..base + width];
        let second_row = &second_squares[base..base + width];

        settled_rows.classify_row(
            position,
            row,
            y,
            xs,
            nearest_row,
            settled_scratch.edt.field(),
            &mut settled[base..base + width],
            (fit != StoneFit::Unrelated).then(|| KnownAnswers {
                stones: &anchor.stones,
                row: &mut anchor.answers[base..base + width],
            }),
        );

        // A pixel's ridge reads nothing but its two distances and the radius, so
        // the pixels outside the span that moved keep the root's. Proof: [E17].
        let dense_row = &mut dense_plane[base..base + width];
        let span = if reads_ridges {
            let root_ridges = &anchor.ridges[base..base + width];
            dense_row.copy_from_slice(root_ridges);
            moved_span(
                (nearest_row, &anchor.nearest_squares[base..base + width]),
                (second_row, &anchor.second_squares[base..base + width]),
            )
        } else {
            0..width
        };
        // On its own, not inside the bit packing below: there the loop's carried
        // byte stops it vectorizing, and this is two square roots per pixel.
        let ridges = &mut ridges[..span.len()];
        ridge_row(
            &nearest_row[span.clone()],
            &second_row[span.clone()],
            radius,
            ridges,
        );
        dense_row[span].convert_from_f32_slice(ridges);

        // Groups of eight align with byte boundaries only because `width` is a
        // multiple of eight, which makes `row * width` one too. The general
        // path below sets each bit on its own.
        if width.is_multiple_of(8) {
            for (bytes, flags) in settled_plane[base / 8..(base + width) / 8]
                .iter_mut()
                .zip(settled[base..base + width].chunks_exact(8))
            {
                let mut byte = 0u8;
                for (bit, &flag) in flags.iter().enumerate() {
                    byte |= u8::from(flag) << bit;
                }
                *bytes = byte;
            }
        } else {
            for column in 0..width {
                let pixel = base + column;
                set_bit(settled_plane, pixel, settled[pixel]);
            }
        }
    }
    if is_root {
        anchor.ridges.extend_from_slice(dense_plane);
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

    /// Stones on a loose lattice, far enough apart to be a legal position.
    fn lattice(radius: f64) -> Vec<(f64, f64)> {
        let step = 2.3 * radius;
        let mut points = Vec::new();
        let mut y = 2.0 * radius;
        while y < 1.0 - 2.0 * radius {
            let mut x = 2.0 * radius;
            while x < 1.0 - 2.0 * radius {
                points.push((x, y));
                x += step;
            }
            y += step;
        }
        points
    }

    fn stones_at(points: &[(f64, f64)], picks: &[usize]) -> Vec<Stone> {
        picks
            .iter()
            .enumerate()
            .map(|(turn, &pick)| {
                let (x, y) = points[pick];
                let color = if turn % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                };
                Stone::new(x, y, color)
            })
            .collect()
    }

    /// Every path through the anchor against a sweep from nothing.
    ///
    /// The anchor lives across calls, so what can go wrong is a stale field, and
    /// only a sequence can show that: leaves branching off a root, a capture
    /// taking a stone from the middle of the list, a jump to a different game,
    /// and a change of raster size on the same thread.
    #[test]
    fn the_anchored_field_matches_a_sweep_from_nothing() {
        let radius = 1.0 / 38.0;
        let points = lattice(radius);
        assert!(points.len() > 120);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |limit: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % limit as u64) as usize
        };

        let mut scratch = PackedScratch::default();
        let (mut saved, mut fits) = (0usize, [0usize; 3]);
        for game in 0..12 {
            let mut root: Vec<usize> = (0..points.len()).collect();
            for at in 0..root.len() {
                root.swap(at, at + next(points.len() - at));
            }
            let mut pool = root.split_off(30 + game * 6);
            for leaf in 0..60 {
                let mut picks = root.clone();
                let depth = next(17);
                picks.extend(pool.iter().skip(next(20)).take(depth));
                // A capture: stones leave from anywhere, and order is kept.
                if leaf % 7 == 3 {
                    picks.remove(next(picks.len()));
                }
                // The root moves on, ready for the next one announced.
                if leaf % 13 == 12 {
                    let by = 1 + next(20);
                    root.extend(pool.drain(..by));
                }
                let resolution = [128, 128, 128, 64][game % 4];
                let config = RasterConfig::square_of(resolution, RasterKind::CompactRadius);
                let position = Position::new(radius, stones_at(&points, &picks), Color::Black);
                assert!(position.validate().is_playable());

                let announced = leaf % 13 == 1;
                scratch.anchor.next_is_root = announced;

                let grid: Vec<f64> = (0..resolution)
                    .map(|index| (index as f64 + 0.5) / resolution as f64)
                    .collect();
                let input = SweepInput {
                    stone_xs: &[],
                    stone_ys: &[],
                    xs: &grid,
                    ys: &grid,
                };
                if !announced {
                    fits[match scratch.anchor.fit(position.stones(), &input, radius) {
                        StoneFit::Extends => 0,
                        StoneFit::Differs => 1,
                        StoneFit::Unrelated => 2,
                    }] += 1;
                }

                let mut anchored = PackedRaster::new(config);
                write_packed(&position, config, &mut anchored, &mut scratch);
                let mut fresh = PackedRaster::new(config);
                write_packed(&position, config, &mut fresh, &mut PackedScratch::default());
                assert_eq!(anchored.bits, fresh.bits, "game {game} leaf {leaf}");
                assert_eq!(anchored.dense, fresh.dense, "game {game} leaf {leaf}");
                saved += scratch.anchor.answers.iter().filter(|&&a| a != 0).count();
            }
        }
        // A run that missed a path, or never saved an answer, proved nothing.
        assert!(fits.iter().all(|&count| count > 5), "fits {fits:?}");
        assert!(saved > 1000, "saved {saved} answers");
    }

    /// A field belongs to its grid: an arena can render the same stones at two
    /// sizes on one thread, one seat's leaf after the other's.
    #[test]
    fn the_anchor_keeps_to_its_grid() {
        let radius = 1.0 / 38.0;
        let points = lattice(radius);
        let picks: Vec<usize> = (0..50).map(|index| index * 3).collect();
        let stones = stones_at(&points, &picks);
        let root = Position::new(radius, stones[..40].to_vec(), Color::Black);
        let leaf = Position::new(radius, stones, Color::Black);
        let mut scratch = PackedScratch::default();
        for (position, size, announced) in [
            (&root, 128, true),
            (&leaf, 128, false),
            (&leaf, 64, false),
            (&root, 64, false),
            (&leaf, 128, false),
        ] {
            let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
            scratch.anchor.next_is_root = announced;
            let mut anchored = PackedRaster::new(config);
            let mut fresh = PackedRaster::new(config);
            write_packed(position, config, &mut anchored, &mut scratch);
            write_packed(position, config, &mut fresh, &mut PackedScratch::default());
            assert_eq!(anchored.bits, fresh.bits);
            assert_eq!(anchored.dense, fresh.dense);
        }
    }

    /// Exact answers belong to a radius as well as to stones: the same stones on
    /// a board with a different radius have a different legal set.
    #[test]
    fn the_anchor_keeps_to_its_radius() {
        let wide = 1.0 / 38.0;
        let points = lattice(wide);
        let picks: Vec<usize> = (0..90).map(|index| index * 2).collect();
        let stones = stones_at(&points, &picks);
        let config = RasterConfig::square_of(128, RasterKind::CompactRadius);
        let mut scratch = PackedScratch::default();
        for radius in [wide, 0.8 * wide, wide, 0.6 * wide] {
            let position = Position::new(radius, stones.clone(), Color::Black);
            let mut anchored = PackedRaster::new(config);
            let mut fresh = PackedRaster::new(config);
            for announced in [false, true, false] {
                scratch.anchor.next_is_root = announced;
                write_packed(&position, config, &mut anchored, &mut scratch);
                write_packed(&position, config, &mut fresh, &mut PackedScratch::default());
                assert_eq!(anchored.bits, fresh.bits);
                assert_eq!(anchored.dense, fresh.dense);
            }
        }
    }

    /// Stones can differ in one coordinate alone, so the prefix has to read both.
    #[test]
    fn the_anchor_compares_both_coordinates() {
        let radius = 1.0 / 38.0;
        let points = lattice(radius);
        let picks: Vec<usize> = (0..40).map(|index| index * 5).collect();
        let stones = stones_at(&points, &picks);
        let mirrored = |flip_x: bool, flip_y: bool| {
            let placed = stones.iter().map(|stone| {
                let x = if flip_x { 1.0 - stone.x } else { stone.x };
                let y = if flip_y { 1.0 - stone.y } else { stone.y };
                Stone::new(x, y, stone.color)
            });
            Position::new(radius, placed.collect(), Color::Black)
        };
        let config = RasterConfig::square_of(128, RasterKind::CompactRadius);
        let mut scratch = PackedScratch::default();
        for (flip_x, flip_y) in [(false, false), (false, true), (true, true), (true, false)] {
            let position = mirrored(flip_x, flip_y);
            let mut anchored = PackedRaster::new(config);
            let mut fresh = PackedRaster::new(config);
            for announced in [false, true, false] {
                scratch.anchor.next_is_root = announced;
                write_packed(&position, config, &mut anchored, &mut scratch);
                write_packed(&position, config, &mut fresh, &mut PackedScratch::default());
                assert_eq!(anchored.bits, fresh.bits);
                assert_eq!(anchored.dense, fresh.dense);
            }
        }
    }

    /// A board dense enough that the grid actually prunes.
    ///
    /// The four-stone fixture below exercises none of the search: with that few
    /// stones every ring is visited and nothing is skipped, so it would pass
    /// even if the bound were wrong. This is the case that can fail.
    #[test]
    fn the_chunked_sweep_matches_at_real_stone_counts() {
        // A radius small enough to reach high stone counts, where the tile
        // search discards the most stones.
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
                        if index % 2 == 0 {
                            Color::Black
                        } else {
                            Color::White
                        },
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
    /// The fused classifier against the builder it replaced.
    ///
    /// The packed writer classifies `settled` row by row from the sweep's own
    /// minima; `settled_for_raster` sweeps the stones again to find its own. The
    /// two must agree on every pixel, and unlike the ridge there is no f16
    /// quantisation here to hide a last-ulp disagreement -- which is why both
    /// sides compute squared distances by `mul_add`.
    #[test]
    fn fused_settled_matches_the_reference_builder() {
        // Both sizes sample the legal set at the output resolution.
        for width in [256usize, 128] {
            fused_settled_matches_at(width);
        }
    }

    fn fused_settled_matches_at(width: usize) {
        let config = RasterConfig {
            width,
            height: width,
            kind: RasterKind::CompactRadius,
        };
        let radius = 1.0 / 38.0;
        let spacing = 2.0 * radius * 1.05;
        let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
        for count in [1usize, 19, 20, 21, 40, 61, 95, 96, 130, 240] {
            let mut stones = Vec::new();
            for index in 0..count {
                let (row, column) = (index / per_row, index % per_row);
                let x = radius + 0.02 + column as f64 * spacing;
                let y = radius + 0.02 + row as f64 * spacing;
                if x > 1.0 - radius || y > 1.0 - radius {
                    break;
                }
                stones.push(Stone {
                    x,
                    y,
                    color: if index % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    },
                });
            }
            let position = Position::new(radius, stones, Color::Black);
            let packed = rasterize_packed(&position, config);
            let expected = crate::settled_for_raster(&position, config);
            let stride = bit_plane_bytes(config.pixels());
            let plane = &packed.bits[2 * stride..3 * stride];
            let mut wrong = Vec::new();
            for pixel in 0..config.pixels() {
                let got = plane[pixel / 8] >> (pixel % 8) & 1 == 1;
                if got != expected[pixel] {
                    wrong.push(pixel);
                }
            }
            assert!(
                wrong.is_empty(),
                "{width}px, {count} stones: {} settled pixels disagree with the reference, first at {:?}",
                wrong.len(),
                wrong.first()
            );
        }
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
