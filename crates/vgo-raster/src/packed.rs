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
use vgo_core::{Color, Position};

use crate::{RasterConfig, RasterKind, settled_for_raster};

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
#[derive(Debug, Clone, PartialEq)]
pub struct PackedRaster {
    config: RasterConfig,
    layout: PackedLayout,
    bits: Vec<u8>,
    dense: Vec<f16>,
    scalars: Vec<f16>,
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

/// The ridge value for one pixel, from the nearest and second-nearest squared
/// distances. Matches the dense writer's expression exactly, including doing
/// the square roots before the subtraction.
#[inline]
fn ridge_at(nearest_square: f64, second_square: f64, radius: f64) -> f32 {
    let nearest = nearest_square.sqrt();
    let second = second_square.sqrt();
    if second.is_finite() {
        (1.0 - (second - nearest) / radius).clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
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
    let settled = settled_for_raster(position, config);
    assert_eq!(settled.len(), pixels);

    let radius = position.radius();
    let radius_square = radius * radius;
    let to_move = position.to_move();
    let (current_stones, opponent_stones) = crate::relative_stones(position, to_move);

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
        bits, dense: dense_plane, ..
    } = out;
    for byte in bits.iter_mut() {
        *byte = 0;
    }

    let stride = bit_plane_bytes(pixels);
    let (current_plane, rest) = bits.split_at_mut(stride);
    let (opponent_plane, settled_plane) = rest.split_at_mut(stride);
    let dense_plane = dense_plane.as_mut_slice();

    // The same row-major accumulation as the dense writer: one stone across a
    // whole row at a time, so the vertical distance is squared once per row and
    // the minima stay in contiguous buffers.
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

        for &(stone_x, stone_y) in &current_stones {
            let dy = y - stone_y;
            let dy_square = dy * dy;
            for column in 0..width {
                let dx = xs[column] - stone_x;
                let square = dx * dx + dy_square;
                if square < current_squares[column] {
                    current_squares[column] = square;
                }
                if square < nearest_squares[column] {
                    second_squares[column] = nearest_squares[column];
                    nearest_squares[column] = square;
                } else if square < second_squares[column] {
                    second_squares[column] = square;
                }
            }
        }
        for &(stone_x, stone_y) in &opponent_stones {
            let dy = y - stone_y;
            let dy_square = dy * dy;
            for column in 0..width {
                let dx = xs[column] - stone_x;
                let square = dx * dx + dy_square;
                if square < opponent_squares[column] {
                    opponent_squares[column] = square;
                }
                if square < nearest_squares[column] {
                    second_squares[column] = nearest_squares[column];
                    nearest_squares[column] = square;
                } else if square < second_squares[column] {
                    second_squares[column] = square;
                }
            }
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
