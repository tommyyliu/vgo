#![forbid(unsafe_code)]

use vgo_core::{Color, Position, SettledRegion, legal_set_vertices};


pub mod edt;
pub mod packed;
mod policy;
pub use policy::DensePolicy;
pub use edt::{
    settled_mask_by_bounded_distance,
    settled_mask_by_distance,
};

/// Stone count above which the distance-transform settled mask is worth its
/// fixed cost. Measured crossover sits between 14 and 28 stones.
const DISTANCE_SETTLED_MINIMUM_STONES: usize = 20;

/// Whether `settled` takes the distance-transform path, and at what oversample,
/// or `None` for the per-stone solve.
///
/// The transform samples the legal set at the output resolution. It used to need
/// six cells per stone radius -- below that, legal slivers fell between samples
/// and the region beside them read as settled -- and coarse rasters oversampled
/// the legality grid to get there. Marking the legal-set vertices on the grid
/// (see `sampled_legal_set_into`) catches every sliver at any resolution: on 103
/// real 1/38 positions at 128, 1x and 3x both disagreed with the definition on 4
/// of 1.56M pixels, and 1x rasterizes 17% faster.
pub(crate) fn settled_oversample(position: &Position, _config: RasterConfig) -> Option<usize> {
    (position.stones().len() >= DISTANCE_SETTLED_MINIMUM_STONES).then_some(1)
}

pub const CHANNEL_COUNT: usize = 12;


/// Indices into [`CHANNELS`] that [`RasterKind::Compact`] keeps.
pub(crate) const COMPACT_CHANNELS: [usize; 5] = [
    0,  // current_stones
    1,  // opponent_stones
    6,  // voronoi_ridge
    10, // settled
    11, // komi
];

/// Indices [`RasterKind::CompactRadius`] keeps: [`COMPACT_CHANNELS`], then
/// whether the previous move was a pass, then the radius.
///
/// Without the pass plane a net cannot tell that passing now would end the
/// game -- it can neither pass to close out a win nor see that passing while
/// behind hands over the result.
///
/// The board is the unit square whatever the stone size, so radius *is* the
/// board size: `voronoigo.com` plays 18, 26 and 38 units across, which are
/// radii of 1/18, 1/26 and 1/38 here. Nothing else in the layout carries it. An
/// empty board renders identically at every radius -- no stones, so no ridge,
/// no settled region, and the two scalars say nothing about scale -- and a net
/// asked to open on a board whose size it cannot see is guessing which game it
/// is playing.
///
/// The plane holds `2r`, the stone diameter, matching what the semantic
/// rasterizer writes at index 8. That is in `[0, 1]` for every playable radius
/// and reads directly as "how much of the board does one stone span".
pub(crate) const COMPACT_RADIUS_CHANNELS: [usize; 7] = [
    0,  // current_stones
    1,  // opponent_stones
    6,  // voronoi_ridge
    10, // settled
    11, // komi
    9,  // previous_pass
    8,  // radius
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelScale {
    Unit,
    Signed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub scale: ChannelScale,
}

pub const CHANNELS: [ChannelSpec; CHANNEL_COUNT] = [
    ChannelSpec {
        name: "current_stones",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "opponent_stones",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "current_voronoi",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "opponent_voronoi",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "current_distance",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "opponent_distance",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "voronoi_ridge",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "legal_clearance",
        scale: ChannelScale::Signed,
    },
    ChannelSpec {
        name: "radius",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "previous_pass",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "settled",
        scale: ChannelScale::Unit,
    },
    ChannelSpec {
        name: "komi",
        scale: ChannelScale::Signed,
    },
];

/// Which channel layout a raster carries.
///
/// `Semantic` is the twelve engineered channels; the compact layouts are subsets
/// of them. A model trained on one cannot read another, so this belongs to a
/// run's identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RasterKind {
    #[default]
    Semantic,
    /// The four channels an ablation preferred, plus komi.
    ///
    /// Measured over 24 epochs on 30720 samples: current_stones,
    /// opponent_stones, voronoi_ridge and settled reached policy CE 2.8177 and
    /// 35.2% argmax agreement against all eleven channels' 2.8308 and 34.4%,
    /// and the ten-channel set's 2.8425 and 32.2%. The spread is under a
    /// percent and one seed, so this is a preference rather than a finding --
    /// but fewer channels is also less memory, and the replay window is what
    /// the run is short of.
    ///
    /// Komi joins them because a net that cannot see what it must win by
    /// cannot evaluate a position.
    Compact,
    /// [`Compact`](Self::Compact) plus the previous-pass and radius planes:
    /// the layout every model trains on. See
    /// [`COMPACT_RADIUS_CHANNELS`] for why the plane is needed rather than
    /// inferable.
    CompactRadius,
}

impl RasterKind {
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Semantic => CHANNEL_COUNT,
            Self::Compact => COMPACT_CHANNELS.len(),
            Self::CompactRadius => COMPACT_RADIUS_CHANNELS.len(),
        }
    }

    /// Which entries of [`CHANNELS`] this layout writes, in order.
    #[must_use]
    pub const fn indices(self) -> &'static [usize] {
        match self {
            Self::Semantic => &[],
            Self::Compact => &COMPACT_CHANNELS,
            Self::CompactRadius => &COMPACT_RADIUS_CHANNELS,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Compact => "compact",
            Self::CompactRadius => "compact-radius",
        }
    }
}

impl std::str::FromStr for RasterKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "semantic" => Ok(Self::Semantic),
            "compact" => Ok(Self::Compact),
            "compact-radius" => Ok(Self::CompactRadius),
            _ => Err(format!("unsupported raster kind: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RasterConfig {
    pub width: usize,
    pub height: usize,
    /// Which channel layout to write. Carried here because every consumer that
    /// needs the raster's shape already has the config, so nothing else has to
    /// be threaded alongside it.
    pub kind: RasterKind,
}

impl RasterConfig {
    #[must_use]
    pub const fn square(size: usize) -> Self {
        Self {
            width: size,
            height: size,
            kind: RasterKind::Semantic,
        }
    }

    #[must_use]
    pub const fn square_of(size: usize, kind: RasterKind) -> Self {
        Self {
            width: size,
            height: size,
            kind,
        }
    }

    /// Channels a raster written with this config carries.
    #[must_use]
    pub const fn channels(self) -> usize {
        self.kind.channels()
    }

    #[must_use]
    pub const fn pixels(self) -> usize {
        self.width * self.height
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticRaster {
    config: RasterConfig,
    data: Vec<f32>,
}

impl SemanticRaster {
    /// Wraps caller-owned channel-first data.
    ///
    /// For callers that produced the planes themselves — the CUDA path computes
    /// `settled` on the device and fills the rest here, so it has the buffer
    /// before it has a `SemanticRaster`.
    #[must_use]
    pub fn from_parts(config: RasterConfig, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), config.channels() * config.pixels());
        Self { config, data }
    }

    #[must_use]
    pub const fn config(&self) -> RasterConfig {
        self.config
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn channel(&self, channel: usize) -> &[f32] {
        let pixels = self.config.pixels();
        &self.data[channel * pixels..(channel + 1) * pixels]
    }

    #[must_use]
    pub fn at(&self, channel: usize, row: usize, column: usize) -> f32 {
        self.channel(channel)[row * self.config.width + column]
    }

}

/// Rasterize a position into whichever layout `config.kind` names.
///
/// Callers hold a `SemanticRaster` either way and never index channels by
/// meaning, so the two layouts are interchangeable everywhere downstream of
/// this call: the search, the broker, and the shard writer all treat the data
/// as an opaque block of `config.channels()` planes.
#[must_use]
pub fn rasterize(position: &Position, config: RasterConfig) -> SemanticRaster {
    let mut data = vec![0.0_f32; config.channels() * config.pixels()];
    rasterize_any_into(position, config, &mut data);
    SemanticRaster { config, data }
}

/// Writes whichever layout `config.kind` names into caller-owned storage.
pub fn rasterize_any_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    match config.kind {
        RasterKind::Semantic => rasterize_into(position, config, data),
        RasterKind::Compact => rasterize_compact_into(position, config, data),
        RasterKind::CompactRadius => rasterize_compact_radius_into(position, config, data),
    }
}

/// Writes [`RasterKind::CompactRadius`]: the compact planes plus pass and radius.
///
/// Both trailing planes are constant over the board, so they are filled rather
/// than rendered. The first five come from the shared compact writer.
pub fn rasterize_compact_radius_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    let pixels = config.pixels();
    assert_eq!(data.len(), COMPACT_RADIUS_CHANNELS.len() * pixels);
    let predicate = settled_for_raster(position, config);
    let compact = RasterConfig {
        kind: RasterKind::Compact,
        ..config
    };
    let (head, rest) = data.split_at_mut(COMPACT_CHANNELS.len() * pixels);
    rasterize_compact_with_predicate_into(position, compact, &predicate, head);
    let (pass, radius) = rest.split_at_mut(pixels);
    pass.fill(f32::from(position.consecutive_passes() > 0));
    radius.fill((2.0 * position.radius()) as f32);
}

/// Writes the [`RasterKind::Compact`] subset.
///
/// This shares the semantic raster's geometry helpers but writes only the five
/// requested planes. In particular, it does not allocate, render, and copy a
/// twelve-plane temporary for every inference position.
/// The `settled` channel, from whichever implementation is configured.
///
/// Every writer goes through here so the compact and semantic rasters cannot
/// disagree on it — which is exactly what broke when only the compact writer
/// was switched: `compact_is_a_subset_of_the_semantic_raster` failed, correctly.
#[must_use]
pub fn settled_for_raster(position: &Position, config: RasterConfig) -> Vec<bool> {
    // Dispatch on stone count. The distance-transform form pays a fixed
    // O(pixels) cost and wins only once the per-stone solve's O(n^2) exceeds it:
    // measured 0.5x at 14 stones, 2.8x at 28, 7.2x at 52. Real shards are not
    // all late-game -- the corpus this was tuned against runs min 0, mean 26.2
    // stones -- so always taking it gave back a third of the gain on early
    // positions.
    //
    // This used to sit behind a `distance-settled` feature, off by default,
    // "pending an A/B on real shards". The A/B never happened and the flag
    // never moved, so every run since paid the quadratic path. Two things
    // settle it without one:
    //
    //   * The distance-transform form is *closer* to the definition, not
    //     merely faster. `bounded_distance_agrees_with_the_definition` pins it
    //     at zero wrong pixels; the per-stone solve walks a contour at 1/128
    //     tolerance and is wrong on one or two of 16384.
    //   * A build-time flag that changes a network input is worse than a
    //     config field, because nothing records it. A resumed run could render
    //     different inputs than it trained on and the identity check would see
    //     nothing.
    //
    // So the fast path is simply the path now, and there is no flag to forget.
    match settled_oversample(position, config) {
        Some(scale) => edt::settled_mask_by_bounded_distance(position, config, scale).0,
        None => settled_mask(position, config),
    }
}

pub(crate) fn settled_for_raster_into(
    position: &Position,
    config: RasterConfig,
    scratch: &mut edt::EdtScratch,
    output: &mut Vec<bool>,
) {
    if let Some(scale) = settled_oversample(position, config) {
        edt::settled_mask_by_bounded_distance_into(position, config, scale, scratch, output);
    } else {
        output.clear();
        output.extend_from_slice(&settled_for_raster(position, config));
    }
}

pub(crate) fn rasterize_compact_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    let settled = settled_for_raster(position, config);
    rasterize_compact_with_predicate_into(position, config, &settled, data);
}

/// The five compact planes, with slot 3 -- the capture predicate -- supplied
/// rather than computed.
///
/// Split out because that plane is where the cost is: 92% of this function at
/// the median stone count under the per-stone geometric solve, and still 60-80%
/// under the distance transform, while the other four are per-pixel work over
/// the stone list.
///
/// The predicate is `settled` for this repository's rules and the dead zone for
/// the official ones. Nothing below cares which: both are a boolean per pixel
/// saying whether this point is beyond further contest, and the layouts that
/// name one or the other put it in the same slot deliberately.
///
/// The mask must be `config.pixels()` long and indexed row-major, exactly as
/// [`settled_mask`] returns it.
pub(crate) fn rasterize_compact_with_predicate_into(
    position: &Position,
    config: RasterConfig,
    settled: &[bool],
    data: &mut [f32],
) {
    assert!(config.width > 0 && config.height > 0);
    // Caller invariant, not this function's business, and an O(n^2)
    // sweep per rasterization if checked in release. See `game::place`.
    debug_assert!(position.validate().is_playable());
    let pixels = config.pixels();
    assert_eq!(data.len(), COMPACT_CHANNELS.len() * pixels);
    assert_eq!(settled.len(), pixels);
    let radius = position.radius();
    let radius_square = radius * radius;
    let to_move = position.to_move();
    let mover_komi = match to_move {
        Color::Black => position.komi() as f32,
        Color::White => -position.komi() as f32,
    };
    let (current_stones, opponent_stones) = relative_stones(position, to_move);

    // Komi is constant over the board, so write its plane once rather than in
    // the pixel loop below.
    data[4 * pixels..5 * pixels].fill(mover_komi);

    // Walk one stone across a whole raster row at a time. The old pixel-major
    // loop reread both stone arrays for every pixel and recomputed the same
    // vertical distance once per column. Row-major accumulation hoists that
    // square, keeps the four minima in contiguous buffers, and gives LLVM a
    // simple inner loop to vectorize. Each pixel still sees current stones and
    // then opponent stones in their original order, with the same arithmetic
    // and comparisons, so the resulting planes remain bit-for-bit identical to
    // the semantic writer.
    let width = config.width;
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

        for column in 0..width {
            let pixel = row * width + column;
            let current_square = current_squares[column];
            let opponent_square = opponent_squares[column];
            let nearest_square = nearest_squares[column];
            let second_square = second_squares[column];
            let nearest = nearest_square.sqrt();
            let second = second_square.sqrt();

            // Squaring is monotonic for nonnegative distances, so the two
            // stone-disc planes do not need a square root per pixel. The ridge
            // still needs the actual nearest and second-nearest distances.
            data[pixel] = f32::from(current_square <= radius_square);
            data[pixels + pixel] = f32::from(opponent_square <= radius_square);
            data[2 * pixels + pixel] = if second.is_finite() {
                (1.0 - (second - nearest) / radius).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            data[3 * pixels + pixel] = f32::from(settled[pixel]);
        }
    }
}

/// Writes a semantic raster into caller-owned contiguous channel-first storage.
///
/// Reusable or pinned inference buffers can use this entry point to avoid an
/// intermediate per-position allocation and host-side gather.
pub(crate) fn rasterize_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    assert!(config.width > 0 && config.height > 0);
    // Caller invariant, not this function's business, and an O(n^2)
    // sweep per rasterization if checked in release. See `game::place`.
    debug_assert!(position.validate().is_playable());
    let pixels = config.pixels();
    assert_eq!(data.len(), CHANNEL_COUNT * pixels);
    let radius = position.radius();
    let distance_scale = (4.0 * radius).max(f64::EPSILON);

    // The inner loop tracks *squared* distances. Squaring is monotonic on
    // nonnegative reals, so every minimum and ordering below is unchanged, but it
    // replaces one `hypot` per (pixel, stone) pair with a multiply-add. Only the
    // four surviving distances need a square root, per pixel rather than per
    // stone. At 96x96 with 30 stones that is 276k transcendental calls traded for
    // 37k -- this loop was measured at 66% of all self-play CPU time.
    let to_move = position.to_move();
    // What the side to move must win by, from its own seat. Every other channel
    // is mover-relative and komi has to be too: the same position is a win for
    // one seat and a loss for the other at the same komi, so feeding Black's
    // komi alongside White's stones states the wrong target. KataGo signs it
    // the same way -- `selfKomi` in nninputs.cpp is relative to the player to
    // move.
    //
    // Scoring is `black - white - komi > 0` for a Black win, so a positive komi
    // is a margin Black must overcome and one White may fall short by. Voronoi
    // area totals 1.0, so this is already order one and needs no scaling.
    let mover_komi = match to_move {
        Color::Black => position.komi() as f32,
        Color::White => -position.komi() as f32,
    };
    // Splitting by colour once hoists the per-stone colour comparison out of the
    // pixel loop entirely.
    // The settled region, as a mask built once rather than per pixel.
    //
    // Each stone's boundary is solved once as a contour and filled, which is
    // what the client does. Testing every pixel against every stone's radial
    // solve instead costs 573k solves at 35 stones against the contour's ~20k
    // ray evaluations -- 29x more work, and it measured 239 ms against the
    // whole rest of the raster's 0.5 ms.
    // Same source as the compact writer, or the two disagree on channel 10
    // and `compact_is_a_subset_of_the_semantic_raster` fails -- correctly.
    let settled_mask = settled_for_raster(position, config);
    let (current_stones, opponent_stones) = relative_stones(position, to_move);

    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        for column in 0..config.width {
            let x = (column as f64 + 0.5) / config.width as f64;
            let pixel = row * config.width + column;
            let (current_square, opponent_square, nearest_square, second_square) =
                squared_distances(x, y, &current_stones, &opponent_stones);

            let current_distance = current_square.sqrt();
            let opponent_distance = opponent_square.sqrt();
            let nearest = nearest_square.sqrt();
            let second = second_square.sqrt();

            set(data, pixels, 0, pixel, inside(current_distance, radius));
            set(data, pixels, 1, pixel, inside(opponent_distance, radius));
            let (current_area, opponent_area) = ownership(current_distance, opponent_distance);
            set(data, pixels, 2, pixel, current_area);
            set(data, pixels, 3, pixel, opponent_area);
            set(
                data,
                pixels,
                4,
                pixel,
                normalized_distance(current_distance, distance_scale),
            );
            set(
                data,
                pixels,
                5,
                pixel,
                normalized_distance(opponent_distance, distance_scale),
            );
            let ridge = if second.is_finite() {
                (1.0 - (second - nearest) / radius).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            set(data, pixels, 6, pixel, ridge);

            // Settled is the union over stones (A15), attributed to whichever
            // side owns the nearest stone -- the same ownership rule the
            // voronoi channels use.
            set(data, pixels, 10, pixel, f32::from(settled_mask[pixel]));
            set(data, pixels, 11, pixel, mover_komi);

            let board_clearance = (x - radius)
                .min(1.0 - radius - x)
                .min(y - radius)
                .min(1.0 - radius - y);
            let stone_clearance = if nearest.is_finite() {
                nearest - 2.0 * radius
            } else {
                f64::INFINITY
            };
            let legal_clearance = board_clearance.min(stone_clearance);
            set(
                data,
                pixels,
                7,
                pixel,
                (legal_clearance / radius).clamp(-1.0, 1.0) as f32,
            );
            set(data, pixels, 8, pixel, (2.0 * radius) as f32);
            set(
                data,
                pixels,
                9,
                pixel,
                f32::from(position.consecutive_passes() > 0),
            );
        }
    }
}

/// The `settled` channel as a mask, for callers that render the other channels
/// themselves.
///
/// Public because the GPU path needs it: `compact.wgsl` computes the four
/// per-pixel channels from the stone list, but this one is per-stone contour
/// geometry scanline-filled, which is not pixel-shader work. The host computes
/// it and uploads 64 KB rather than the 327 KB whole tensor.
pub fn settled_mask(position: &Position, config: RasterConfig) -> Vec<bool> {
    let pixels = config.pixels();
    let stones = position.stones();
    let known_vertices = legal_set_vertices(position);
    let mut settled = vec![false; pixels];
    let mut contour = Vec::new();
    let mut crossings: Vec<f64> = Vec::with_capacity(16);
    for index in 0..stones.len() {
        let region = SettledRegion::new(position, index, &known_vertices);
        // One pixel: the contour is only used to classify pixel centres, and
        // finer chord detail cannot be represented by the output mask. This is
        // far cheaper than the 2e-5 the client needs for a zoomable vector.
        region.contour_within_into(1.0 / config.width.max(config.height) as f64, &mut contour);
        if contour.len() < 3 {
            continue;
        }
        // Only the rows the contour spans need testing.
        let (mut low_y, mut high_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for point in &contour {
            low_y = low_y.min(point.y);
            high_y = high_y.max(point.y);
        }
        let first_row = ((low_y * config.height as f64 - 0.5).floor().max(0.0)) as usize;
        let last_row =
            ((high_y * config.height as f64 - 0.5).ceil() as usize).min(config.height - 1);
        // Scanline fill computes a row's crossings once instead of once per
        // pixel. The same buffer is reused for every row and every stone.
        for row in first_row..=last_row {
            let y = (row as f64 + 0.5) / config.height as f64;
            crossings.clear();
            let mut previous = contour[contour.len() - 1];
            for &current in &contour {
                if (current.y > y) != (previous.y > y) {
                    let t = (y - current.y) / (previous.y - current.y);
                    crossings.push(current.x + t * (previous.x - current.x));
                }
                previous = current;
            }
            if crossings.is_empty() {
                continue;
            }
            crossings.sort_by(f64::total_cmp);
            // Star-shaped loops are simple, so spans pair up in order.
            for span in crossings.chunks_exact(2) {
                let from = ((span[0] * config.width as f64 - 0.5).ceil()).max(0.0) as usize;
                let to = ((span[1] * config.width as f64 - 0.5).floor()).max(-1.0);
                if to < 0.0 {
                    continue;
                }
                let to = (to as usize).min(config.width - 1);
                for column in from..=to {
                    settled[row * config.width + column] = true;
                }
            }
        }
    }
    settled
}

pub(crate) fn relative_stones(
    position: &Position,
    to_move: Color,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let stones = position.stones();
    let mut current = Vec::with_capacity(stones.len());
    let mut opponent = Vec::with_capacity(stones.len());
    for stone in stones {
        if stone.color == to_move {
            current.push((stone.x, stone.y));
        } else {
            opponent.push((stone.x, stone.y));
        }
    }
    (current, opponent)
}

#[inline]
fn squared_distances(
    x: f64,
    y: f64,
    current_stones: &[(f64, f64)],
    opponent_stones: &[(f64, f64)],
) -> (f64, f64, f64, f64) {
    let mut current_square = f64::INFINITY;
    let mut opponent_square = f64::INFINITY;
    let mut nearest_square = f64::INFINITY;
    let mut second_square = f64::INFINITY;

    for &(sx, sy) in current_stones {
        let dx = x - sx;
        let dy = y - sy;
        let square = dx * dx + dy * dy;
        if square < current_square {
            current_square = square;
        }
        if square < nearest_square {
            second_square = nearest_square;
            nearest_square = square;
        } else if square < second_square {
            second_square = square;
        }
    }
    for &(sx, sy) in opponent_stones {
        let dx = x - sx;
        let dy = y - sy;
        let square = dx * dx + dy * dy;
        if square < opponent_square {
            opponent_square = square;
        }
        if square < nearest_square {
            second_square = nearest_square;
            nearest_square = square;
        } else if square < second_square {
            second_square = square;
        }
    }
    (
        current_square,
        opponent_square,
        nearest_square,
        second_square,
    )
}

#[must_use]
pub fn action_pixel(x: f64, y: f64, config: RasterConfig) -> usize {
    let column = (x * config.width as f64).floor() as usize;
    let row = (y * config.height as f64).floor() as usize;
    row.min(config.height - 1) * config.width + column.min(config.width - 1)
}

fn set(data: &mut [f32], pixels: usize, channel: usize, pixel: usize, value: f32) {
    data[channel * pixels + pixel] = value;
}

fn inside(distance: f64, radius: f64) -> f32 {
    f32::from(distance <= radius)
}

fn ownership(current: f64, opponent: f64) -> (f32, f32) {
    if !current.is_finite() && !opponent.is_finite() {
        (0.0, 0.0)
    } else if current < opponent {
        (1.0, 0.0)
    } else if opponent < current {
        (0.0, 1.0)
    } else {
        (0.5, 0.5)
    }
}

fn normalized_distance(distance: f64, scale: f64) -> f32 {
    if distance.is_finite() {
        (distance / scale).clamp(0.0, 1.0) as f32
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position, Stone};

    use super::{
        CHANNEL_COUNT, CHANNELS, COMPACT_CHANNELS, RasterConfig, RasterKind, action_pixel,
        rasterize, rasterize_any_into, rasterize_into,
    };

    /// The pre-optimization formulation: one `hypot` per (pixel, stone) pair.
    /// `rasterize_into` now accumulates squared distances and takes four square
    /// roots per pixel instead; squaring is monotonic so every min and ordering
    /// is preserved. This reference pins that equivalence.
    fn hypot_reference(position: &Position, config: RasterConfig) -> Vec<f32> {
        let vertices = vgo_core::legal_set_vertices(position);
        let pixels = config.pixels();
        let mut data = vec![0.0f32; CHANNEL_COUNT * pixels];
        let radius = position.radius();
        let scale = (4.0 * radius).max(f64::EPSILON);
        for row in 0..config.height {
            let y = (row as f64 + 0.5) / config.height as f64;
            for column in 0..config.width {
                let x = (column as f64 + 0.5) / config.width as f64;
                let pixel = row * config.width + column;
                let (mut current, mut opponent) = (f64::INFINITY, f64::INFINITY);
                let (mut nearest, mut second) = (f64::INFINITY, f64::INFINITY);
                for stone in position.stones() {
                    let distance = (x - stone.x).hypot(y - stone.y);
                    if stone.color == position.to_move() {
                        current = current.min(distance);
                    } else {
                        opponent = opponent.min(distance);
                    }
                    if distance < nearest {
                        second = nearest;
                        nearest = distance;
                    } else if distance < second {
                        second = distance;
                    }
                }
                data[pixel] = super::inside(current, radius);
                data[pixels + pixel] = super::inside(opponent, radius);
                let (owned, taken) = super::ownership(current, opponent);
                data[2 * pixels + pixel] = owned;
                data[3 * pixels + pixel] = taken;
                data[4 * pixels + pixel] = super::normalized_distance(current, scale);
                data[5 * pixels + pixel] = super::normalized_distance(opponent, scale);
                data[6 * pixels + pixel] = if second.is_finite() {
                    (1.0 - (second - nearest) / radius).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                let board = (x - radius)
                    .min(1.0 - radius - x)
                    .min(y - radius)
                    .min(1.0 - radius - y);
                let clear = if nearest.is_finite() {
                    nearest - 2.0 * radius
                } else {
                    f64::INFINITY
                };
                data[7 * pixels + pixel] = (board.min(clear) / radius).clamp(-1.0, 1.0) as f32;
                data[8 * pixels + pixel] = (2.0 * radius) as f32;
                data[9 * pixels + pixel] = f32::from(position.consecutive_passes() > 0);
                // The definition itself, not the radial solve: a point is
                // settled when some stone is at least as near as the legal set.
                let free = vgo_core::distance_to_legal_set(
                    position,
                    vgo_core::Point::new(x, y),
                    Some(&vertices),
                );
                let settled = position
                    .stones()
                    .iter()
                    .any(|s| (x - s.x).hypot(y - s.y) <= free);
                data[10 * pixels + pixel] = f32::from(settled);
                data[11 * pixels + pixel] = match position.to_move() {
                    Color::Black => position.komi() as f32,
                    Color::White => -position.komi() as f32,
                };
            }
        }
        data
    }

    fn scattered_position(stones: usize) -> Position {
        let radius = 1.0 / 18.0;
        let mut placed: Vec<Stone> = Vec::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut attempts = 0;
        while placed.len() < stones && attempts < 100_000 {
            attempts += 1;
            let x = radius + (1.0 - 2.0 * radius) * next();
            let y = radius + (1.0 - 2.0 * radius) * next();
            let color = if placed.len() % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            let probe = Position::new(radius, placed.clone(), Color::Black);
            if vgo_core::is_legal_placement(&probe, x, y) {
                placed.push(Stone { x, y, color });
            }
        }
        Position::new(radius, placed, Color::Black)
    }

    /// The squared-distance rewrite must not change the raster. Exact equality is
    /// the bar: `hypot` and `sqrt(dx*dx + dy*dy)` can differ by an ulp, which on a
    /// pixel exactly equidistant from two stones could flip Voronoi ownership
    /// between a tie and a winner. Real positions do not manufacture such ties;
    /// a symmetric lattice does, so this uses scattered placements.
    #[test]
    fn squared_distance_raster_matches_the_hypot_formulation() {
        for stones in [0usize, 1, 5, 17, 40] {
            let position = scattered_position(stones);
            let config = RasterConfig::square(48);
            let produced = rasterize(&position, config);
            let expected = hypot_reference(&position, config);
            // Channels 0-9 must match bit for bit.
            let plain = 10 * config.pixels();
            assert_eq!(
                &produced.data()[..plain],
                &expected[..plain],
                "raster diverged from the hypot reference at {stones} stones"
            );
            // The settled channels are filled from a contour subdivided to a
            // third of a pixel, so a boundary pixel may land either side of the
            // exact predicate. Bound that rather than requiring equality.
            let differing = produced.data()[plain..]
                .iter()
                .zip(&expected[plain..])
                .filter(|(a, b)| a != b)
                .count();
            let share = differing as f64 / config.pixels() as f64;
            // Measured 0.087% at 40 stones; this leaves room for a denser
            // board without admitting a real regression.
            assert!(
                share < 0.002,
                "settled channel differs on {differing} of {} values \
                 ({:.2}%) at {stones} stones",
                config.pixels(),
                100.0 * share
            );
        }
    }

    /// Channel 7's sign is the legality predicate the training mask is built
    /// from, so it must agree with the exact simulator, not merely with the old
    /// floating-point formulation.
    #[test]
    fn legal_clearance_sign_agrees_with_the_exact_predicate() {
        let position = scattered_position(12);
        let config = RasterConfig::square(48);
        let raster = rasterize(&position, config);
        let pixels = config.pixels();
        for row in 0..config.height {
            for column in 0..config.width {
                let pixel = row * config.width + column;
                let x = (column as f64 + 0.5) / config.width as f64;
                let y = (row as f64 + 0.5) / config.height as f64;
                let clearance = raster.data()[7 * pixels + pixel];
                let legal = vgo_core::is_legal_placement(&position, x, y);
                if clearance > 0.02 {
                    assert!(legal, "positive clearance at ({x}, {y}) must be legal");
                } else if clearance < -0.02 {
                    assert!(!legal, "negative clearance at ({x}, {y}) must be illegal");
                }
            }
        }
    }

    #[test]
    #[ignore = "timing"]
    fn measure_rasterize_cost() {
        for stones in [8usize, 20, 35] {
            let position = scattered_position(stones);
            let config = RasterConfig::square(128);
            let mut data = vec![0.0_f32; CHANNEL_COUNT * config.pixels()];
            for _ in 0..3 {
                rasterize_into(&position, config, &mut data);
            }
            let started = std::time::Instant::now();
            let runs = 20;
            for _ in 0..runs {
                rasterize_into(&position, config, &mut data);
            }
            println!(
                "  {stones:2} stones: {:.3} ms per 128x128 raster",
                started.elapsed().as_secs_f64() / f64::from(runs) * 1000.0
            );
        }
    }

    /// The komi channel states what the side to move must win by.
    ///
    /// The sign is silent if wrong -- a net simply learns the opposite of the
    /// truth -- so it is pinned against the scoring rule rather than described.
    #[test]
    fn komi_channel_is_relative_to_the_side_to_move() {
        let komi = 0.18;
        let stones = vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.75, 0.75, Color::White),
        ];
        let config = RasterConfig::square(16);
        let pixels = config.pixels();

        let black = Position::new(0.1, stones.clone(), Color::Black).with_komi(komi);
        let white = Position::new(0.1, stones, Color::White).with_komi(komi);
        let from_black = rasterize(&black, config);
        let from_white = rasterize(&white, config);

        // Scoring is `black - white - komi > 0`, so komi is a margin Black must
        // overcome and one White may fall short by.
        assert!(
            (from_black.data()[11 * pixels] - komi as f32).abs() < 1.0e-6,
            "Black must see the komi it has to overcome, got {}",
            from_black.data()[11 * pixels]
        );
        assert!(
            (from_white.data()[11 * pixels] + komi as f32).abs() < 1.0e-6,
            "White must see the komi it receives, got {}",
            from_white.data()[11 * pixels]
        );
        // Constant over the board, like radius.
        for pixel in 0..pixels {
            assert_eq!(
                from_black.data()[11 * pixels + pixel],
                from_black.data()[11 * pixels]
            );
        }
    }

    #[test]
    fn compact_is_a_subset_of_the_semantic_raster() {
        for (width, height) in [(48, 48), (63, 47), (128, 128)] {
            let full = RasterConfig {
                width,
                height,
                kind: RasterKind::Semantic,
            };
            let compact = RasterConfig {
                width,
                height,
                kind: RasterKind::Compact,
            };
            assert_eq!(compact.channels(), COMPACT_CHANNELS.len());
            let pixels = full.pixels();

            for stones in [0, 1, 12, 40] {
                let fixture = scattered_position(stones);
                for to_move in [Color::Black, Color::White] {
                    let position =
                        Position::new(fixture.radius(), fixture.stones().to_vec(), to_move)
                            .with_komi(0.15);
                    let whole = rasterize(&position, full);
                    let subset = rasterize(&position, compact);
                    for (slot, &channel) in COMPACT_CHANNELS.iter().enumerate() {
                        assert_eq!(
                            &subset.data()[slot * pixels..(slot + 1) * pixels],
                            &whole.data()[channel * pixels..(channel + 1) * pixels],
                            "compact plane {slot} must equal semantic channel {channel} ({}) at \
                             {width}x{height}, {stones} stones with {to_move:?} to move",
                            CHANNELS[channel].name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn raster_is_player_relative_and_order_independent() {
        let stones = vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.75, 0.75, Color::White),
        ];
        let mut reversed = stones.clone();
        reversed.reverse();
        let swapped = stones
            .iter()
            .map(|stone| Stone::new(stone.x, stone.y, stone.color.other()))
            .collect();
        let config = RasterConfig::square(24);
        let first = rasterize(&Position::new(0.1, stones, Color::Black), config);
        let reordered = rasterize(&Position::new(0.1, reversed, Color::Black), config);
        let color_swapped = rasterize(&Position::new(0.1, swapped, Color::White), config);
        assert_eq!(first, reordered);
        assert_eq!(first, color_swapped);
    }

    #[test]
    fn action_pixels_follow_raster_orientation() {
        let config = RasterConfig {
            width: 4,
            height: 2,
            kind: RasterKind::Semantic,
        };
        assert_eq!(action_pixel(0.1, 0.1, config), 0);
        assert_eq!(action_pixel(0.9, 0.9, config), 7);
        assert_eq!(action_pixel(1.0, 1.0, config), 7);
    }

    #[test]
    fn caller_owned_raster_matches_owned_raster() {
        let position = Position::new(
            0.1,
            vec![Stone::new(0.25, 0.75, Color::Black)],
            Color::White,
        );
        let config = RasterConfig::square(16);
        let expected = rasterize(&position, config);
        let mut data = vec![f32::NAN; CHANNEL_COUNT * config.pixels()];
        rasterize_into(&position, config, &mut data);
        assert_eq!(data, expected.data());
    }


    /// The pass plane is the pass count, not a lossy summary of it: two passes
    /// end the game, so a live position is only ever at zero or one.
    #[test]
    fn the_pass_plane_carries_the_whole_pass_state() {
        let position = Position::new(
            0.1,
            vec![Stone::new(0.3, 0.3, Color::Black), Stone::new(0.7, 0.7, Color::White)],
            Color::Black,
        );
        let config = RasterConfig::square_of(16, RasterKind::CompactRadius);
        let pixels = config.pixels();
        let slot = 5 * pixels;

        for (passes, expected) in [(0_u32, 0.0_f32), (1, 1.0)] {
            let position = position.clone().with_passes(passes);
            let mut data = vec![f32::NAN; config.channels() * pixels];
            super::rasterize_any_into(&position, config, &mut data);
            assert!(
                data[slot..slot + pixels].iter().all(|value| *value == expected),
                "{passes} passes should paint the plane {expected}"
            );
        }
    }


    /// The whole reason the layout exists: two board sizes must not render
    /// identically. An empty board is the case that matters, because it is
    /// where every other plane is zero and the net has nothing else to go on.
    #[test]
    fn the_radius_layout_separates_board_sizes() {
        let config = RasterConfig::square_of(64, RasterKind::CompactRadius);
        let mut mini = vec![0.0f32; config.channels() * config.pixels()];
        let mut standard = vec![0.0f32; config.channels() * config.pixels()];
        rasterize_any_into(
            &Position::new(1.0 / 18.0, Vec::new(), Color::Black),
            config,
            &mut mini,
        );
        rasterize_any_into(
            &Position::new(1.0 / 38.0, Vec::new(), Color::Black),
            config,
            &mut standard,
        );
        assert_ne!(mini, standard, "empty boards must differ by radius");

        // The plane holds the stone diameter, the same value the semantic
        // rasterizer writes at index 8.
        let pixels = config.pixels();
        let plane = &standard[6 * pixels..7 * pixels];
        assert!(plane.iter().all(|&v| (v - (2.0 / 38.0) as f32).abs() < 1e-6));
    }



    #[test]
    fn raster_has_stable_shape_and_ranges() {
        let position = Position::new(
            0.1,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.75, Color::White),
            ],
            Color::Black,
        );
        let raster = rasterize(&position, RasterConfig::square(32));
        assert_eq!(raster.data().len(), CHANNEL_COUNT * 32 * 32);
        for (channel, values) in (0..CHANNEL_COUNT).map(|index| (index, raster.channel(index))) {
            let minimum = if channel == 7 { -1.0 } else { 0.0 };
            assert!(
                values
                    .iter()
                    .all(|value| *value >= minimum && *value <= 1.0)
            );
        }
    }
}

