#![forbid(unsafe_code)]

use vgo_core::{Color, Position, SettledRegion, legal_set_vertices};

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::settled_mask_gpu;

mod edt;
mod policy;
pub use policy::DensePolicy;
pub use edt::{
    dead_zone_mask, settled_and_dead_zone, settled_mask_by_bounded_distance,
    settled_mask_by_distance,
};

/// Stone count above which the distance-transform settled mask is worth its
/// fixed cost. Measured crossover sits between 14 and 28 stones.
const DISTANCE_SETTLED_MINIMUM_STONES: usize = 20;

/// Grid cells per stone radius below which the distance-transform mask is not
/// trustworthy.
///
/// Its bound assumes the sampled legal set resolves the real one, and the legal
/// set between densely packed stones is a sliver a couple of cells wide. Coarsen
/// the grid and those slivers fall between samples entirely: measured on the
/// same fixture at r = 1/18, a 128² raster (7.11 cells per radius) disagreed
/// with the definition on 0 pixels, while a 48² raster (2.67) disagreed on
/// **9.16%**. Six is chosen between those two points with margin toward the
/// safe side; it is calibrated, not derived.
const DISTANCE_SETTLED_MINIMUM_CELLS_PER_RADIUS: f64 = 6.0;

pub const CHANNEL_COUNT: usize = 12;

/// Entries in [`CHANNELS`], which is a *catalogue* rather than a layout.
///
/// It is deliberately larger than [`CHANNEL_COUNT`]: the semantic raster is the
/// first [`CHANNEL_COUNT`] of these, and later entries exist for layouts that
/// name channels by index without the semantic writer emitting them. Growing
/// this is safe; growing `CHANNEL_COUNT` is not, because that number is the
/// semantic tensor's shape and is baked into inference frames and ONNX profiles.
pub const CHANNEL_SPEC_COUNT: usize = 13;

/// Channels written by [`rasterize_rgb_into`]: red, green, blue.
pub const RGB_CHANNEL_COUNT: usize = 3;

/// Indices into [`CHANNELS`] that [`RasterKind::Compact`] keeps.
pub const COMPACT_CHANNELS: [usize; 5] = [
    0,  // current_stones
    1,  // opponent_stones
    6,  // voronoi_ridge
    10, // settled
    11, // komi
];

/// Indices [`RasterKind::CompactPass`] keeps: [`COMPACT_CHANNELS`], then
/// whether the previous move was a pass.
///
/// `Compact` cannot see a pending pass, so a net reading it cannot tell that
/// passing now would end the game -- it can neither pass to close out a win nor
/// see that passing while behind hands over the result. The plane is constant
/// over the board and therefore costs nothing to write.
pub const COMPACT_PASS_CHANNELS: [usize; 6] = [
    0,  // current_stones
    1,  // opponent_stones
    6,  // voronoi_ridge
    10, // settled          <- the capture predicate
    11, // komi
    9,  // previous_pass
];

/// Indices [`RasterKind::CompactDeadZone`] keeps: [`COMPACT_PASS_CHANNELS`]
/// with the dead zone in place of `settled`.
///
/// The two layouts differ in exactly one slot, and that is the point. Slot 3 is
/// "the capture predicate", `settled` for this repository's rules and
/// `dead_zone` for the official ones, so a model warm-starts from one onto the
/// other with every other input plane keeping its meaning and its weights. It
/// also makes the comparison between the two rulesets a one-plane A/B rather
/// than a change of representation.
pub const COMPACT_DEAD_ZONE_CHANNELS: [usize; 6] = [
    0,  // current_stones
    1,  // opponent_stones
    6,  // voronoi_ridge
    12, // dead_zone        <- the capture predicate
    11, // komi
    9,  // previous_pass
];
pub const DATASET_MAGIC: [u8; 8] = *b"VGODATA1";
pub const DATASET_VERSION: u32 = 2;

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

pub const CHANNELS: [ChannelSpec; CHANNEL_SPEC_COUNT] = [
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
    ChannelSpec {
        name: "dead_zone",
        scale: ChannelScale::Unit,
    },
];

/// Which channel layout a raster carries.
///
/// `Semantic` is the ten engineered channels. `Rgb` is the board as a player
/// sees it -- stone discs over Voronoi territory fill, three channels, no
/// derived fields. The two are not interchangeable inputs: a model trained on
/// one cannot read the other, so this belongs to a run's identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RasterKind {
    #[default]
    Semantic,
    Rgb,
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
    /// [`Compact`](Self::Compact), plus whether the previous move was a pass.
    ///
    /// This repository's rules, with the one thing `Compact` cannot express.
    /// Two passes end the game under every ruleset here, so a net that cannot
    /// see a pending pass is evaluating a different game than the one being
    /// played -- and only in the endgame, which is where the value head is
    /// asked the questions that decide results.
    CompactPass,
    /// [`CompactPass`](Self::CompactPass) with the dead zone in place of
    /// `settled`: the official rules' capture predicate.
    ///
    /// `settled` encodes *this* repository's capture rule -- a group lives
    /// while some future stone can still take area from it. The rules at
    /// `voronoigo.com` ask a different question: a group lives while a future
    /// stone could still be placed touching its territory, and dies once that
    /// territory is covered by the dead zone. That is a strictly more
    /// aggressive rule, so a net given only `settled` has to infer the
    /// condition it is actually judged by.
    ///
    /// `settled` is dropped rather than kept alongside. It is the wrong
    /// predicate here, and it is not a cheap passenger: measured at 128 square,
    /// it is 60-80% of the raster's cost, so carrying it for a ruleset that does
    /// not use it would more than double the price of every position.
    CompactDeadZone,
}

impl RasterKind {
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Semantic => CHANNEL_COUNT,
            Self::Rgb => RGB_CHANNEL_COUNT,
            Self::Compact => COMPACT_CHANNELS.len(),
            Self::CompactPass => COMPACT_PASS_CHANNELS.len(),
            Self::CompactDeadZone => COMPACT_DEAD_ZONE_CHANNELS.len(),
        }
    }

    /// Which entries of [`CHANNELS`] this layout writes, in order.
    #[must_use]
    pub const fn indices(self) -> &'static [usize] {
        match self {
            Self::Semantic | Self::Rgb => &[],
            Self::Compact => &COMPACT_CHANNELS,
            Self::CompactPass => &COMPACT_PASS_CHANNELS,
            Self::CompactDeadZone => &COMPACT_DEAD_ZONE_CHANNELS,
        }
    }

    /// Whether this layout carries the `dead_zone` plane.
    #[must_use]
    pub const fn has_dead_zone(self) -> bool {
        matches!(self, Self::CompactDeadZone)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Rgb => "rgb",
            Self::Compact => "compact",
            Self::CompactPass => "compact-pass",
            Self::CompactDeadZone => "compact-dead-zone",
        }
    }
}

impl std::str::FromStr for RasterKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "semantic" => Ok(Self::Semantic),
            "rgb" => Ok(Self::Rgb),
            "compact" => Ok(Self::Compact),
            "compact-pass" => Ok(Self::CompactPass),
            "compact-dead-zone" => Ok(Self::CompactDeadZone),
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
    pub fn into_data(self) -> Vec<f32> {
        self.data
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

    #[must_use]
    pub fn channel_rgb(&self, channel: usize) -> Vec<u8> {
        let spec = CHANNELS[channel];
        let mut rgb = Vec::with_capacity(self.config.pixels() * 3);
        for &value in self.channel(channel) {
            let color = match spec.scale {
                ChannelScale::Unit => unit_color(value),
                ChannelScale::Signed => signed_color(value),
            };
            rgb.extend_from_slice(&color);
        }
        rgb
    }

    #[must_use]
    pub fn overview_rgb(&self) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(self.config.pixels() * 3);
        for pixel in 0..self.config.pixels() {
            let current_stone = self.channel(0)[pixel];
            let opponent_stone = self.channel(1)[pixel];
            let current_area = self.channel(2)[pixel];
            let opponent_area = self.channel(3)[pixel];
            let ridge = self.channel(6)[pixel];
            let legal = self.channel(7)[pixel];

            let mut color = [232.0_f32, 235.0, 229.0];
            blend(&mut color, [39.0, 145.0, 154.0], 0.38 * current_area);
            blend(&mut color, [218.0, 91.0, 75.0], 0.38 * opponent_area);
            if legal < 0.0 {
                blend(&mut color, [42.0, 44.0, 48.0], 0.34 * -legal);
            } else {
                blend(&mut color, [238.0, 242.0, 226.0], 0.18 * legal);
            }
            blend(&mut color, [255.0, 255.0, 255.0], 0.34 * ridge);
            if current_stone > 0.0 {
                blend(&mut color, [10.0, 83.0, 91.0], current_stone);
            }
            if opponent_stone > 0.0 {
                blend(&mut color, [145.0, 37.0, 34.0], opponent_stone);
            }
            rgb.extend(color.map(to_u8));
        }
        rgb
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
        RasterKind::Rgb => rasterize_rgb_into(position, config, data),
        RasterKind::Compact => rasterize_compact_into(position, config, data),
        RasterKind::CompactPass | RasterKind::CompactDeadZone => {
            rasterize_compact_six_into(position, config, data);
        }
    }
}

/// Writes the six-plane layouts: [`RasterKind::CompactPass`] and
/// [`RasterKind::CompactDeadZone`].
///
/// The two differ only in slot 3, the capture predicate, so they share
/// everything here and diverge on one mask. `settled` and the dead zone are
/// both thresholds on the distance to the legal set, and each is asked for
/// alone: computing the pair and discarding one costs 60-80% of the raster for
/// nothing.
///
/// The pass plane is constant over the board. Two passes end the game, so a
/// position that is still being played has a count of 0 or 1 and the boolean is
/// the count rather than a summary of it.
pub fn rasterize_compact_six_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    let pixels = config.pixels();
    assert_eq!(data.len(), COMPACT_PASS_CHANNELS.len() * pixels);
    let predicate = match config.kind {
        RasterKind::CompactDeadZone => edt::dead_zone_mask(position, config, 1).0,
        _ => settled_for_raster(position, config),
    };
    let compact = RasterConfig {
        kind: RasterKind::Compact,
        ..config
    };
    let (head, tail) = data.split_at_mut(COMPACT_CHANNELS.len() * pixels);
    rasterize_compact_with_predicate_into(position, compact, &predicate, head);
    tail.fill(f32::from(position.consecutive_passes() > 0));
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
    let cells_per_radius = config.width.min(config.height) as f64 * position.radius();
    if position.stones().len() >= DISTANCE_SETTLED_MINIMUM_STONES
        && cells_per_radius >= DISTANCE_SETTLED_MINIMUM_CELLS_PER_RADIUS
    {
        return edt::settled_mask_by_bounded_distance(position, config, 1).0;
    }
    settled_mask(position, config)
}

pub fn rasterize_compact_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    let settled = settled_for_raster(position, config);
    rasterize_compact_with_predicate_into(position, config, &settled, data);
}

/// The five compact planes, with slot 3 -- the capture predicate -- supplied
/// rather than computed.
///
/// Split out because that plane is where the cost is: 92% of this function at
/// the median stone count under the per-stone geometric solve, and still 60-80%
/// under the distance transform, while the other four are per-pixel work over
/// the stone list. That asymmetry means the two want different hardware, and
/// `vgo-raster-cuda` exists to compute the mask for a whole batch in one launch
/// and hand it here.
///
/// The predicate is `settled` for this repository's rules and the dead zone for
/// the official ones. Nothing below cares which: both are a boolean per pixel
/// saying whether this point is beyond further contest, and the layouts that
/// name one or the other put it in the same slot deliberately.
///
/// The mask must be `config.pixels()` long and indexed row-major, exactly as
/// [`settled_mask`] returns it.
pub fn rasterize_compact_with_predicate_into(
    position: &Position,
    config: RasterConfig,
    settled: &[bool],
    data: &mut [f32],
) {
    assert!(config.width > 0 && config.height > 0);
    assert!(position.validate().is_playable());
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

/// The compact raster as `compact.wgsl` computes it: f32 throughout.
///
/// `rasterize_compact_into` is authoritative and works in f64. WGSL has no f64,
/// so the shader is a deliberate narrowing, and this function is the same
/// arithmetic in the same order so the cost of that narrowing can be measured
/// on the host. It is not a second implementation of the raster -- it exists to
/// be compared against the f64 writer, and `compact.wgsl` must be kept in step
/// with it. See docs/CLIENT_BOT.md.
///
/// `settled` is taken as a caller-supplied mask rather than recomputed: on the
/// GPU that channel is uploaded, because its cost is per-stone contour geometry
/// rather than per-pixel work and it does not belong in a pixel shader.
pub fn rasterize_compact_shader_reference_into(
    position: &Position,
    config: RasterConfig,
    settled: &[bool],
    data: &mut [f32],
) {
    assert!(config.width > 0 && config.height > 0);
    let pixels = config.pixels();
    assert_eq!(data.len(), COMPACT_CHANNELS.len() * pixels);
    assert_eq!(settled.len(), pixels);

    // Matches `const NONE` in compact.wgsl: coordinates are normalised, so the
    // largest real squared distance is 2 and any sentinel far above it reads as
    // "no stone seen yet".
    const NONE: f32 = 1.0e30;

    let radius = position.radius() as f32;
    let radius_square = radius * radius;
    let to_move = position.to_move();
    let mover_komi = match to_move {
        Color::Black => position.komi() as f32,
        Color::White => -(position.komi() as f32),
    };
    let (current_stones, opponent_stones) = relative_stones(position, to_move);

    for row in 0..config.height {
        let y = (row as f32 + 0.5) / config.height as f32;
        for column in 0..config.width {
            let x = (column as f32 + 0.5) / config.width as f32;
            let pixel = row * config.width + column;

            let mut current_square = NONE;
            let mut opponent_square = NONE;
            let mut nearest_square = NONE;
            let mut second_square = NONE;

            for &(stone_x, stone_y) in &current_stones {
                let dx = x - stone_x as f32;
                let dy = y - stone_y as f32;
                let square = dx * dx + dy * dy;
                current_square = current_square.min(square);
                if square < nearest_square {
                    second_square = nearest_square;
                    nearest_square = square;
                } else if square < second_square {
                    second_square = square;
                }
            }
            for &(stone_x, stone_y) in &opponent_stones {
                let dx = x - stone_x as f32;
                let dy = y - stone_y as f32;
                let square = dx * dx + dy * dy;
                opponent_square = opponent_square.min(square);
                if square < nearest_square {
                    second_square = nearest_square;
                    nearest_square = square;
                } else if square < second_square {
                    second_square = square;
                }
            }

            data[pixel] = f32::from(current_square <= radius_square);
            data[pixels + pixel] = f32::from(opponent_square <= radius_square);
            data[2 * pixels + pixel] = if second_square < NONE {
                (1.0 - (second_square.sqrt() - nearest_square.sqrt()) / radius).clamp(0.0, 1.0)
            } else {
                0.0
            };
            data[3 * pixels + pixel] = f32::from(settled[pixel]);
            data[4 * pixels + pixel] = mover_komi;
        }
    }
}

/// Writes a semantic raster into caller-owned contiguous channel-first storage.
///
/// Reusable or pinned inference buffers can use this entry point to avoid an
/// intermediate per-position allocation and host-side gather.
pub fn rasterize_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    assert!(config.width > 0 && config.height > 0);
    assert!(position.validate().is_playable());
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

fn relative_stones(position: &Position, to_move: Color) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
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

/// The player-facing palette, taken from `reference/js-reference/voronoi_go.html`
/// (`COLORS`, line 199) so the network sees the same picture a human does.
/// Values are 0-255 to keep them legible against the source; the raster is
/// normalized to unit range on write.
///
/// The JS names the two sides Black and White but draws them blue and orange.
/// Here they are relative to the side to move, matching every other channel in
/// this crate: `CURRENT_*` is whoever is about to play.
const CURRENT_STONE: [f32; 3] = [90.0, 162.0, 236.0]; // #5aa2ec
const CURRENT_REGION: [f32; 3] = [34.0, 64.0, 92.0]; // #22405c
const OPPONENT_STONE: [f32; 3] = [240.0, 151.0, 90.0]; // #f0975a
const OPPONENT_REGION: [f32; 3] = [104.0, 57.0, 26.0]; // #68391a
const BOARD_BACKGROUND: [f32; 3] = [14.0, 17.0, 22.0]; // #0e1116

/// The legal-placement overlay from `freeMaskSVG` (voronoi_go.html line 369):
/// legal cells tinted `rgb(205,214,232)` at alpha 42/255, illegal left bare.
const LEGAL_TINT: [f32; 3] = [205.0, 214.0, 232.0];
const LEGAL_TINT_ALPHA: f32 = 42.0 / 255.0;

/// Renders the board as a player sees it: three channels, stone discs over
/// Voronoi territory fill.
///
/// This deliberately carries no derived fields -- no distance transform, no
/// ridge, no legality map. The question it exists to answer is whether a
/// convolutional tower can recover that structure from the picture alone, so
/// handing it any of those channels would defeat the experiment. See
/// `docs/RGB_REPRESENTATION_EXPERIMENT.md`.
///
/// Geometry is shared with [`rasterize_into`]: territory is the nearer-stone
/// test and a stone is the disc within `radius`, so the two rasters agree about
/// the position and differ only in what they expose.
pub fn rasterize_rgb_into(position: &Position, config: RasterConfig, data: &mut [f32]) {
    assert!(config.width > 0 && config.height > 0);
    assert!(position.validate().is_playable());
    let pixels = config.pixels();
    assert_eq!(data.len(), RGB_CHANNEL_COUNT * pixels);
    let radius = position.radius();

    let to_move = position.to_move();
    let mut current_stones = Vec::with_capacity(position.stones().len());
    let mut opponent_stones = Vec::with_capacity(position.stones().len());
    for stone in position.stones() {
        if stone.color == to_move {
            current_stones.push((stone.x, stone.y));
        } else {
            opponent_stones.push((stone.x, stone.y));
        }
    }

    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        for column in 0..config.width {
            let x = (column as f64 + 0.5) / config.width as f64;
            let pixel = row * config.width + column;

            // Squared distances for the same reason as the semantic raster: the
            // ordering is unchanged and it keeps a hypot out of the inner loop.
            let mut current_square = f64::INFINITY;
            let mut opponent_square = f64::INFINITY;
            for &(sx, sy) in &current_stones {
                let dx = x - sx;
                let dy = y - sy;
                let square = dx * dx + dy * dy;
                if square < current_square {
                    current_square = square;
                }
            }
            for &(sx, sy) in &opponent_stones {
                let dx = x - sx;
                let dy = y - sy;
                let square = dx * dx + dy * dy;
                if square < opponent_square {
                    opponent_square = square;
                }
            }
            // Nearest stone of either colour, which is what legality turns on.
            let nearest_square = current_square.min(opponent_square);

            let mut color = BOARD_BACKGROUND;
            // Draw order follows renderBoard(): filled cells, then the legal
            // mask, then stones on top.
            //
            // The legal overlay is the reason this is worth carrying. A plain
            // picture of stones and territory does not say where a move is
            // *allowed*, so a model reading it has to infer the placement rule
            // from scratch -- and the first RGB run predicted MCTS targets well
            // (policy_kl 1.21 against semantic's 4.48) while losing the
            // head-to-head 0.271, which is what a policy proposing unplayable
            // moves during search would look like.
            //
            // Legality is the same predicate `legal_clearance` computes above:
            // a placement is legal when it clears the board edge and sits at
            // least two radii from every stone.
            //
            // `ownership` only compares its two arguments and tests them for
            // finiteness, and squaring preserves both on nonnegative reals
            // (INFINITY squared is still INFINITY), so squared distances give
            // the same answer without the square roots.
            let (current_area, opponent_area) = ownership(current_square, opponent_square);
            blend(&mut color, CURRENT_REGION, current_area);
            blend(&mut color, OPPONENT_REGION, opponent_area);

            let board_clearance = (x - radius)
                .min(1.0 - radius - x)
                .min(y - radius)
                .min(1.0 - radius - y);
            let stone_clearance = if nearest_square.is_finite() {
                nearest_square.sqrt() - 2.0 * radius
            } else {
                f64::INFINITY
            };
            if board_clearance.min(stone_clearance) > 0.0 {
                blend(&mut color, LEGAL_TINT, LEGAL_TINT_ALPHA);
            }

            if current_square <= radius * radius {
                blend(&mut color, CURRENT_STONE, 1.0);
            }
            if opponent_square <= radius * radius {
                blend(&mut color, OPPONENT_STONE, 1.0);
            }

            for (channel, value) in color.iter().enumerate() {
                set(data, pixels, channel, pixel, value / 255.0);
            }
        }
    }
}

#[must_use]
pub fn rasterize_rgb(position: &Position, config: RasterConfig) -> Vec<f32> {
    let mut data = vec![0.0_f32; RGB_CHANNEL_COUNT * config.pixels()];
    rasterize_rgb_into(position, config, &mut data);
    data
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

fn unit_color(value: f32) -> [u8; 3] {
    let value = value.clamp(0.0, 1.0);
    [
        to_u8(22.0 + 226.0 * value),
        to_u8(31.0 + 193.0 * value),
        to_u8(48.0 + 64.0 * (1.0 - value)),
    ]
}

fn signed_color(value: f32) -> [u8; 3] {
    let value = value.clamp(-1.0, 1.0);
    if value < 0.0 {
        let amount = -value;
        [
            to_u8(245.0),
            to_u8(241.0 * (1.0 - amount) + 66.0 * amount),
            to_u8(235.0 * (1.0 - amount) + 63.0 * amount),
        ]
    } else {
        [
            to_u8(245.0 * (1.0 - value) + 42.0 * value),
            to_u8(241.0 * (1.0 - value) + 157.0 * value),
            to_u8(235.0 * (1.0 - value) + 108.0 * value),
        ]
    }
}

fn blend(target: &mut [f32; 3], source: [f32; 3], alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    for channel in 0..3 {
        target[channel] = target[channel].mul_add(1.0 - alpha, source[channel] * alpha);
    }
}

fn to_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position, Stone};

    use super::{
        CHANNEL_COUNT, CHANNELS, COMPACT_CHANNELS, RGB_CHANNEL_COUNT, RasterConfig, RasterKind,
        action_pixel, rasterize, rasterize_compact_into,
        rasterize_compact_shader_reference_into, rasterize_into, rasterize_rgb, settled_for_raster,
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

    /// The direct compact writer must stay bit-for-bit identical to the
    /// semantic planes it names across sparse and dense positions.
    #[test]
    fn the_shader_reference_matches_the_f64_writer() {
        // compact.wgsl computes in f32 because WGSL has no f64. This measures
        // what that costs against the authoritative writer, over positions
        // shaped like real ones: the median game carries 28 stones and the
        // longest 52.
        //
        // The two disc channels are threshold tests, so a pixel centre landing
        // within f32 epsilon of a stone's edge can legitimately fall either
        // way. Those are counted and required to be vanishingly rare rather
        // than forbidden -- forbidding them would be pinning luck. The ridge is
        // continuous and is held to a tolerance.
        let radius = 0.055_714_285_714_285_716;
        let config = RasterConfig::square_of(128, RasterKind::Compact);
        let pixels = config.pixels();
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut boundary_disagreements = 0usize;
        let mut worst_ridge = 0.0_f32;
        let mut compared = 0usize;

        for count in [1usize, 2, 7, 28, 52] {
            let mut stones = Vec::new();
            let mut attempts = 0;
            while stones.len() < count && attempts < 4000 {
                attempts += 1;
                let x = 0.06 + next() * 0.88;
                let y = 0.06 + next() * 0.88;
                // Stones may not overlap within 2r, which is what the exact
                // simulator enforces; a fixture that violates it is not a
                // position the shader will ever see.
                if stones.iter().any(|s: &Stone| {
                    let (dx, dy) = (s.x - x, s.y - y);
                    dx * dx + dy * dy < (2.0 * radius) * (2.0 * radius)
                }) {
                    continue;
                }
                let colour = if stones.len() % 2 == 0 { Color::Black } else { Color::White };
                stones.push(Stone::new(x, y, colour));
            }
            if stones.len() < count {
                continue;
            }
            let to_move = if count % 2 == 0 { Color::Black } else { Color::White };
            let position = Position::new(radius, stones, to_move).with_komi(0.104);
            if !position.validate().is_playable() {
                continue;
            }

            let mut exact = vec![0.0_f32; COMPACT_CHANNELS.len() * pixels];
            rasterize_compact_into(&position, config, &mut exact);
            let settled = settled_for_raster(&position, config);
            let mut narrowed = vec![0.0_f32; COMPACT_CHANNELS.len() * pixels];
            rasterize_compact_shader_reference_into(&position, config, &settled, &mut narrowed);

            compared += 1;
            for pixel in 0..pixels {
                for channel in [0usize, 1] {
                    if exact[channel * pixels + pixel] != narrowed[channel * pixels + pixel] {
                        boundary_disagreements += 1;
                    }
                }
                let delta = (exact[2 * pixels + pixel] - narrowed[2 * pixels + pixel]).abs();
                worst_ridge = worst_ridge.max(delta);
                // settled and komi are copied, not computed, so they must be exact.
                assert_eq!(exact[3 * pixels + pixel], narrowed[3 * pixels + pixel]);
                assert_eq!(exact[4 * pixels + pixel], narrowed[4 * pixels + pixel]);
            }
        }

        assert!(compared >= 4, "fixture generation failed, only {compared} positions");
        let total = compared * pixels * 2;
        assert!(
            boundary_disagreements * 100_000 < total,
            "f32 flipped {boundary_disagreements} of {total} disc-channel pixels, \
             which is more than edge cases"
        );
        assert!(
            worst_ridge < 1.0e-4,
            "ridge drifted by {worst_ridge} in f32, beyond rounding"
        );
        println!(
            "f32 vs f64 over {compared} positions: {boundary_disagreements}/{total} disc pixels \
             differ, worst ridge delta {worst_ridge:.3e}"
        );
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

    fn two_stone_position() -> Position {
        Position::new(
            0.1,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.75, Color::White),
            ],
            Color::Black,
        )
    }

    #[test]
    fn rgb_raster_has_three_channels_in_unit_range() {
        let raster = rasterize_rgb(&two_stone_position(), RasterConfig::square(32));
        assert_eq!(raster.len(), RGB_CHANNEL_COUNT * 32 * 32);
        assert_eq!(RasterKind::Rgb.channels(), RGB_CHANNEL_COUNT);
        assert_eq!(RasterKind::Semantic.channels(), CHANNEL_COUNT);
        assert!(raster.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    }

    #[test]
    fn rgb_raster_paints_each_stone_in_its_own_colour() {
        let config = RasterConfig::square(64);
        let raster = rasterize_rgb(&two_stone_position(), config);
        let pixels = config.pixels();
        let sample = |x: f64, y: f64| {
            let column = (x * config.width as f64) as usize;
            let row = (y * config.height as f64) as usize;
            let pixel = row * config.width + column;
            [
                raster[pixel],
                raster[pixels + pixel],
                raster[2 * pixels + pixel],
            ]
        };

        // The side to move is Black, so its stone takes the current colour and
        // White's takes the opponent colour. CURRENT_STONE is blue-dominant and
        // OPPONENT_STONE is red-dominant, which is the cheapest stable way to
        // tell them apart without pinning exact palette values.
        let current = sample(0.25, 0.25);
        assert!(
            current[2] > current[0],
            "current stone should be blue-dominant, got {current:?}"
        );
        let opponent = sample(0.75, 0.75);
        assert!(
            opponent[0] > opponent[2],
            "opponent stone should be red-dominant, got {opponent:?}"
        );
    }

    #[test]
    fn rgb_and_semantic_rasters_agree_about_stones_and_territory() {
        // The experiment only means anything if both rasters describe the same
        // position; they must differ in what they expose, not in what they show.
        let config = RasterConfig::square(48);
        let position = two_stone_position();
        let semantic = rasterize(&position, config);
        let rgb = rasterize_rgb(&position, config);
        let pixels = config.pixels();

        for pixel in 0..pixels {
            let current_stone = semantic.channel(0)[pixel] > 0.0;
            let opponent_stone = semantic.channel(1)[pixel] > 0.0;
            let color = [rgb[pixel], rgb[pixels + pixel], rgb[2 * pixels + pixel]];
            if current_stone {
                assert!(
                    color[2] > color[0],
                    "pixel {pixel} is a current stone in the semantic raster but not blue-dominant in RGB"
                );
            }
            if opponent_stone {
                assert!(
                    color[0] > color[2],
                    "pixel {pixel} is an opponent stone in the semantic raster but not red-dominant in RGB"
                );
            }
        }
    }

    #[test]
    fn rgb_legal_overlay_agrees_with_the_semantic_clearance_channel() {
        // The overlay exists so the picture states where a move is allowed.
        // Both rasters must agree about legality, or the RGB model is being
        // taught a different rule than the search enforces.
        //
        // Territory fill varies the base colour across the board, so this
        // compares the two populations rather than each pixel against a fixed
        // background: every legal cell is brighter than every illegal one that
        // shares its territory owner.
        let config = RasterConfig::square(48);
        let rgb_config = RasterConfig::square_of(48, RasterKind::Rgb);
        let position = two_stone_position();
        let semantic = rasterize(&position, config);
        let rgb = rasterize_rgb(&position, rgb_config);
        let pixels = config.pixels();

        let mut legal_brightness: Vec<f32> = Vec::new();
        let mut illegal_brightness: Vec<f32> = Vec::new();
        for pixel in 0..pixels {
            // Skip stones: a disc paints over the tint, so legality is not
            // readable there in either raster. Skip the current player's
            // territory too, so both populations share one base colour.
            if semantic.channel(0)[pixel] > 0.0
                || semantic.channel(1)[pixel] > 0.0
                || semantic.channel(2)[pixel] > 0.0
            {
                continue;
            }
            let clearance = semantic.channel(7)[pixel];
            // Near the boundary a half-pixel of sampling difference decides the
            // sign, so only compare where the predicate is unambiguous.
            if clearance.abs() < 0.05 {
                continue;
            }
            let brightness: f32 = (0..3).map(|c| rgb[c * pixels + pixel]).sum();
            if clearance > 0.0 {
                legal_brightness.push(brightness);
            } else {
                illegal_brightness.push(brightness);
            }
        }

        assert!(
            legal_brightness.len() > 50 && illegal_brightness.len() > 50,
            "expected a meaningful sample of each: {} legal, {} illegal",
            legal_brightness.len(),
            illegal_brightness.len()
        );
        let dimmest_legal = legal_brightness.iter().copied().fold(f32::MAX, f32::min);
        let brightest_illegal = illegal_brightness.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            dimmest_legal > brightest_illegal,
            "legal cells must be tinted brighter than illegal ones: \
             dimmest legal {dimmest_legal}, brightest illegal {brightest_illegal}"
        );
    }

    #[test]
    fn rgb_raster_is_a_relative_view_like_every_other_channel() {
        // Swapping both the stone colours and the side to move leaves the
        // position identical from the mover's seat, so the picture must not move.
        let config = RasterConfig::square(32);
        let black_to_play = rasterize_rgb(&two_stone_position(), config);
        let white_to_play = rasterize_rgb(
            &Position::new(
                0.1,
                vec![
                    Stone::new(0.25, 0.25, Color::White),
                    Stone::new(0.75, 0.75, Color::Black),
                ],
                Color::White,
            ),
            config,
        );
        assert_eq!(black_to_play, white_to_play);
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
        assert_eq!(raster.overview_rgb().len(), 32 * 32 * 3);
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

    /// The two six-plane layouts must differ in exactly one slot, and it must be
    /// the capture predicate.
    ///
    /// This is what lets a model move between rulesets by reinitialising one
    /// input slice instead of relearning what every plane means, and what makes
    /// a ruleset comparison a one-plane A/B. It is an easy property to break by
    /// appending a channel to one list and not the other, so it is asserted
    /// rather than left to the comment above the constants.
    #[test]
    fn the_two_rulesets_differ_in_one_plane() {
        let ours = RasterKind::CompactPass.indices();
        let theirs = RasterKind::CompactDeadZone.indices();
        assert_eq!(ours.len(), theirs.len());
        let differing: Vec<usize> = (0..ours.len()).filter(|i| ours[*i] != theirs[*i]).collect();
        assert_eq!(differing, vec![3], "only slot 3 may differ");
        assert_eq!(CHANNELS[ours[3]].name, "settled");
        assert_eq!(CHANNELS[theirs[3]].name, "dead_zone");

        // And the first five of ours are Compact's, so a Compact model warm
        // starts by adding a plane rather than by permuting the ones it has.
        assert_eq!(&ours[..COMPACT_CHANNELS.len()], &COMPACT_CHANNELS[..]);
        assert_eq!(CHANNELS[ours[5]].name, "previous_pass");
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
        let config = RasterConfig::square_of(16, RasterKind::CompactPass);
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

    /// Both six-plane layouts agree everywhere except the predicate, on a real
    /// position rather than by construction.
    #[test]
    fn the_rulesets_share_every_plane_but_the_predicate() {
        let position = Position::new(
            1.0 / 18.0,
            vec![
                Stone::new(0.3, 0.35, Color::Black),
                Stone::new(0.45, 0.3, Color::Black),
                Stone::new(0.7, 0.65, Color::White),
                Stone::new(0.6, 0.78, Color::White),
            ],
            Color::Black,
        )
        .with_komi(0.104);
        assert!(position.validate().is_playable());
        let size = 64;
        let pixels = size * size;

        let mut ours = vec![0.0_f32; 6 * pixels];
        let mut theirs = vec![0.0_f32; 6 * pixels];
        super::rasterize_any_into(&position, RasterConfig::square_of(size, RasterKind::CompactPass), &mut ours);
        super::rasterize_any_into(
            &position,
            RasterConfig::square_of(size, RasterKind::CompactDeadZone),
            &mut theirs,
        );

        for slot in 0..6 {
            let range = slot * pixels..(slot + 1) * pixels;
            let same = ours[range.clone()] == theirs[range];
            assert_eq!(same, slot != 3, "slot {slot} sameness");
        }
    }
}
