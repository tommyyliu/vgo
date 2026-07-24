#![forbid(unsafe_code)]

use vgo_core::Position;

pub const CHANNEL_COUNT: usize = 10;
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
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RasterConfig {
    pub width: usize,
    pub height: usize,
}

impl RasterConfig {
    #[must_use]
    pub const fn square(size: usize) -> Self {
        Self {
            width: size,
            height: size,
        }
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

#[must_use]
pub fn rasterize(position: &Position, config: RasterConfig) -> SemanticRaster {
    let mut data = vec![0.0_f32; CHANNEL_COUNT * config.pixels()];
    rasterize_into(position, config, &mut data);
    SemanticRaster { config, data }
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
    let stones = position.stones();
    // Splitting by colour once hoists the per-stone colour comparison out of the
    // pixel loop entirely.
    let mut current_stones = Vec::with_capacity(stones.len());
    let mut opponent_stones = Vec::with_capacity(stones.len());
    for stone in stones {
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
            let mut current_square = f64::INFINITY;
            let mut opponent_square = f64::INFINITY;
            let mut nearest_square = f64::INFINITY;
            let mut second_square = f64::INFINITY;

            for &(sx, sy) in &current_stones {
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
            for &(sx, sy) in &opponent_stones {
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

    use super::{CHANNEL_COUNT, RasterConfig, action_pixel, rasterize, rasterize_into};

    /// The pre-optimization formulation: one `hypot` per (pixel, stone) pair.
    /// `rasterize_into` now accumulates squared distances and takes four square
    /// roots per pixel instead; squaring is monotonic so every min and ordering
    /// is preserved. This reference pins that equivalence.
    fn hypot_reference(position: &Position, config: RasterConfig) -> Vec<f32> {
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
            assert_eq!(
                produced.data(),
                expected.as_slice(),
                "raster diverged from the hypot reference at {stones} stones"
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
}
