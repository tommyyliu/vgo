//! What rasterization costs as the board gets finer.
//!
//!     cargo run --release -p vgo-raster --example raster_scaling
//!
//! Two things move at once when the radius shrinks: the grid needs more pixels
//! to keep the same precision per stone, and the position holds more stones.
//! Resolution alone is close to O(pixels), but `settled` and the Voronoi ridge
//! also walk the stones, so halving the radius costs more than the pixel count
//! says. Measured here rather than assumed, because the intuition that "the
//! model gets slower but the raster is fine" is backwards.
use std::hint::black_box;
use std::time::Instant;
use vgo_core::{Color, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind, rasterize_any_into};

fn fixture(count: usize, radius: f64, seed: u64) -> Position {
    let mut state = seed | 1;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 };
    let spacing = 2.0 * radius * 1.05;
    let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
    let jitter = (spacing - 2.0 * radius) * 0.45;
    let mut stones: Vec<Stone> = Vec::new();
    for index in 0..count {
        let row = index / per_row; let col = index % per_row;
        let x = radius + 0.02 + col as f64 * spacing + (next() - 0.5) * jitter;
        let y = radius + 0.02 + row as f64 * spacing + (next() - 0.5) * jitter;
        if x > 1.0 - radius || y > 1.0 - radius { break; }
        stones.push(Stone { x, y, color: if index % 2 == 0 { Color::Black } else { Color::White } });
    }
    Position::new(radius, stones, Color::Black)
}

fn main() {
    println!("{:>8} {:>7} {:>7} {:>12} {:>9}", "radius", "res", "stones", "ms/raster", "vs base");
    let mut base = 0.0f64;
    for (label, radius, res, stones) in [
        ("1/18", 1.0/18.0, 128usize, 28usize),
        ("1/18", 1.0/18.0, 192, 28),
        ("1/18", 1.0/18.0, 256, 28),
        ("1/36", 1.0/36.0, 256, 120),
        ("1/36", 1.0/36.0, 384, 120),
    ] {
        let position = fixture(stones, radius, 7);
        let config = RasterConfig::square_of(res, RasterKind::CompactPass);
        let mut data = vec![0.0f32; config.channels() * config.pixels()];
        for _ in 0..3 { rasterize_any_into(&position, config, &mut data); }
        let runs = if res >= 384 { 20 } else { 60 };
        let t = Instant::now();
        for _ in 0..runs { rasterize_any_into(black_box(&position), config, &mut data); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        if base == 0.0 { base = ms; }
        println!("{:>8} {:>7} {:>7} {:>12.3} {:>8.2}x", label, res, position.stones().len(), ms, ms / base);
    }
}
