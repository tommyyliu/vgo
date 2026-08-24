//! Where rasterization's time goes as the board fills.
//!
//!     cargo run --release -p vgo-raster --example raster_split
//!
//! `settled` is a distance transform and pays O(pixels). The other four planes
//! walk the stone list per pixel for the nearest and second-nearest, which is
//! O(pixels * stones). Which one dominates therefore depends on stone count,
//! and a finer board moves that balance.
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{
    RasterConfig, RasterKind, COMPACT_CHANNELS, rasterize_compact_with_predicate_into,
    settled_mask_by_bounded_distance,
};

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

/// Minimum over several batches, not the mean of one.
///
/// Generation saturates every core while this runs, so a mean measures the
/// scheduler as much as the code -- badly enough that the first attempt read
/// 60 stones as slower than 120. The fastest batch is the one that got a clean
/// run at the work, which is what is being compared.
fn time<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    for _ in 0..3 { f(); }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..runs { f(); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        if ms < best { best = ms; }
    }
    best
}

fn main() {
    println!("{:>7} {:>7} {:>7} {:>11} {:>11} {:>11} {:>8} {:>10}",
             "radius", "res", "stones", "settled ms", "4 planes ms", "total ms", "planes %", "exact");
    for (radius, res, want) in [
        (1.0/18.0, 128usize, 28usize), (1.0/18.0, 128, 52),
        (1.0/38.0, 128, 90), (1.0/38.0, 128, 180), (1.0/38.0, 128, 330),
        (1.0/38.0, 192, 180), (1.0/38.0, 192, 330),
    ] {
        let position = fixture(want, radius, 7);
        let config = RasterConfig::square_of(res, RasterKind::CompactPass);
        let compact = RasterConfig { kind: RasterKind::Compact, ..config };
        let n = position.stones().len();
        let runs = if res >= 384 { 15 } else { 40 };
        let s = time(runs, || { black_box(settled_mask_by_bounded_distance(&position, config, 1)); });
        let predicate = settled_mask_by_bounded_distance(&position, config, 1).0;
        let mut data = vec![0.0f32; COMPACT_CHANNELS.len() * config.pixels()];
        let p = time(runs, || {
            rasterize_compact_with_predicate_into(black_box(&position), compact, &predicate, &mut data);
        });
        let exact = settled_mask_by_bounded_distance(&position, config, 1).1;
        println!("{:>7} {:>7} {:>7} {:>11.3} {:>11.3} {:>11.3} {:>7.0}% {:>10}",
                 format!("1/{}", (1.0/radius).round()), res, n, s, p, s + p,
                 100.0 * p / (s + p), exact);
    }
}
