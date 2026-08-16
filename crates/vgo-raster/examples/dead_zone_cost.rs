//! What does the dead-zone plane add to a raster?
//!
//!     cargo run --release -p vgo-raster --example dead_zone_cost
//!
//! The claim being checked is that it is nearly free: `settled` already builds
//! the distance-to-legal-set field, and the dead zone is another threshold on
//! it. The costs that are *not* shared are the legal-set vertices and stamping a
//! disc around each.
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind, rasterize_any_into};

fn fixture(count: usize, radius: f64, seed: u64) -> Position {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let spacing = 2.0 * radius * 1.05;
    let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
    let jitter = (spacing - 2.0 * radius) * 0.45;
    let mut stones: Vec<Stone> = Vec::new();
    for index in 0..count {
        let (row, column) = (index / per_row, index % per_row);
        let x = 0.06 + (column as f64 + 0.5) * spacing + (next() - 0.5) * jitter;
        let y = 0.06 + (row as f64 + 0.5) * spacing + (next() - 0.5) * jitter;
        if x > 0.97 || y > 0.97 {
            break;
        }
        stones.push(Stone::new(x, y, if index % 2 == 0 { Color::Black } else { Color::White }));
    }
    Position::new(radius, stones, Color::Black).with_komi(0.104)
}

fn time(position: &Position, kind: RasterKind) -> f64 {
    let config = RasterConfig::square_of(128, kind);
    let mut data = vec![0.0_f32; config.channels() * config.pixels()];
    let iterations = 50;
    let started = Instant::now();
    for _ in 0..iterations {
        rasterize_any_into(position, config, &mut data);
        black_box(&data);
    }
    started.elapsed().as_secs_f64() / iterations as f64 * 1e3
}

fn main() {
    let radius = 1.0 / 18.0;
    println!("{:>7} {:>12} {:>18} {:>10}", "stones", "compact ms", "+dead-zone ms", "ratio");
    for (count, seed) in [(0usize, 1u64), (14, 1), (28, 2), (52, 4)] {
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() {
            continue;
        }
        let stones = position.stones().len();
        let compact = time(&position, RasterKind::Compact);
        let with_dead = time(&position, RasterKind::CompactDeadZone);
        println!("{stones:>7} {compact:>12.3} {with_dead:>18.3} {:>9.2}x", with_dead / compact);
    }
    println!();
    println!("`compact` picks its settled implementation by stone count; the dead-zone");
    println!("layout always takes the distance transform, so the ratio below ~20 stones");
    println!("is that switch as much as the new plane.");
}
