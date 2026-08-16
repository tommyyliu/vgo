//! Every raster layout, timed side by side.
//!
//!     cargo run --release -p vgo-raster --example raster_cost
//!
//! Calls the writers directly rather than going through `rasterize_any_into`,
//! so the `distance-settled` feature cannot change what is being compared: both
//! `settled` implementations appear in every run.
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{
    CHANNEL_COUNT, COMPACT_CHANNELS, COMPACT_DEAD_ZONE_CHANNELS, RasterConfig, RasterKind,
    rasterize_compact_dead_zone_into, rasterize_compact_with_settled_into, rasterize_into,
    settled_mask, settled_mask_by_bounded_distance,
};

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

fn time(mut body: impl FnMut()) -> f64 {
    for _ in 0..5 {
        body();
    }
    let iterations = 60;
    let started = Instant::now();
    for _ in 0..iterations {
        body();
    }
    started.elapsed().as_secs_f64() / iterations as f64 * 1e3
}

fn main() {
    // The official radius. Every run so far trained at 39/700, 0.286% larger,
    // which moves none of these numbers meaningfully.
    let radius = 1.0 / 18.0;
    let size = 128;
    let semantic = RasterConfig::square_of(size, RasterKind::Semantic);
    let compact = RasterConfig::square_of(size, RasterKind::Compact);
    let with_dead = RasterConfig::square_of(size, RasterKind::CompactDeadZone);
    let pixels = compact.pixels();

    println!("{size}x{size}, milliseconds per position, one thread\n");
    println!(
        "{:>7} {:>12} {:>12} {:>12} {:>14}",
        "stones", "semantic", "compact", "compact-edt", "compact+dead"
    );
    println!(
        "{:>7} {:>12} {:>12} {:>12} {:>14}",
        "", format!("{CHANNEL_COUNT}ch"),
        format!("{}ch", COMPACT_CHANNELS.len()),
        format!("{}ch", COMPACT_CHANNELS.len()),
        format!("{}ch", COMPACT_DEAD_ZONE_CHANNELS.len()),
    );

    let mut totals = [0.0_f64; 4];
    let counts = [0usize, 7, 14, 21, 28, 40, 52];
    for &count in &counts {
        let position = fixture(count, radius, count as u64 + 1);
        if !position.validate().is_playable() {
            continue;
        }
        let stones = position.stones().len();

        let mut wide = vec![0.0_f32; CHANNEL_COUNT * pixels];
        let mut five = vec![0.0_f32; COMPACT_CHANNELS.len() * pixels];
        let mut six = vec![0.0_f32; COMPACT_DEAD_ZONE_CHANNELS.len() * pixels];

        let a = time(|| {
            rasterize_into(&position, semantic, &mut wide);
            black_box(&wide);
        });
        let b = time(|| {
            let mask = settled_mask(&position, compact);
            rasterize_compact_with_settled_into(&position, compact, &mask, &mut five);
            black_box(&five);
        });
        let c = time(|| {
            let mask = settled_mask_by_bounded_distance(&position, compact, 1).0;
            rasterize_compact_with_settled_into(&position, compact, &mask, &mut five);
            black_box(&five);
        });
        let d = time(|| {
            rasterize_compact_dead_zone_into(&position, with_dead, &mut six);
            black_box(&six);
        });
        for (total, value) in totals.iter_mut().zip([a, b, c, d]) {
            *total += value;
        }
        println!("{stones:>7} {a:>12.3} {b:>12.3} {c:>12.3} {d:>14.3}");
    }

    let n = counts.len() as f64;
    println!(
        "{:>7} {:>12.3} {:>12.3} {:>12.3} {:>14.3}   <- mean over the curve",
        "", totals[0] / n, totals[1] / n, totals[2] / n, totals[3] / n
    );
    println!();
    println!("compact      = today's production default: per-stone geometric `settled`.");
    println!("compact-edt  = the same five planes with `settled` from the distance transform");
    println!("               (the `distance-settled` feature).");
    println!("compact+dead = six planes, both masks from one distance transform.");
}
