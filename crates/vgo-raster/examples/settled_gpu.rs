//! Runs the GPU settled-mask path against the CPU authoritative path: checks
//! they agree per-pixel and times each. Run with:
//!
//!     cargo run -p vgo-raster --features gpu --release --example settled_gpu

use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind, settled_mask, settled_mask_gpu};

fn fixture(count: usize, radius: f64) -> Position {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let spacing = 2.0 * radius * 1.05;
    let per_row = ((0.88 / spacing).floor() as usize).max(1);
    let jitter = (spacing - 2.0 * radius) * 0.45;
    let mut stones: Vec<Stone> = Vec::new();
    for index in 0..count {
        let row = index / per_row;
        let column = index % per_row;
        let x = 0.06 + (column as f64 + 0.5) * spacing + (next() - 0.5) * jitter;
        let y = 0.06 + (row as f64 + 0.5) * spacing + (next() - 0.5) * jitter;
        if x > 0.97 || y > 0.97 {
            break;
        }
        let colour = if index % 2 == 0 { Color::Black } else { Color::White };
        stones.push(Stone::new(x, y, colour));
    }
    Position::new(radius, stones, Color::Black).with_komi(0.104)
}

fn main() {
    let radius = 0.055_714_285_714_285_716;
    let config = RasterConfig::square_of(128, RasterKind::Compact);
    let pixels = config.pixels();

    for count in [14usize, 28, 52] {
        let position = fixture(count, radius);
        if position.stones().len() < count {
            println!("{count:>3} stones: fixture generation failed");
            continue;
        }
        let cpu = settled_mask(&position, config);
        let gpu = match settled_mask_gpu(&position, config) {
            Some(mask) => mask,
            None => {
                println!("{count:>3} stones: GPU path returned None");
                continue;
            }
        };
        assert_eq!(cpu.len(), gpu.len());
        let mut disagreements = 0usize;
        for idx in 0..pixels {
            if cpu[idx] != gpu[idx] {
                disagreements += 1;
            }
        }
        let settled_count = cpu.iter().filter(|b| **b).count();
        println!(
            "{count:>3} stones: {disagreements}/{pixels} pixels differ, {settled_count} settled"
        );

        // Time the CPU path.
        let iterations = 200;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask(&position, config));
        }
        let cpu_ms = started.elapsed().as_secs_f64() / iterations as f64 * 1e3;

        // Time the GPU path (includes upload + dispatch + readback).
        let iterations = 200;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask_gpu(&position, config));
        }
        let gpu_ms = started.elapsed().as_secs_f64() / iterations as f64 * 1e3;

        println!(
            "{count:>3} stones: cpu {:>7.3} ms | gpu {:>7.3} ms | {:>5.1}x",
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}
