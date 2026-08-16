//! f64 against f32 for the compact raster, on realistic stone counts.
//!
//! Rasterization is the largest single CPU cost in generation -- 13,270 core
//! seconds against 1,977 for all inference on shard 29 of ddrnet-deep-komi, a
//! third of the machine. The writer computes in f64 and stores f32, and the
//! shader port measured f32 throughout as costing nothing observable
//! (0 of 131,072 disc pixels changed, worst ridge delta 2.3e-6, against fp16
//! inference that rounds at ~1e-3). This is what that narrowing buys.
//!
//!     cargo bench -p vgo-raster
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{
    RasterConfig, RasterKind, rasterize_compact_into, rasterize_compact_shader_reference_into,
    settled_mask,
};

fn fixture(count: usize, radius: f64) -> Position {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    // Jittered grid rather than rejection sampling: 52 stones at 2r separation
    // is dense enough that random placement stalls, and a failed fixture reads
    // as a missing measurement rather than a slow one.
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
    // Median game carries 28 stones, longest 52.
    for count in [14usize, 28, 52] {
        let position = fixture(count, radius);
        if position.stones().len() < count {
            println!("{count:>3} stones: fixture generation failed");
            continue;
        }
        let settled = settled_mask(&position, config);
        let mut data = vec![0.0_f32; 5 * pixels];

        // Warm the caches and the branch predictors before timing either.
        for _ in 0..8 {
            rasterize_compact_into(&position, config, &mut data);
            rasterize_compact_shader_reference_into(&position, config, &settled, &mut data);
        }

        let iterations = 200;
        let started = Instant::now();
        for _ in 0..iterations {
            rasterize_compact_into(&position, config, &mut data);
            black_box(&data);
        }
        let exact = started.elapsed().as_secs_f64() / iterations as f64;

        let started = Instant::now();
        for _ in 0..iterations {
            rasterize_compact_shader_reference_into(&position, config, &settled, &mut data);
            black_box(&data);
        }
        let narrowed = started.elapsed().as_secs_f64() / iterations as f64;

        // The f32 path is handed `settled`; the f64 path computes it. Time that
        // separately so the comparison is like for like rather than crediting
        // f32 with work it never did.
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask(&position, config));
        }
        let mask = started.elapsed().as_secs_f64() / iterations as f64;

        println!(
            "{count:>3} stones | total {:>7.3} ms | settled {:>7.3} ({:>2.0}%) | \
             voronoi {:>7.3} ({:>2.0}%) | naive-f32 voronoi {:>7.3}",
            exact * 1e3,
            mask * 1e3,
            mask / exact * 100.0,
            (exact - mask) * 1e3,
            (exact - mask) / exact * 100.0,
            narrowed * 1e3,
        );
    }
}
