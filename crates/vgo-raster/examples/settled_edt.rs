//! Is the distance-transform settled mask fast enough, and accurate enough?
//!
//!     cargo run --release -p vgo-raster --example settled_edt
//!
//! Compares three implementations of the same set:
//!
//!   exact    `settled_mask` — per-stone radial solve, contour, scanline fill.
//!            O(n²) in stones and 92-96% of rasterization cost.
//!   edt      `settled_mask_by_distance` — D_S <= D_L as two distance fields.
//!            O(pixels), but D_L is sampled and therefore an overestimate.
//!
//! Accuracy is judged against neither of them. Every disagreeing pixel is
//! re-tested against the definition in f64 on the host, so the column says
//! which implementation is actually right rather than which is in the majority.

use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Point, Position, Stone, distance_to_legal_set, legal_set_vertices};
use vgo_raster::{
    RasterConfig, RasterKind, settled_mask, settled_mask_by_distance,
    settled_mask_by_bounded_distance,
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

fn settled_by_definition(position: &Position, x: f64, y: f64, vertices: &[Point]) -> bool {
    let nearest = position
        .stones()
        .iter()
        .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min);
    nearest <= distance_to_legal_set(position, Point::new(x, y), Some(vertices))
}

fn main() {
    let radius = 0.055_714_285_714_285_716;
    let config = RasterConfig::square_of(128, RasterKind::Compact);
    let pixels = config.pixels();

    println!(
        "{:>7} {:>4} {:>11} {:>11} {:>9} {:>9} {:>10} {:>10}",
        "stones", "over", "exact ms", "edt ms", "speedup", "differ", "edt right", "exact right"
    );

    for (count, seed) in [(14usize, 1u64), (28, 2), (28, 3), (52, 4), (52, 5)] {
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() {
            continue;
        }
        let stones = position.stones().len();
        let vertices = legal_set_vertices(&position);
        let truth = settled_mask(&position, config);

        let iterations = 20;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask(&position, config));
        }
        let exact_seconds = started.elapsed().as_secs_f64() / iterations as f64;

        for oversample in [1usize, 2, 4] {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(settled_mask_by_distance(&position, config, oversample));
            }
            let edt_seconds = started.elapsed().as_secs_f64() / iterations as f64;
            let candidate = settled_mask_by_distance(&position, config, oversample);

            let (mut differ, mut edt_right, mut exact_right) = (0usize, 0usize, 0usize);
            for pixel in 0..pixels {
                if truth[pixel] == candidate[pixel] {
                    continue;
                }
                differ += 1;
                let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
                let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
                let actual = settled_by_definition(&position, x, y, &vertices);
                if actual == candidate[pixel] {
                    edt_right += 1;
                } else if actual == truth[pixel] {
                    exact_right += 1;
                }
            }
            println!(
                "{stones:>7} {oversample:>4} {:>11.3} {:>11.3} {:>8.1}x {:>9} {:>10} {:>10}",
                exact_seconds * 1e3,
                edt_seconds * 1e3,
                exact_seconds / edt_seconds,
                differ,
                edt_right,
                exact_right,
            );
        }
    }
    // The hybrid: sampled bounds decide what they can, the exact continuous
    // test decides the rest. Should be bit-for-bit the definition.
    println!();
    println!(
        "{:>7} {:>4} {:>11} {:>11} {:>9} {:>12}",
        "stones", "over", "shipping ms", "hybrid ms", "speedup", "exact tests"
    );
    for (count, seed, over) in [(14usize, 1u64, 1usize), (28, 2, 1), (52, 4, 1), (52, 4, 2), (52, 4, 3), (52, 5, 2)] {
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() {
            continue;
        }
        let stones = position.stones().len();
        let vertices = legal_set_vertices(&position);
        let truth = settled_mask(&position, config);
        let iterations = 20;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask(&position, config));
        }
        let exact_seconds = started.elapsed().as_secs_f64() / iterations as f64;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(settled_mask_by_bounded_distance(&position, config, over));
        }
        let hybrid_seconds = started.elapsed().as_secs_f64() / iterations as f64;
        let (candidate, tests) = settled_mask_by_bounded_distance(&position, config, over);
        let mut wrong = 0usize;
        for pixel in 0..pixels {
            let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
            let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
            if candidate[pixel] != settled_by_definition(&position, x, y, &vertices) {
                wrong += 1;
            }
        }
        let differ = (0..pixels).filter(|p| truth[*p] != candidate[*p]).count();
        // The shipping implementation is not exact either: it walks a contour
        // at 1/128 tolerance. Count its errors on the same footing.
        let mut truth_wrong = 0usize;
        for pixel in 0..pixels {
            let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
            let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
            if truth[pixel] != settled_by_definition(&position, x, y, &vertices) {
                truth_wrong += 1;
            }
        }
        println!(
            "{stones:>7} {over:>4} {:>11.3} {:>11.3} {:>8.1}x {:>12}   wrong: hybrid {wrong}, shipping {truth_wrong} (differ {differ})",
            exact_seconds * 1e3,
            hybrid_seconds * 1e3,
            exact_seconds / hybrid_seconds,
            tests,
        );
    }

    println!();
    println!("'differ' is against the existing implementation, of {pixels} pixels.");
    println!("'edt right' / 'exact right' break those down by what the definition says.");
}
