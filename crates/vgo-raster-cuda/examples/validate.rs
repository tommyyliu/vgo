//! Does the CUDA settled mask agree with the CPU one, and is it faster?
//!
//!     cargo run --release -p vgo-raster-cuda --example validate
//!
//! Exact agreement is not expected and would be suspicious. The CPU path solves
//! a per-stone radial equation, walks the resulting contour at 1/128 tolerance
//! and scanline-fills it, so pixels within about one cell of the boundary can
//! fall either way. The kernel evaluates the definition directly and is exact.
//! Disagreement should therefore be confined to a thin boundary band and should
//! be *the CPU's* error, not the GPU's -- which this checks by re-testing every
//! disagreeing pixel against the definition on the host, in f64, independently
//! of both implementations.

use std::time::Instant;

use vgo_core::{Color, Point, Position, Stone, distance_to_legal_set, legal_set_vertices};
use vgo_raster::{RasterConfig, RasterKind, settled_mask};
use vgo_raster_cuda::{Precision, SettledRasterizer};

fn fixture(count: usize, radius: f64, seed: u64) -> Position {
    let mut state = seed | 1;
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

/// The definition itself, on the host, as a third opinion.
fn settled_by_definition(position: &Position, x: f64, y: f64, vertices: &[Point]) -> bool {
    let point = Point::new(x, y);
    let nearest = position
        .stones()
        .iter()
        .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min);
    nearest <= distance_to_legal_set(position, point, Some(vertices))
}

fn main() {
    let radius = 0.055_714_285_714_285_716;
    let config = RasterConfig::square_of(128, RasterKind::Compact);
    let pixels = config.pixels();

    let rasterizer = match SettledRasterizer::new(0) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("cannot run: {error}");
            std::process::exit(1);
        }
    };
    println!("kernel compiled by NVRTC and loaded\n");

    // Single position first: correctness against the definition.
    println!("correctness, one position at a time");
    println!("  {:>7} {:>9} {:>11}", "stones", "differ", "gpu right");
    let mut fixtures = Vec::new();
    for (count, seed) in [(14usize, 1u64), (28, 2), (28, 3), (52, 4), (52, 5)] {
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() {
            continue;
        }
        let vertices = legal_set_vertices(&position);
        let host = settled_mask(&position, config);
        let device = rasterizer.mask(&position, config).expect("launch");
        let (mut differ, mut gpu_right) = (0usize, 0usize);
        for pixel in 0..pixels {
            if host[pixel] == device[pixel] {
                continue;
            }
            differ += 1;
            let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
            let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
            if settled_by_definition(&position, x, y, &vertices) == device[pixel] {
                gpu_right += 1;
            }
        }
        println!(
            "  {:>7} {:>9} {:>11}",
            position.stones().len(),
            differ,
            format!("{gpu_right}/{differ}")
        );
        fixtures.push(position);
    }

    // A batch must give bit-identical masks to the same positions run singly,
    // or the offsets are wrong in a way correctness-against-definition would
    // not catch (every position would still be individually plausible).
    let refs: Vec<&_> = fixtures.iter().collect();
    let batched = rasterizer.masks(&refs, config).expect("batch launch");
    let mut mismatched = 0usize;
    for (index, position) in fixtures.iter().enumerate() {
        let single = rasterizer.mask(position, config).expect("launch");
        if single != batched[index] {
            mismatched += 1;
        }
    }
    println!(
        "\nbatch of {} agrees with single launches: {}",
        fixtures.len(),
        if mismatched == 0 { "yes".to_string() } else { format!("NO, {mismatched} differ") }
    );

    // Throughput. The broker batches 32 positions per inference, so that is the
    // shape that matters; 1 is shown to expose launch overhead.
    println!("\nthroughput, 28-stone positions");
    println!(
        "  {:>6} {:>12} {:>12} {:>10} {:>12} {:>10}",
        "batch", "cpu ms/pos", "f64 ms/pos", "speedup", "f32 ms/pos", "speedup"
    );
    let position = fixture(28, radius, 2);
    let started = Instant::now();
    for _ in 0..50 {
        std::hint::black_box(settled_mask(&position, config));
    }
    let cpu_each = started.elapsed().as_secs_f64() / 50.0;

    let single_precision = SettledRasterizer::with_precision(0, Precision::Single)
        .expect("f32 kernel");
    for batch in [1usize, 8, 32, 64, 128] {
        let batch_positions: Vec<&_> = (0..batch).map(|_| &position).collect();
        let _ = rasterizer.masks(&batch_positions, config).expect("warm");
        let iterations = if batch >= 64 { 10 } else { 30 };
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(rasterizer.masks(&batch_positions, config).expect("launch"));
        }
        let per_launch = started.elapsed().as_secs_f64() / iterations as f64;
        let per_position = per_launch / batch as f64;
        let _ = single_precision.masks(&batch_positions, config).expect("warm");
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(single_precision.masks(&batch_positions, config).expect("launch"));
        }
        let f32_per_position = started.elapsed().as_secs_f64() / iterations as f64 / batch as f64;
        println!(
            "  {:>6} {:>12.3} {:>12.3} {:>9.1}x {:>12.3} {:>9.1}x",
            batch,
            cpu_each * 1e3,
            per_position * 1e3,
            cpu_each / per_position,
            f32_per_position * 1e3,
            cpu_each / f32_per_position,
        );
    }

    // Does f32 change any pixel? Same test as above: the definition, in f64, on
    // the host, decides.
    println!("\nf32 correctness against the definition");
    println!("  {:>7} {:>9} {:>11}", "stones", "differ", "f32 right");
    for position in &fixtures {
        let vertices = legal_set_vertices(position);
        let exact = rasterizer.mask(position, config).expect("f64");
        let narrow = single_precision.mask(position, config).expect("f32");
        let (mut differ, mut right) = (0usize, 0usize);
        for pixel in 0..pixels {
            if exact[pixel] == narrow[pixel] {
                continue;
            }
            differ += 1;
            let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
            let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
            if settled_by_definition(position, x, y, &vertices) == narrow[pixel] {
                right += 1;
            }
        }
        println!(
            "  {:>7} {:>9} {:>11}",
            position.stones().len(),
            differ,
            format!("{right}/{differ}")
        );
    }
    println!();
    println!("cpu ms/pos is settled_mask alone, single-threaded.");
    println!("gpu includes host->device copies and the readback, which the real");
    println!("integration removes by writing into the ONNX input tensor in place.");
}
