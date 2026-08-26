//! Where rasterization spends its time, and whether a change to it is correct.
//!
//! Rasterization is the largest CPU cost in self-play generation -- a sampling
//! profile of the generator put it at 86% of the actor threads' time -- so this
//! isolates it from everything else. No GPU, no model, no ONNX Runtime, no
//! network: it builds positions, rasterizes them, and prints where the time
//! went. It should run anywhere `cargo build` does.
//!
//! ## What it measures
//!
//! Positions come from a lattice at a chosen radius and stone count, because
//! both matter and neither is optional. Half the games a run generates are
//! 38-unit boards playing to 312 plies, so the positions that dominate carry
//! well over a hundred stones -- benchmarking at 28 on an 18-unit board is the
//! mistake that misled three separate estimates during this work.
//!
//! Four phases, timed separately:
//!
//!   legal set   marking which grid cells a stone could still be played on
//!   transform   the Euclidean distance transform over that mask
//!   settled     both of the above plus the dispatch around them
//!   raster      everything: settled, the stone sweeps, ridge, and packing
//!
//! The first three are phases of the standalone `settled` builder, which the
//! dense writer calls. The packed writer classifies settled row by row from the
//! minima its own sweep already computes, so `raster - settled` is not the sweep
//! and output loop -- those have no entry point of their own to measure.
//!
//! ## Verifying a change
//!
//! `--verify` checks the packed writer against the dense one for every
//! configuration, bit for bit: every bit plane, every fp16 ridge value, every
//! scalar. Any optimization to either has to keep that passing. It is not a
//! tolerance check -- the two must agree exactly -- because this raster is a
//! network input, and a change that quietly perturbs it produces a model that
//! is slightly wrong about positions in a way nothing downstream would flag.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p vgo-raster --bin vgo-raster-bench
//! cargo run --release -p vgo-raster --bin vgo-raster-bench -- --verify
//! cargo run --release -p vgo-raster --bin vgo-raster-bench -- --json
//! cargo run --release -p vgo-raster --bin vgo-raster-bench -- --stones 240 --units 38 --size 256
//! ```
//!
//! Timings are medians of `--rounds` repetitions. Run it on an idle machine:
//! this measures a few hundred microseconds, and anything else using the cache
//! will show up as noise larger than most of the wins worth having.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_raster::packed::{PackedRaster, rasterize_compact_radius_packed_into};
use vgo_raster::edt::{sampled_legal_set, squared_distance_transform};
use vgo_raster::{
    RasterConfig, RasterKind, rasterize_compact_radius_into, settled_for_raster,
};

/// A playable lattice of `count` stones at `radius`.
///
/// Spaced at 2.2 radii: stones must be at least a diameter apart to be legal,
/// and packing them tighter than that is not a position the rules allow.
fn lattice(count: usize, radius: f64) -> Position {
    if count == 0 {
        return Position::new(radius, Vec::new(), Color::White).with_komi(0.104);
    }
    let step = 2.2 * radius;
    let mut stones = Vec::new();
    let mut placed = 0usize;
    'outer: for row in 0..64 {
        for column in 0..64 {
            let x = 0.04 + step * f64::from(column);
            let y = 0.04 + step * f64::from(row);
            if x > 0.97 || y > 0.97 {
                continue;
            }
            stones.push(Stone::new(
                x,
                y,
                if placed % 2 == 0 { Color::Black } else { Color::White },
            ));
            placed += 1;
            if placed == count {
                break 'outer;
            }
        }
    }
    Position::new(radius, stones, Color::White).with_komi(0.104)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn time<F: FnMut()>(rounds: usize, mut body: F) -> f64 {
    body(); // warm
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let started = Instant::now();
        body();
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    median(samples)
}

struct Row {
    units: usize,
    stones: usize,
    size: usize,
    legal: f64,
    transform: f64,
    settled: f64,
    raster: f64,
    dense: f64,
}

fn measure(units: usize, stones: usize, size: usize, rounds: usize) -> Row {
    let radius = 1.0 / units as f64;
    let position = lattice(stones, radius);
    let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
    let pixels = config.pixels();

    let legal = time(rounds, || {
        black_box(sampled_legal_set(&position, size, size));
    });
    let mask = sampled_legal_set(&position, size, size);
    let transform = time(rounds, || {
        black_box(squared_distance_transform(&mask, size, size));
    });
    let settled = time(rounds, || {
        black_box(settled_for_raster(&position, config));
    });

    let mut packed = PackedRaster::new(config);
    let raster = time(rounds, || {
        rasterize_compact_radius_packed_into(&position, config, &mut packed);
    });
    let mut dense_buffer = vec![0.0f32; config.channels() * pixels];
    let dense = time(rounds, || {
        rasterize_compact_radius_into(&position, config, &mut dense_buffer);
    });

    Row { units, stones, size, legal, transform, settled, raster, dense }
}

/// Bit-for-bit agreement between the packed writer and the dense one.
fn verify(units: usize, stones: usize, size: usize) -> Result<(), String> {
    let radius = 1.0 / units as f64;
    let position = lattice(stones, radius);
    let config = RasterConfig::square_of(size, RasterKind::CompactRadius);
    let pixels = config.pixels();

    let mut dense = vec![0.0f32; config.channels() * pixels];
    rasterize_compact_radius_into(&position, config, &mut dense);
    let mut packed = PackedRaster::new(config);
    rasterize_compact_radius_packed_into(&position, config, &mut packed);

    let layout = packed.layout();
    for (slot, &channel) in layout.binary.iter().enumerate() {
        for pixel in 0..pixels {
            let expected = dense[channel * pixels + pixel] != 0.0;
            if packed.binary_pixel(slot, pixel) != expected {
                return Err(format!(
                    "{units}u {stones} stones {size}px: binary channel {channel}, pixel {pixel}"
                ));
            }
        }
    }
    for (slot, &channel) in layout.continuous.iter().enumerate() {
        for pixel in 0..pixels {
            let expected = half::f16::from_f32(dense[channel * pixels + pixel]);
            let actual = packed.dense()[slot * pixels + pixel];
            if actual != expected {
                return Err(format!(
                    "{units}u {stones} stones {size}px: channel {channel}, pixel {pixel}: \
                     {actual} against {expected}"
                ));
            }
        }
    }
    for (slot, &channel) in layout.scalar.iter().enumerate() {
        let expected = half::f16::from_f32(dense[channel * pixels]);
        if packed.scalars()[slot] != expected {
            return Err(format!("{units}u {stones} stones {size}px: scalar channel {channel}"));
        }
    }
    Ok(())
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let flag = |name: &str| arguments.iter().any(|a| a == name);
    let value = |name: &str, fallback: &str| -> String {
        arguments
            .iter()
            .position(|a| a == name)
            .and_then(|i| arguments.get(i + 1))
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    };
    let list = |name: &str, fallback: &str| -> Vec<usize> {
        value(name, fallback)
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect()
    };

    if flag("--help") || flag("-h") {
        println!("{}", include_str!("raster_bench_help.txt"));
        return;
    }

    let rounds: usize = value("--rounds", "40").parse().unwrap_or(40);
    let sizes = list("--size", "256");
    let units_list = list("--units", "18,38");
    let stones_list = list("--stones", "0,28,60,120,240");
    let json = flag("--json");

    if flag("--verify") {
        let mut failures = 0;
        for &size in &[64usize, 128, 256] {
            for &units in &units_list {
                for &stones in &stones_list {
                    // A board only holds so many stones; skip what will not fit.
                    if lattice(stones, 1.0 / units as f64).stones().len() < stones {
                        continue;
                    }
                    match verify(units, stones, size) {
                        Ok(()) => {}
                        Err(why) => {
                            println!("  FAIL  {why}");
                            failures += 1;
                        }
                    }
                }
            }
        }
        if failures == 0 {
            println!("  packed and dense agree bit for bit on every configuration");
        } else {
            println!("  {failures} configurations disagree");
            std::process::exit(1);
        }
        return;
    }

    let mut rows = Vec::new();
    for &size in &sizes {
        for &units in &units_list {
            for &stones in &stones_list {
                if lattice(stones, 1.0 / units as f64).stones().len() < stones {
                    continue;
                }
                rows.push(measure(units, stones, size, rounds));
            }
        }
    }

    if json {
        println!("[");
        for (index, r) in rows.iter().enumerate() {
            let comma = if index + 1 == rows.len() { "" } else { "," };
            println!(
                "  {{\"units\":{},\"stones\":{},\"size\":{},\"legal_ms\":{:.4},\
                 \"transform_ms\":{:.4},\"settled_ms\":{:.4},\"packed_ms\":{:.4},\
                 \"dense_ms\":{:.4}}}{comma}",
                r.units, r.stones, r.size, r.legal, r.transform, r.settled, r.raster, r.dense
            );
        }
        println!("]");
        return;
    }

    println!(
        "  {:>5} {:>7} {:>5} | {:>8} {:>9} {:>8} | {:>8} {:>8}",
        "units", "stones", "px", "legal", "transform", "settled", "packed", "dense"
    );
    println!("  {}", "-".repeat(77));
    for r in &rows {
        println!(
            "  {:>5} {:>7} {:>5} | {:>8.3} {:>9.3} {:>8.3} | {:>8.3} {:>8.3}",
            r.units,
            r.stones,
            r.size,
            r.legal,
            r.transform,
            r.settled,
            r.raster,
            r.dense
        );
    }
    println!();
    println!("  milliseconds, median of {rounds}. The three phase columns are the");
    println!("  standalone `settled` builder, which is what the *dense* writer calls.");
    println!("  The packed writer no longer calls it: it classifies settled row by row");
    println!("  from the minima its own sweep computes, so the phases do not sum to");
    println!("  `packed` and subtracting them from it does not give the sweep.");
    println!("  Run --verify after any change -- with the two writers on different");
    println!("  settled implementations it now covers settled too.");
}
