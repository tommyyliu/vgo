//! Where does the sampled dead zone disagree with the definition, and why?
//!
//!     cargo run --release -p vgo-raster --example dead_zone_probe
//!
//! `dead(x) <=> dist(x, L) > r`. The sampled legal set can only *miss* parts of
//! `L`, never invent them, so every disagreement should be a pixel called dead
//! that is really alive -- and the interesting question is whether raising the
//! sampling resolution removes them or whether they survive it, which would mean
//! the missed part of `L` has no area to sample at all.
use vgo_core::{Color, Point, Position, Stone, distance_to_legal_set, legal_set_vertices};
use vgo_raster::{RasterConfig, RasterKind, dead_zone_mask};

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

/// A perfect lattice at `gap` times the exclusion diameter: the adversarial
/// case, because every legal gap has the same sub-pixel width and none of them
/// contains a sample point.
fn lattice(count: usize, radius: f64, gap: f64) -> Position {
    let spacing = 2.0 * radius * gap;
    let per_row = ((0.86_f64 / spacing).floor() as usize).max(1);
    let mut stones = Vec::new();
    for index in 0..count {
        let (row, column) = (index / per_row, index % per_row);
        let x = 0.07 + (column as f64 + 0.5) * spacing;
        let y = 0.07 + (row as f64 + 0.5) * spacing;
        if x > 0.96 || y > 0.96 {
            break;
        }
        stones.push(Stone::new(x, y, if index % 2 == 0 { Color::Black } else { Color::White }));
    }
    Position::new(radius, stones, Color::Black).with_komi(0.104)
}

fn report(label: &str, position: &Position, config: RasterConfig, radius: f64) {
    let vertices = legal_set_vertices(position);
    let truth: Vec<bool> = (0..config.pixels())
        .map(|pixel| {
            let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
            let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
            distance_to_legal_set(position, Point::new(x, y), Some(&vertices)) > radius
        })
        .collect();
    for oversample in [1usize, 3, 5, 9] {
        let (dead, exact) = dead_zone_mask(position, config, oversample);
        let false_dead = (0..config.pixels()).filter(|p| dead[*p] && !truth[*p]).count();
        let false_alive = (0..config.pixels()).filter(|p| !dead[*p] && truth[*p]).count();
        println!("{label:>22} {oversample:>5} {:>8} {false_dead:>12} {false_alive:>12} {exact:>10}",
            false_dead + false_alive);
    }
}

fn main() {
    let radius = 1.0 / 18.0;
    let config = RasterConfig::square_of(128, RasterKind::Compact);
    println!("{:>7} {:>5} {:>8} {:>12} {:>12} {:>10}",
        "stones", "over", "wrong", "false-dead", "false-alive", "exact");
    for (count, seed) in [(8usize, 1u64), (28, 2), (52, 4)] {
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() {
            continue;
        }
        let stones = position.stones().len();
        let vertices = legal_set_vertices(&position);
        let truth: Vec<bool> = (0..config.pixels())
            .map(|pixel| {
                let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
                let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
                distance_to_legal_set(&position, Point::new(x, y), Some(&vertices)) > radius
            })
            .collect();
        for oversample in [1usize, 3, 5, 9] {
            let (dead, exact) = dead_zone_mask(&position, config, oversample);
            let false_dead = (0..config.pixels()).filter(|p| dead[*p] && !truth[*p]).count();
            let false_alive = (0..config.pixels()).filter(|p| !dead[*p] && truth[*p]).count();
            println!("{stones:>7} {oversample:>5} {:>8} {false_dead:>12} {false_alive:>12} {exact:>10}",
                false_dead + false_alive);
        }
    }
    println!();
    println!("{:>22} {:>5} {:>8} {:>12} {:>12} {:>10}",
        "lattice gap", "over", "wrong", "false-dead", "false-alive", "exact");
    for gap in [1.02_f64, 1.05, 1.08, 1.2, 1.5] {
        let position = lattice(64, radius, gap);
        if !position.validate().is_playable() || position.stones().is_empty() {
            continue;
        }
        let label = format!("{gap:.2}x ({} stones)", position.stones().len());
        report(&label, &position, config, radius);
    }

    println!();
    println!("false-dead: called dead, really alive (the sampled legal set missed something).");
    println!("false-alive: called alive, really dead. Should be zero: sampling cannot invent L.");
}
