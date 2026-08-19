//! What does the connection field cost, and how much of it is real?
//!
//!     cargo run --release -p vgo-core --example connectivity_cost
//!
//! A raster channel has to be affordable at generation rates. The reference
//! implementation is cheap because most pairs never reach the geometry: they are
//! either too far to judge or close enough to be uncuttable outright. This
//! measures whether the same holds here, where the geometry step is a Voronoi
//! rebuild rather than an incremental zone query.
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{
    Color, CutKind, MAX_PAIR_CUT_DISTANCE, Position, SAFE_PAIR_DISTANCE, Stone, connected_pairs,
    pair_cut, planar_length,
};

fn fixture(count: usize, radius: f64, seed: u64) -> Position {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut stones: Vec<Stone> = Vec::new();
    let mut attempts = 0;
    while stones.len() < count && attempts < count * 400 {
        attempts += 1;
        let x = radius + next() * (1.0 - 2.0 * radius);
        let y = radius + next() * (1.0 - 2.0 * radius);
        if stones.iter().any(|s| planar_length(s.x - x, s.y - y) < 2.0 * radius) {
            continue;
        }
        let colour = if stones.len() % 2 == 0 { Color::Black } else { Color::White };
        stones.push(Stone::new(x, y, colour));
    }
    Position::new(radius, stones, Color::Black).with_komi(0.104)
}

fn main() {
    let radius = 1.0 / 18.0;
    println!(
        "{:>7} {:>7} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "stones", "pairs", "too far", "cheap", "geometry", "connected", "ms"
    );
    for &count in &[14usize, 20, 28, 40, 52] {
        let mut totals = [0usize; 4];
        let mut connected = 0usize;
        let mut seconds = 0.0;
        let mut positions = 0usize;
        for seed in 1..=25u64 {
            let position = fixture(count, radius, seed * 13 + count as u64);
            if !position.validate().is_playable() || position.stones().len() < count / 2 {
                continue;
            }
            positions += 1;
            let stones = position.stones().to_vec();
            for a in 0..stones.len() {
                for b in (a + 1)..stones.len() {
                    if stones[a].color != stones[b].color {
                        continue;
                    }
                    totals[0] += 1;
                    let d = planar_length(stones[a].x - stones[b].x, stones[a].y - stones[b].y);
                    if d > MAX_PAIR_CUT_DISTANCE * radius {
                        totals[1] += 1;
                    } else if d < SAFE_PAIR_DISTANCE * radius {
                        totals[2] += 1;
                    } else {
                        totals[3] += 1;
                    }
                }
            }
            let started = Instant::now();
            let pairs = connected_pairs(&position);
            seconds += started.elapsed().as_secs_f64();
            connected += pairs.len();
            black_box(&pairs);
            let _ = pair_cut(&position, 0, 1);
        }
        if positions == 0 {
            continue;
        }
        let per = |n: usize| n as f64 / positions as f64;
        println!(
            "{count:>7} {:>7.0} {:>9.0} {:>9.0} {:>9.1} {:>10.1} {:>10.3}",
            per(totals[0]), per(totals[1]), per(totals[2]), per(totals[3]),
            per(connected), seconds / positions as f64 * 1e3
        );
    }
    println!();
    println!("'geometry' is pairs that reach the two-placement search, which rebuilds");
    println!("the Voronoi diagram. Everything else is a distance test.");
    println!("Compare against the whole raster: about 0.37 ms per position.");
    let _ = CutKind::Connected;
}
