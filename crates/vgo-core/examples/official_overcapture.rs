//! Does our official capture rule kill groups that are actually alive?
//!
//!     cargo run --release -p vgo-core --example official_overcapture
//!
//! `official_cell_is_alive` tests three sufficient witnesses: a cell vertex
//! within `r` of the legal set, a legal-set vertex inside the cell, and a
//! legal-set vertex within `r` of a cell edge. Each proves "alive", so a missed
//! case can only report "dead" -- the error is one-directional and every
//! instance is a group wrongly captured.
//!
//! The true condition is `dist(cell, L) <= r`, whose closest pair can be an edge
//! interior against a smooth arc of `L`'s boundary, with no vertex extremal on
//! either side. That is the case the three tests miss.
//!
//! This needs no reference implementation: sample each cell's boundary densely
//! and evaluate `dist(x, L)` exactly at every sample. Any cell the samples find
//! alive and the three tests call dead is an over-capture. Sampling can only
//! miss more, so this is a *lower bound* on the rate.
use vgo_core::{
    Analysis, Color, Point, Position, Ruleset, Stone, distance_to_legal_set,
    legal_set_vertices, planar_length,
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
    Position::new(radius, stones, Color::Black).with_komi(0.077)
}

fn main() {
    let radius = 1.0 / 18.0;
    let samples = 256;
    println!("{:>7} {:>10} {:>12} {:>14} {:>12}", "stones", "cells", "engine dead", "really alive", "over-capture");

    for &count in &[20usize, 30, 40, 50] {
        let (mut cells, mut dead, mut wrong) = (0usize, 0usize, 0usize);
        for seed in 1..=150u64 {
            let position = fixture(count, radius, seed * 17 + count as u64)
                .with_ruleset(Ruleset::Official);
            if !position.validate().is_playable() || position.stones().len() < count / 2 {
                continue;
            }
            let analysis = Analysis::new(&position);
            let vertices = legal_set_vertices(&position);
            for (index, cell) in analysis.geometry.cells.iter().enumerate() {
                if cell.polygon.len() < 3 {
                    continue;
                }
                cells += 1;
                let group = analysis.geometry.groups[index];
                let engine_dead = analysis.settled_groups.contains(&group);
                if !engine_dead {
                    continue;
                }
                dead += 1;
                // Dense boundary sweep against the exact continuous distance.
                let mut alive = false;
                'edges: for (a, b) in cell
                    .polygon
                    .iter()
                    .zip(cell.polygon.iter().cycle().skip(1))
                    .take(cell.polygon.len())
                {
                    for step in 0..samples {
                        let t = step as f64 / samples as f64;
                        let p = Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
                        if distance_to_legal_set(&position, p, Some(&vertices)) <= radius {
                            alive = true;
                            break 'edges;
                        }
                    }
                }
                if alive {
                    wrong += 1;
                }
            }
        }
        let pct = if dead == 0 { 0.0 } else { wrong as f64 / dead as f64 * 100.0 };
        println!("{count:>7} {cells:>10} {dead:>12} {wrong:>14} {pct:>11.2}%");
    }
    println!();
    println!("'over-capture' is cells the engine captured that a dense sweep of the");
    println!("boundary proves were still reachable. A lower bound: sampling misses more.");
}
