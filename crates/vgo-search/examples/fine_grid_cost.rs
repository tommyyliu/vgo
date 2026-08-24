//! What a fine grid costs as the board fills.
//!
//!     cargo run --release -p vgo-search --example fine_grid_cost
//!
//! One of these is built per node, so its cost is multiplied by the simulation
//! count on every move. It walks every policy cell and asks `is_legal_placement`,
//! which is O(stones) -- so the whole build is O(cells * stones), and stones go
//! as 1/r^2 on a finer board.
use std::hint::black_box;
use std::time::Instant;

use vgo_core::{Color, Position, Stone};
use vgo_search::FineGrid;

fn fixture(count: usize, radius: f64) -> Position {
    let spacing = 2.0 * radius * 1.05;
    let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
    let mut stones = Vec::new();
    for index in 0..count {
        let (row, col) = (index / per_row, index % per_row);
        let x = radius + 0.02 + col as f64 * spacing;
        let y = radius + 0.02 + row as f64 * spacing;
        if x > 1.0 - radius || y > 1.0 - radius { break; }
        stones.push(Stone { x, y, color: if index % 2 == 0 { Color::Black } else { Color::White } });
    }
    Position::new(radius, stones, Color::Black)
}

fn main() {
    println!("{:>8} {:>7} {:>10} {:>12}", "radius", "stones", "ms/build", "vs mini");
    let mut base = 0.0f64;
    for (radius, want) in [(1.0/18.0, 28usize), (1.0/18.0, 52), (1.0/38.0, 120), (1.0/38.0, 330)] {
        let position = fixture(want, radius);
        let n = position.stones().len();
        let runs = if n > 200 { 3 } else { 20 };
        for _ in 0..2 { black_box(FineGrid::build(&position, 128, 128, 16, |_, _| 0.0)); }
        let t = Instant::now();
        for _ in 0..runs { black_box(FineGrid::build(&position, 128, 128, 16, |_, _| 0.0)); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        if base == 0.0 { base = ms; }
        println!("{:>8} {:>7} {:>10.2} {:>11.1}x", format!("1/{}", (1.0/radius).round()), n, ms, ms/base);
    }
}
