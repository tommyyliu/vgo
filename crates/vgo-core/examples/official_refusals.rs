//! How often do the official rules refuse a placement, and is that right?
//!
//!     cargo run --release -p vgo-core --example official_refusals
//!
//! `Ruleset::Official` rejects a move that takes only the mover's own stones.
//! Our capture predicate can only err one way -- every test in
//! `official_cell_is_alive` is a sufficient witness for alive, so a missed case
//! reports "dead" -- which means we can only ever *over*-capture. Over-capturing
//! the mover's own group turns a legal move into a refusal, and the search then
//! plays a game with fewer moves than the real one.
//!
//! This counts refusals over real-looking positions. A refusal rate near zero
//! says the mechanism is not in play; a large one says the search is being
//! starved.
use vgo_core::{
    Color, MoveError, Point, Position, Ruleset, Stone, legal_set_vertices,
    nearest_legal_placement, place, planar_length,
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
    println!("{:>7} {:>10} {:>10} {:>9} {:>12}", "stones", "positions", "moves", "refused", "refused %");

    for &count in &[10usize, 20, 30, 40, 50] {
        let (mut positions, mut moves, mut refused) = (0usize, 0usize, 0usize);
        for seed in 1..=120u64 {
            let position = fixture(count, radius, seed * 31 + count as u64);
            if !position.validate().is_playable() || position.stones().len() < count / 2 {
                continue;
            }
            let official = position.clone().with_ruleset(Ruleset::Official);
            positions += 1;
            for vertex in legal_set_vertices(&position) {
                let snapped = nearest_legal_placement(&position, Point::new(vertex.x, vertex.y));
                if !snapped.legal {
                    continue;
                }
                moves += 1;
                if matches!(
                    place(&official, snapped.point.x, snapped.point.y),
                    Err(MoveError::SelfCapture)
                ) {
                    refused += 1;
                }
            }
        }
        let pct = if moves == 0 { 0.0 } else { refused as f64 / moves as f64 * 100.0 };
        println!("{count:>7} {positions:>10} {moves:>10} {refused:>9} {pct:>11.2}%");
    }
    println!();
    println!("Refusals are legitimate when the move really would take only the mover's");
    println!("own stones. This does not separate those from over-capture; it bounds how");
    println!("much room there is for the mechanism to matter at all.");
}
