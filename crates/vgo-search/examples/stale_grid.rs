//! Narrower question: can a candidate be legal when the grid is built and
//! illegal on the SAME position later? That is the only shape consistent with
//! both mcts.rs apply sites, which each use their own node's position.
//!
//! Suspect: `is_legal_placement` is a pure geometric test, but `place()` also
//! requires `position.validate().is_playable()`. A position that fails
//! validation rejects EVERY placement -- so a grid built on a board that later
//! fails validation hands out candidates that all panic.
use vgo_core::{Color, Position, Stone, is_legal_placement, place};

fn main() {
    let radius = 0.055_714_285_714_285_716_f64;
    let mut state = 0x5150_ABCD_1234_9999_u64;
    let mut next = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut boards, mut unplayable, mut contains_ok_place_err) = (0usize, 0usize, 0usize);
    for _ in 0..400 {
        let mut stones: Vec<Stone> = Vec::new();
        let mut tries = 0;
        while stones.len() < 30 && tries < 60_000 {
            tries += 1;
            let x = radius + next() * (1.0 - 2.0 * radius);
            let y = radius + next() * (1.0 - 2.0 * radius);
            if stones.iter().all(|s: &Stone| {
                ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt() >= 2.0 * radius
            }) {
                let c = if stones.len() % 2 == 0 { Color::Black } else { Color::White };
                stones.push(Stone::new(x, y, c));
            }
        }
        if stones.len() < 25 { continue; }
        let position = Position::new(radius, stones, Color::Black);
        boards += 1;
        if !position.validate().is_playable() { unplayable += 1; }
        // Scan the board for points contains() accepts but place() refuses.
        for i in 0..40 {
            for j in 0..40 {
                let x = (i as f64 + 0.5) / 40.0;
                let y = (j as f64 + 0.5) / 40.0;
                if is_legal_placement(&position, x, y) && place(&position, x, y).is_err() {
                    contains_ok_place_err += 1;
                }
            }
        }
    }
    println!("boards tested                        : {boards}");
    println!("boards failing validate().is_playable: {unplayable}");
    println!("contains() ok but place() errors     : {contains_ok_place_err}");
}
