//! Isolate the removed duplicate transition in Action::try_apply.
//! cargo run --release -p vgo-search --example action_cost
use std::{hint::black_box, time::Instant};
use vgo_core::{Color, MoveError, MoveResult, Position};
use vgo_search::{Action, generate_candidates};

fn previous(action: Action, position: &Position) -> Option<MoveResult> {
    if let Action::Place(p) = action {
        if matches!(
            vgo_core::place(position, p.x, p.y),
            Err(MoveError::SelfCapture)
        ) {
            return None;
        }
    }
    Some(action.apply(position))
}

fn main() {
    let mut p = Position::new(1.0 / 18.0, vec![], Color::Black);
    for ply in 0..40 {
        let moves = generate_candidates(&p, 32, 20260911 + ply);
        let action = moves
            .iter()
            .filter(|c| matches!(c.action, Action::Place(_)))
            .nth(ply as usize % 8)
            .unwrap()
            .action;
        p = action.apply(&p).position;
    }
    let actions: Vec<_> = generate_candidates(&p, 16, 43)
        .into_iter()
        .map(|c| c.action)
        .filter(|a| matches!(a, Action::Place(_)))
        .collect();
    assert!(!actions.is_empty());
    for &a in &actions {
        let old = previous(a, &p).unwrap();
        let new = a.try_apply(&p).unwrap();
        assert_eq!(old.position, new.position);
        assert_eq!(old.analysis.geometry, new.analysis.geometry);
        assert_eq!(old.events, new.events);
    }
    let mut old = Vec::new();
    let mut new = Vec::new();
    for round in 0..8 {
        for variant in [round % 2, 1 - round % 2] {
            let start = Instant::now();
            for _ in 0..100 {
                for &a in &actions {
                    black_box(if variant == 0 {
                        previous(a, black_box(&p))
                    } else {
                        a.try_apply(black_box(&p))
                    });
                }
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / (100 * actions.len()) as f64;
            if round > 0 {
                if variant == 0 {
                    old.push(us);
                } else {
                    new.push(us);
                }
            }
        }
    }
    old.sort_by(f64::total_cmp);
    new.sort_by(f64::total_cmp);
    println!("stones,candidates,previous_us,current_us,speedup");
    println!(
        "{},{},{:.3},{:.3},{:.3}",
        p.stones().len(),
        actions.len(),
        old[3],
        new[3],
        old[3] / new[3]
    );
}
