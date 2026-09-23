//! Active-path memory comparison; forward transitions still rebuild geometry.
use std::{hint::black_box, time::Instant};
use vgo_core::iteration_lab::ReversiblePosition;
use vgo_core::{Phase, Point, Position};
#[path = "iteration_lab.rs"]
#[allow(dead_code)]
mod fixtures;

fn main() {
    let reps = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(5);
    assert!(reps > 0);
    println!("fixture,start_stones,depth,variant,history_bytes,median_roundtrip_us");
    for (name, parent, _) in fixtures::corpus()
        .into_iter()
        .filter(|(name, _, _)| name.starts_with("play") && name.ends_with("ply100"))
    {
        let mut expected = parent.clone();
        let mut path = Vec::new();
        let mut checkpoints = Vec::new();
        let mut rng = 73u64;
        let mut random = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..32 {
            if expected.phase() == Phase::Finished {
                break;
            }
            let radius = expected.radius();
            let mut point = None;
            for _ in 0..128 {
                let p = Point::new(
                    radius + random() * (1.0 - 2.0 * radius),
                    radius + random() * (1.0 - 2.0 * radius),
                );
                if vgo_core::is_legal_placement(&expected, p.x, p.y) {
                    point = Some(p);
                    break;
                }
            }
            if point.is_none() {
                let vertices = vgo_core::legal_set_vertices(&expected);
                point = vertices
                    .get((random() * vertices.len() as f64) as usize)
                    .copied();
            }
            checkpoints.push(expected.clone());
            expected = match point {
                Some(p) => vgo_core::place(&expected, p.x, p.y),
                None => vgo_core::pass(&expected),
            }
            .unwrap()
            .position;
            path.push(point);
        }
        let mut times = [Vec::new(), Vec::new()];
        let mut bytes = [0, 0];
        for round in 0..reps {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                if variant == 0 {
                    let mut current = parent.clone();
                    let mut history = Vec::new();
                    for point in &path {
                        let next = match point {
                            Some(p) => vgo_core::place(&current, p.x, p.y),
                            None => vgo_core::pass(&current),
                        }
                        .unwrap()
                        .position;
                        history.push(std::mem::replace(&mut current, next));
                    }
                    bytes[variant] = history.capacity() * size_of::<Position>()
                        + history
                            .iter()
                            .map(|p| std::mem::size_of_val(p.stones()))
                            .sum::<usize>();
                    while let Some(old) = history.pop() {
                        current = old;
                    }
                    black_box(current);
                } else {
                    let mut current = ReversiblePosition::new(parent.clone());
                    for point in &path {
                        black_box(
                            match point {
                                Some(p) => current.place(p.x, p.y),
                                None => current.pass(),
                            }
                            .unwrap(),
                        );
                    }
                    bytes[variant] = current.undo_bytes();
                    current.rollback(0);
                    black_box(current);
                }
                times[variant].push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        // Untimed check of every restored ancestor, not just the final root.
        let mut state = ReversiblePosition::new(parent.clone());
        for point in &path {
            match point {
                Some(p) => state.place(p.x, p.y),
                None => state.pass(),
            }
            .unwrap();
        }
        assert_eq!(format!("{:?}", state.position()), format!("{expected:?}"));
        for ancestor in checkpoints.iter().rev() {
            assert!(state.undo());
            assert_eq!(format!("{:?}", state.position()), format!("{ancestor:?}"));
        }
        for variant in 0..2 {
            times[variant].sort_by(f64::total_cmp);
            println!(
                "{name},{},{},{},{},{:.3}",
                parent.stones().len(),
                path.len(),
                if variant == 0 { "snapshots" } else { "undo" },
                bytes[variant],
                times[variant][reps / 2]
            );
        }
    }
    eprintln!("gate: leaf positions and all restored ancestors matched");
}
