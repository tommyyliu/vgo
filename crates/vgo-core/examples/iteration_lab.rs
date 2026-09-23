//! Reproducible, paired full-move benchmark and differential correctness gate.
//! cargo run --release -p vgo-core --features iteration-lab --example iteration_lab -- 10
use std::{hint::black_box, time::Instant};
use vgo_core::iteration_lab::{self as lab, Strategy};
use vgo_core::{
    Analysis, Color, Point, Position, Ruleset, Stone, is_legal_placement, legal_set_vertices,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn candidates(position: &Position, rng: &mut Rng) -> Vec<Point> {
    let mut points = Vec::new();
    let r = position.radius();
    for _ in 0..3000 {
        let p = Point::new(
            r + rng.next() * (1.0 - 2.0 * r),
            r + rng.next() * (1.0 - 2.0 * r),
        );
        if is_legal_placement(position, p.x, p.y) {
            points.push(p);
            if points.len() == 8 {
                break;
            }
        }
    }
    // Analytic boundary placements exercise tangency and residual late-game gaps.
    points.extend(legal_set_vertices(position).into_iter().take(8));
    points
}

pub(crate) fn corpus() -> Vec<(String, Position, Vec<Point>)> {
    let mut out = Vec::new();
    for (i, r) in [1.0 / 10.0, 1.0 / 18.0, 1.0 / 38.0].into_iter().enumerate() {
        let mut rng = Rng(20260911 + i as u64);
        let mut position = Position::new(r, vec![], Color::Black).with_komi(0.104);
        for ply in 0..160 {
            let points = candidates(&position, &mut rng);
            if ply % 20 == 0 && !points.is_empty() {
                out.push((
                    format!("play-r{}-ply{ply}", i),
                    position.clone(),
                    points.clone(),
                ));
            }
            if points.is_empty() {
                break;
            }
            let p = points[(rng.next() * points.len() as f64) as usize];
            let result = vgo_core::place(&position, p.x, p.y).unwrap();
            position = result.position;
            if position.phase() == vgo_core::Phase::Finished {
                break;
            }
        }
    }
    // Valid setup diagrams deliberately stress density, isolated groups, and capture.
    for (side, r, spacing) in [(4, 0.1, 0.21), (8, 0.05, 0.115), (16, 0.025, 0.06)] {
        let stones = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| {
                    Stone::new(
                        r + x as f64 * spacing,
                        r + y as f64 * spacing,
                        if (x + y) % 2 == 0 {
                            Color::Black
                        } else {
                            Color::White
                        },
                    )
                })
            })
            .collect();
        let p = Position::new(r, stones, Color::Black);
        assert!(p.validate().is_playable());
        let points = candidates(&p, &mut Rng(71));
        out.push((format!("lattice-{side}"), p, points));
    }
    out
}

fn timed(mut operation: impl FnMut(), repetitions: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..repetitions {
        operation();
    }
    start.elapsed().as_secs_f64() * 1e6 / repetitions as f64
}

fn main() {
    let reps = std::env::args()
        .nth(1)
        .map(|v| v.parse::<usize>().expect("positive repetitions"))
        .unwrap_or(10);
    assert!(reps > 0);
    let filter = std::env::args().nth(2).unwrap_or_default();
    let corpus: Vec<_> = corpus()
        .into_iter()
        .filter(|(name, _, _)| name.contains(&filter))
        .collect();
    assert!(!corpus.is_empty(), "fixture filter matched nothing");
    let mut checked = 0;
    let mut captures = 0;
    let mut self_captures = 0;
    let mut no_ops = 0;
    let mut refusals = 0;
    // Correctness is outside the timing region. Include both rulesets and errors.
    for (_, position, points) in &corpus {
        for rules in [Ruleset::Vgo, Ruleset::Official] {
            let p = position.clone().with_ruleset(rules);
            let prepared = lab::PreparedPosition::new(&p);
            for q in points
                .iter()
                .copied()
                .chain([Point::new(-1.0, 0.5), Point::new(f64::NAN, 0.5)])
            {
                let expected = vgo_core::place(&p, q.x, q.y);
                lab::assert_same(&expected, &lab::place(&p, q.x, q.y, Strategy::WitnessFirst));
                lab::assert_same(&expected, &lab::place(&p, q.x, q.y, Strategy::Indexed));
                lab::assert_same(&expected, &lab::place(&p, q.x, q.y, Strategy::QueryFirst));
                lab::assert_same(&expected, &prepared.place(q.x, q.y));
                if let Ok(result) = &expected {
                    captures += usize::from(result.captured > 0);
                    self_captures += usize::from(
                        result
                            .events
                            .iter()
                            .any(|e| matches!(e, vgo_core::GameEvent::SelfCapture { .. })),
                    );
                    no_ops += usize::from(result.position.stones() == p.stones());
                } else {
                    refusals += 1;
                }
                checked += 1;
            }
        }
    }
    eprintln!(
        "gate: {checked} transitions matched; {captures} capture moves, {self_captures} self-captures, {no_ops} no-ops, {refusals} refusals (includes invalid coordinates)"
    );
    println!(
        "fixture,stones,moves,validate_us,geometry_us,legal_vertices_us,analysis_us,baseline_move_us,witness_move_us,indexed_move_us,query_move_us,prepare_us,certificates,cached_move_us,cached_amortized_us,cached_speedup"
    );
    let mut base_total = 0.0;
    let mut fast_total = 0.0;
    let mut indexed_total = 0.0;
    let mut query_total = 0.0;
    let mut cached_total = 0.0;
    let mut move_count = 0;
    for (name, p, points) in &corpus {
        let validate = timed(
            || {
                black_box(p.validate());
            },
            reps,
        );
        let geometry = timed(
            || {
                black_box(lab::geometry(p));
            },
            reps,
        );
        let legal = timed(
            || {
                black_box(legal_set_vertices(p));
            },
            reps,
        );
        let analysis = timed(
            || {
                black_box(Analysis::new(p));
            },
            reps,
        );
        if points.is_empty() {
            continue;
        }
        let prepare = timed(
            || {
                black_box(lab::PreparedPosition::new(p));
            },
            reps,
        );
        let prepared = lab::PreparedPosition::new(p);
        let mut base = Vec::new();
        let mut fast = Vec::new();
        let mut indexed = Vec::new();
        let mut query = Vec::new();
        let mut cached = Vec::new();
        for round in 0..7 {
            let strategies = [
                Strategy::Baseline,
                Strategy::WitnessFirst,
                Strategy::Indexed,
                Strategy::QueryFirst,
            ];
            for offset in 0..5 {
                let variant = (round + offset) % 5;
                let us = timed(
                    || {
                        for q in points {
                            black_box(if variant == 4 {
                                prepared.place(q.x, q.y)
                            } else {
                                lab::place(black_box(p), q.x, q.y, strategies[variant])
                            })
                            .unwrap();
                        }
                    },
                    reps,
                ) / points.len() as f64;
                match variant {
                    0 => base.push(us),
                    1 => fast.push(us),
                    2 => indexed.push(us),
                    3 => query.push(us),
                    4 => cached.push(us),
                    _ => unreachable!(),
                }
            }
        }
        base.sort_by(f64::total_cmp);
        fast.sort_by(f64::total_cmp);
        indexed.sort_by(f64::total_cmp);
        query.sort_by(f64::total_cmp);
        cached.sort_by(f64::total_cmp);
        let (b, f) = (base[3], fast[3]);
        let ix = indexed[3];
        let q = query[3];
        let c = cached[3];
        let amortized = c + prepare / points.len() as f64;
        println!(
            "{name},{},{},{validate:.3},{geometry:.3},{legal:.3},{analysis:.3},{b:.3},{f:.3},{ix:.3},{q:.3},{prepare:.3},{},{c:.3},{amortized:.3},{:.3}",
            p.stones().len(),
            points.len(),
            prepared.certificate_count(),
            b / amortized
        );
        base_total += b * points.len() as f64;
        fast_total += f * points.len() as f64;
        indexed_total += ix * points.len() as f64;
        query_total += q * points.len() as f64;
        cached_total += amortized * points.len() as f64;
        move_count += points.len();
    }
    eprintln!(
        "weighted median-batch move cost over {move_count} candidates: baseline {:.3} us, witness-first {:.3} us, {:.3}x",
        base_total / move_count as f64,
        fast_total / move_count as f64,
        base_total / fast_total
    );
    eprintln!(
        "indexed {:.3} us, {:.3}x",
        indexed_total / move_count as f64,
        base_total / indexed_total
    );
    eprintln!(
        "query-first {:.3} us, {:.3}x",
        query_total / move_count as f64,
        base_total / query_total
    );
    eprintln!(
        "cached including preparation amortized across each parent's candidates {:.3} us, {:.3}x",
        cached_total / move_count as f64,
        base_total / cached_total
    );
}
