//! Support-point experiment using the exact same corpus as the iteration lab.
//! cargo run --release -p vgo-core --features iteration-lab --example support_lab -- 3
use std::{collections::BTreeMap, hint::black_box, time::Instant};
use vgo_core::iteration_lab::{self as lab, PreparedPosition, SupportPolicy, SupportPosition};
use vgo_core::{Point, Ruleset};

#[path = "iteration_lab.rs"]
#[allow(dead_code)]
mod fixtures;

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_secs_f64() * 1e6 / reps as f64
}

fn main() {
    let reps = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().expect("positive repetitions"))
        .unwrap_or(3);
    assert!(reps > 0);
    let filter = std::env::args().nth(2).unwrap_or_default();
    let corpus: Vec<_> = fixtures::corpus()
        .into_iter()
        .filter(|(name, _, _)| name.contains(&filter))
        .collect();
    assert!(!corpus.is_empty(), "no fixture matches");
    let mut checked = 0;
    for (_, parent, points) in &corpus {
        for ruleset in [Ruleset::Vgo, Ruleset::Official] {
            let parent = parent.clone().with_ruleset(ruleset);
            let all = SupportPosition::new(&parent, SupportPolicy::All);
            let four = SupportPosition::new(&parent, SupportPolicy::DiverseFour);
            let negative = SupportPosition::new(&parent, SupportPolicy::AllWithNegatives);
            for q in points
                .iter()
                .copied()
                .chain([Point::new(-1.0, 0.5), Point::new(f64::NAN, 0.5)])
            {
                let baseline = vgo_core::place(&parent, q.x, q.y);
                lab::assert_same(&baseline, &all.place(q.x, q.y));
                lab::assert_same(&baseline, &four.place(q.x, q.y));
                lab::assert_same(&baseline, &negative.place(q.x, q.y));
                checked += 1;
            }
        }
    }
    eprintln!("gate: {checked} full results matched per support strategy");
    println!(
        "fixture,stones,candidates,variant,prepare_us,move_us,amortized_us,speedup,points,dormant,supports,groups,certified,reactivated,negative_cells,eligible_negative_cells"
    );
    let mut totals: BTreeMap<(String, String), (usize, f64, f64)> = BTreeMap::new();
    for (name, parent, points) in corpus {
        if points.is_empty() {
            continue;
        }
        let legacy = PreparedPosition::new(&parent);
        let all = SupportPosition::new(&parent, SupportPolicy::All);
        let four = SupportPosition::new(&parent, SupportPolicy::DiverseFour);
        let negative = SupportPosition::new(&parent, SupportPolicy::AllWithNegatives);
        let prepare = [
            0.0,
            time(reps, || {
                black_box(PreparedPosition::new(&parent));
            }),
            time(reps, || {
                black_box(SupportPosition::new(&parent, SupportPolicy::All));
            }),
            time(reps, || {
                black_box(SupportPosition::new(&parent, SupportPolicy::DiverseFour));
            }),
            time(reps, || {
                black_box(SupportPosition::new(
                    &parent,
                    SupportPolicy::AllWithNegatives,
                ));
            }),
        ];
        let mut times: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..8 {
            for offset in 0..5 {
                let variant = (round + offset) % 5;
                let us = time(reps, || {
                    for q in &points {
                        black_box(match variant {
                            0 => vgo_core::place(black_box(&parent), q.x, q.y),
                            1 => legacy.place(q.x, q.y),
                            2 => all.place(q.x, q.y),
                            3 => four.place(q.x, q.y),
                            _ => negative.place(q.x, q.y),
                        })
                        .unwrap();
                    }
                }) / points.len() as f64;
                if round > 0 {
                    times[variant].push(us);
                }
            }
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        for (i, variant) in ["baseline", "legacy", "all", "four", "negative"]
            .iter()
            .enumerate()
        {
            let us = times[i][3];
            let amortized = us + prepare[i] / points.len() as f64;
            let (mut groups, mut certified, mut reactivated) = (0, 0, 0);
            let mut counts = (0, 0, 0);
            let (mut negative_count, mut negative_eligible) = (0, 0);
            if i >= 2 {
                let cache = match i {
                    2 => &all,
                    3 => &four,
                    _ => &negative,
                };
                negative_count = cache.negative_count();
                counts = (
                    cache.point_count(),
                    cache.dormant_count(),
                    cache.support_count(),
                );
                for q in &points {
                    let (_, stats) = cache.place_profiled(q.x, q.y);
                    groups += stats.groups;
                    certified += stats.certified_groups;
                    reactivated += stats.reactivated_points;
                    negative_eligible += stats.eligible_negative_cells;
                }
            }
            println!(
                "{name},{},{},{variant},{:.3},{us:.3},{amortized:.3},{:.3},{},{},{},{groups},{certified},{reactivated},{negative_count},{negative_eligible}",
                parent.stones().len(),
                points.len(),
                prepare[i],
                times[0][3] / amortized,
                counts.0,
                counts.1,
                counts.2
            );
            let kind = if name.starts_with("play") {
                "played"
            } else {
                "lattice"
            };
            let total = totals
                .entry((kind.into(), variant.to_string()))
                .or_default();
            total.0 += points.len();
            total.1 += times[0][3] * points.len() as f64;
            total.2 += amortized * points.len() as f64;
        }
    }
    for ((kind, variant), (n, b, t)) in totals {
        eprintln!(
            "{kind} {variant}: {n} moves, {:.3} us amortized, {:.3}x",
            t / n as f64,
            b / t
        );
    }
}
