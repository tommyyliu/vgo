//! Paired, deterministic MCTS memory/latency experiment, no neural inference.
use std::time::Instant;
use vgo_core::{Color, Point, Position, Stone};
use vgo_raster::{DensePolicy, RasterConfig};
use vgo_search::{
    Action, Evaluation, FineGrid, HeapAllocation, Policy, SearchConfig, SteppedSearch,
};

struct TestPolicy {
    inner: DensePolicy,
    width: usize,
    shared: bool,
}
impl Policy for TestPolicy {
    fn logit(&self, action: Action) -> f64 {
        self.inner.logit(action)
    }
    fn heap_allocations(&self) -> Option<Vec<HeapAllocation>> {
        self.inner.heap_allocations()
    }
    fn fine_grid(&self, position: &Position, coarse: usize) -> Option<FineGrid> {
        if self.shared {
            self.inner.fine_grid(position, coarse)
        } else {
            Some(FineGrid::build(
                position,
                self.width,
                self.width,
                coarse,
                |r, c| {
                    self.inner.logit(Action::Place(Point::new(
                        (c as f64 + 0.5) / self.width as f64,
                        (r as f64 + 0.5) / self.width as f64,
                    ))) as f32
                },
            ))
        }
    }
}

fn main() {
    let reps = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(3);
    assert!(reps > 0);
    println!(
        "simulations,batch,width,variant,nodes,node_bytes,edge_bytes,stone_bytes,policy_bytes,sampling_bytes,shared_bytes_avoided,tracked_bytes,median_ms"
    );
    for simulations in [64, 256] {
        for batch in [1, 8] {
            let mut config = SearchConfig::canary(simulations);
            config.leaf_batch = batch;
            config.coarse_pool = 8;
            config.root_exploration_noise = 0.25;
            let stones = (0..20)
                .map(|i| {
                    Stone::new(
                        0.15 + (i % 5) as f64 * 0.15,
                        0.15 + (i / 5) as f64 * 0.15,
                        if i % 2 == 0 {
                            Color::Black
                        } else {
                            Color::White
                        },
                    )
                })
                .collect();
            let parent = Position::new(1.0 / 38.0, stones, Color::Black).with_komi(0.104);
            let width = 128;
            let mut expected = None;
            let mut times = [Vec::new(), Vec::new()];
            let mut memories = [vgo_search::TreeMemory::default(); 2];
            for round in 0..reps {
                for offset in 0..2 {
                    let variant = (round + offset) % 2;
                    let start = Instant::now();
                    let mut search = SteppedSearch::new(parent.clone(), config, 73, 0);
                    while !search.finished() {
                        let evaluations = search
                            .next_batch()
                            .iter()
                            .map(|p| {
                                let logits = (0..width * width + 1)
                                    .map(|i| ((i * 17 + p.stones().len() * 13) % 101) as f32 / 17.0)
                                    .collect();
                                Evaluation::new(
                                    0.0,
                                    Box::new(TestPolicy {
                                        inner: DensePolicy::new(
                                            RasterConfig::square(width),
                                            logits,
                                        ),
                                        width,
                                        shared: variant == 1,
                                    }),
                                )
                            })
                            .collect();
                        if !search.finished() {
                            search.submit(evaluations).unwrap();
                        }
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let memory = search.tree_memory();
                    assert_eq!(memory.unreported_policies, 0);
                    assert_eq!(memory.unreported_candidate_caches, 0);
                    let result = format!("{:?}", search.finish().unwrap());
                    if let Some(expected) = &expected {
                        assert_eq!(expected, &result, "search behavior changed");
                    } else {
                        expected = Some(result);
                    }
                    times[variant].push(elapsed);
                    memories[variant] = memory;
                }
            }
            assert_eq!(memories[0].nodes, memories[1].nodes);
            assert!(memories[1].tracked_bytes() < memories[0].tracked_bytes());
            for variant in 0..2 {
                times[variant].sort_by(f64::total_cmp);
                let m = memories[variant];
                println!(
                    "{simulations},{batch},{width},{},{},{},{},{},{},{},{},{},{:.3}",
                    if variant == 0 { "copied" } else { "shared" },
                    m.nodes,
                    m.node_bytes,
                    m.edge_capacity_bytes,
                    m.position_bytes,
                    m.policy_bytes,
                    m.sampling_bytes,
                    m.shared_bytes_avoided,
                    m.tracked_bytes(),
                    times[variant][reps / 2]
                );
            }
        }
    }
    eprintln!(
        "gate: every paired full SearchResult matched, including noise, priors, proposals, visits, and batching"
    );
}
