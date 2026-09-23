//! Linux process-level peak RAM probe. All actor trees stay alive together.
//! Synthetic dense evaluation, no model/GPU/raster tensor allocation.
use std::{
    fs,
    hash::{Hash, Hasher},
    sync::{Arc, Barrier, mpsc},
    thread,
    time::Instant,
};
use vgo_core::{Color, Position, Stone};
use vgo_raster::{DensePolicy, RasterConfig};
use vgo_search::{Evaluation, SearchConfig, SteppedSearch};

fn status_kib(field: &str) -> u64 {
    let text = fs::read_to_string("/proc/self/status").expect("Linux /proc required");
    text.lines()
        .find_map(|line| line.strip_prefix(field))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

// Diagnostic reader for the current v8 replay format; reject other layouts.
// Replay records do not encode a ruleset: this probe evaluates our Vgo rules.
fn replay_position(path: &str, sample: usize) -> (Position, u32) {
    let blob = fs::read(path).expect("read replay");
    let u32_at = |at| u32::from_le_bytes(blob[at..at + 4].try_into().unwrap()) as usize;
    let f64_at = |at| f64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
    assert_eq!(&blob[..8], b"VGORPLY1");
    assert_eq!(u32_at(8), 8, "only v8 replay supported");
    let samples = u32_at(12);
    assert!(sample < samples);
    let stride = 26 + u32_at(36) * 17 + 4 + u32_at(32) * 12 + 28;
    assert_eq!(blob.len(), 40 + samples * stride);
    let base = 40 + sample * stride;
    assert_eq!(blob[base + 21], 0, "sample must still be playing");
    let count = u32_at(base + 22);
    assert!(count <= u32_at(36));
    let color = |code| match code {
        0 => Color::Black,
        1 => Color::White,
        _ => panic!("invalid color"),
    };
    let stones = (0..count)
        .map(|i| {
            let at = base + 26 + i * 17;
            Stone::new(f64_at(at), f64_at(at + 8), color(blob[at + 16]))
        })
        .collect();
    let position = Position::new(f64_at(base), stones, color(blob[base + 16]))
        .with_komi(f64_at(base + 8))
        .with_passes(u32_at(base + 17) as u32);
    assert!(position.validate().is_playable());
    eprintln!(
        "replay sample={sample} stones={count} radius={} komi={} passes={}",
        position.radius(),
        position.komi(),
        position.consecutive_passes()
    );
    (position, u32_at(base + stride - 12) as u32)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let actors = args
        .get(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(1);
    let simulations = args
        .get(2)
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1400);
    let supports = args.iter().any(|s| s == "--supports");
    assert!(
        !supports || cfg!(feature = "iteration-lab"),
        "--supports requires --features iteration-lab"
    );
    let verify = args.iter().any(|s| s == "--verify-supports");
    assert!(!verify || supports, "--verify-supports requires --supports");
    let stone_count = args
        .windows(2)
        .find(|a| a[0] == "--stones")
        .map(|a| a[1].parse::<usize>().unwrap())
        .unwrap_or(20);
    assert!((1..=256).contains(&stone_count));
    let replay = args.windows(2).find(|a| a[0] == "--shard").map(|a| {
        let sample = args
            .windows(2)
            .find(|a| a[0] == "--sample")
            .expect("--shard requires --sample")[1]
            .parse()
            .unwrap();
        replay_position(&a[1], sample)
    });
    #[cfg(not(feature = "iteration-lab"))]
    let _ = verify;
    // A conservative admission estimate, NOT a runtime allocator limit.
    // Bound this diagnostic's scope and refuse low available system memory.
    assert!((1..=32).contains(&actors) && (1..=5600).contains(&simulations));
    // Initial sweeps measured ~70 KiB/node. Allow 128 KiB/node plus
    // 32 MiB/actor, and still refuse estimates above 8 GiB or 25% available RAM.
    let estimate = actors as u64 * ((u64::from(simulations) + 1) * 128 * 1024 + 32 * 1024 * 1024);
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap();
    let available = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024;
    assert!(
        estimate <= 8 * 1024 * 1024 * 1024 && estimate < available / 4,
        "probe exceeds conservative admission budget"
    );
    let start = Instant::now();
    let begin = Arc::new(Barrier::new(actors + 1));
    let release = Arc::new(Barrier::new(actors + 1));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    for actor in 0..actors {
        let begin = Arc::clone(&begin);
        let release = Arc::clone(&release);
        let tx = tx.clone();
        let replay = replay.clone();
        handles.push(thread::spawn(move || {
            let run = || {
                let mut config = SearchConfig::canary(simulations);
                config.leaf_batch = 4;
                config.coarse_pool = 16;
                config.maximum_candidates = 321;
                config.widening_coefficient = 6.0;
                config.temperature = 1.0;
                config.temperature_plies = 30;
                let stones = (0..stone_count)
                    .map(|i| {
                        let (x, y) = if stone_count <= 20 {
                            (0.15 + (i % 5) as f64 * 0.15, 0.15 + (i / 5) as f64 * 0.15)
                        } else {
                            ((1 + i % 16) as f64 / 17.0, (1 + i / 16) as f64 / 17.0)
                        };
                        Stone::new(
                            x,
                            y,
                            if i % 2 == 0 {
                                Color::Black
                            } else {
                                Color::White
                            },
                        )
                    })
                    .collect();
                let (parent, ply) = replay.unwrap_or_else(|| {
                    (
                        Position::new(1.0 / 38.0, stones, Color::Black).with_komi(0.02),
                        0,
                    )
                });
                let mut search = SteppedSearch::new(parent, config, 73 + actor as u64, ply);
                begin.wait();
                while !search.finished() {
                    let evaluations = search
                        .next_batch()
                        .iter()
                        .map(|p| {
                            let logits = (0..128 * 128 + 1)
                                .map(|i| ((i * 17 + p.stones().len() * 13) % 101) as f32 / 17.0)
                                .collect();
                            Evaluation::new(
                                0.0,
                                Box::new(DensePolicy::new(RasterConfig::square(128), logits)),
                            )
                        })
                        .collect();
                    if !search.finished() {
                        search.submit(evaluations).unwrap();
                    }
                }
                let memory = search.tree_memory();
                assert_eq!(memory.unreported_policies, 0);
                assert_eq!(memory.unreported_candidate_caches, 0);
                tx.send(memory).unwrap();
                // Do not let fast actors release trees while slower actors build.
                release.wait();
                let result = search.finish().unwrap();
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                format!("{result:?}").hash(&mut hash);
                hash.finish()
            };
            if supports {
                #[cfg(feature = "iteration-lab")]
                {
                    let (result, metrics) = vgo_search::transition_lab::with_support_backend(
                        vgo_search::transition_lab::SupportConfig {
                            byte_budget: 1024 * 1024,
                            verify,
                        },
                        run,
                    );
                    eprintln!("actor={actor} {metrics:?}");
                    return result;
                }
                #[cfg(not(feature = "iteration-lab"))]
                panic!("--supports requires --features iteration-lab");
            }
            run()
        }));
    }
    drop(tx);
    begin.wait();
    let reports: Vec<_> = (0..actors).map(|_| rx.recv().unwrap()).collect();
    assert_eq!(reports.len(), actors);
    let resident = status_kib("VmRSS:");
    let held_peak = status_kib("VmHWM:");
    let held_seconds = start.elapsed().as_secs_f64();
    release.wait();
    let signatures: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let peak = status_kib("VmHWM:").max(held_peak);
    println!(
        "{{\"actors\":{actors},\"simulations\":{simulations},\"nodes\":{},\"tracked_bytes\":{},\"resident_kib\":{resident},\"peak_kib\":{peak},\"build_seconds\":{held_seconds:.6},\"signatures\":{signatures:?}}}",
        reports.iter().map(|r| r.nodes).sum::<usize>(),
        reports.iter().map(|r| r.tracked_bytes()).sum::<usize>()
    );
}
