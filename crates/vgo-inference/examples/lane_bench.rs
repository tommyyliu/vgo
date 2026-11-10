//! The generator's inference call alone: N lanes, each its own session, calling
//! `infer` on a full batch back to back. No actors and no broker, so this is what the
//! Rust path can do on the card, to compare with a live generator's lanes.
//!
//!     source scripts/env/ort.sh
//!     cargo run --release -p vgo-inference --example lane_bench -- MODEL LANES SECONDS [deferred]
//!
//! With `pool`, the lanes run in the generator's executor pool, packing on a stager
//! thread unless `VGO_STAGED_PACKING=0`, and outputs are split on the main thread
//! as the broker does.
//!
//! With `deferred`, each lane hands its outputs to one shared thread to split, as the
//! broker does in the generator.
use std::{
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, Position, Stone};
use vgo_inference::{
    BatchService, InferenceInput, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind, packed::rasterize_packed};

const BATCH: usize = 32;

fn positions(seed: u64) -> Vec<Position> {
    // A spread of mid-game big-board positions from a fixed pseudo-random walk.
    let radius = 1.0 / 38.0;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..BATCH)
        .map(|n| {
            let count = 40 + n * 4;
            let stones = (0..count)
                .map(|i| {
                    Stone::new(
                        0.03 + 0.94 * next(),
                        0.03 + 0.94 * next(),
                        if i % 2 == 0 {
                            Color::Black
                        } else {
                            Color::White
                        },
                    )
                })
                .collect();
            Position::new(
                radius,
                stones,
                if n % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            )
            .with_komi(0.02)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(&args[1]);
    let lanes: usize = args[2].parse().expect("lanes");
    let seconds: f64 = args[3].parse().expect("seconds");
    let raster = RasterConfig::square_of(128, RasterKind::CompactRadius);
    if args.get(4).is_some_and(|a| a == "check") {
        return check(&model, raster);
    }
    if args.get(4).is_some_and(|a| a == "pool") {
        return pool(&model, raster, lanes, seconds);
    }
    let inputs: Vec<InferenceInput> = positions(0)
        .iter()
        .enumerate()
        .map(|(i, p)| InferenceInput::packed(i as u64, rasterize_packed(p, raster)))
        .collect();
    let deferred = args.get(4).is_some_and(|a| a == "deferred");
    let (finish_tx, finish_rx) =
        std::sync::mpsc::sync_channel::<vgo_inference::FinishBatch>(lanes * 2);
    let finisher = thread::spawn(move || {
        let mut nanos = 0u64;
        for finish in finish_rx {
            let started = Instant::now();
            finish().expect("finish");
            nanos += started.elapsed().as_nanos() as u64;
        }
        nanos
    });
    let barrier = Arc::new(Barrier::new(lanes));
    // ONNX Runtime rejects sessions loading in parallel ("domain already exist"), and
    // the generator loads them one at a time, so do the same.
    let loading = Arc::new(std::sync::Mutex::new(()));
    let handles: Vec<_> = (0..lanes)
        .map(|_| {
            let (inputs, model, barrier) = (inputs.clone(), model.clone(), Arc::clone(&barrier));
            let loading = Arc::clone(&loading);
            let finish_tx = finish_tx.clone();
            thread::spawn(move || {
                let guard = loading.lock().expect("load lock");
                let mut service = OnnxBatchService::load(&OnnxServiceConfig {
                    model,
                    raster,
                    policy: Some(RasterConfig::square(128)),
                    maximum_batch: BATCH,
                    provider: OnnxProvider::TensorRt,
                    device_id: 0,
                    fp16: true,
                    cache_directory: PathBuf::from("artifacts/onnx-cache"),
                })
                .expect("load");
                drop(guard);
                for _ in 0..50 {
                    service.infer(&inputs).expect("warm-up");
                }
                barrier.wait();
                let (mut batches, mut pack, mut run, mut out) = (0u64, 0u64, 0u64, 0u64);
                let stop = Instant::now() + Duration::from_secs_f64(seconds);
                let started = Instant::now();
                while Instant::now() < stop {
                    if deferred {
                        finish_tx
                            .send(service.infer_deferred(&inputs).expect("infer"))
                            .expect("finisher");
                    } else {
                        service.infer(&inputs).expect("infer");
                    }
                    let s = service.last_inference_stages();
                    pack += s.input_packing_nanoseconds;
                    run += s.session_run_nanoseconds;
                    out += s.output_materialization_nanoseconds;
                    batches += 1;
                }
                (batches, started.elapsed().as_secs_f64(), pack, run, out)
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("lane"))
        .collect();
    drop(finish_tx);
    let finisher_nanos = finisher.join().expect("finisher");
    let elapsed = results.iter().map(|r| r.1).fold(0.0, f64::max);
    let batches: u64 = results.iter().map(|r| r.0).sum();
    let ms = |total: u64| total as f64 / batches as f64 / 1e6;
    println!(
        "lanes {lanes}{}: {:.0} positions/s | per batch: pack {:.3} ms, session {:.3} ms, materialize on lane {:.3} ms, on finisher {:.3} ms",
        if deferred { " deferred" } else { "" },
        batches as f64 * BATCH as f64 / elapsed,
        ms(results.iter().map(|r| r.2).sum()),
        ms(results.iter().map(|r| r.3).sum()),
        ms(results.iter().map(|r| r.4).sum()),
        ms(finisher_nanos),
    );
}

fn load(model: &std::path::Path, raster: RasterConfig) -> OnnxBatchService {
    OnnxBatchService::load(&OnnxServiceConfig {
        model: model.to_path_buf(),
        raster,
        policy: Some(RasterConfig::square(128)),
        maximum_batch: BATCH,
        provider: OnnxProvider::TensorRt,
        device_id: 0,
        fp16: true,
        cache_directory: PathBuf::from("artifacts/onnx-cache"),
    })
    .expect("load")
}

/// The generator's executor pool, kept full: two batches per lane, as the broker does.
fn pool(model: &std::path::Path, raster: RasterConfig, lanes: usize, seconds: f64) {
    use vgo_inference::{InferenceBatch, ThreadedBatchExecutorPool};
    let inputs: Vec<InferenceInput> = positions(0)
        .iter()
        .enumerate()
        .map(|(i, p)| InferenceInput::packed(i as u64, rasterize_packed(p, raster)))
        .collect();
    let services: Vec<_> = (0..lanes).map(|_| load(model, raster)).collect();
    let mut pool = ThreadedBatchExecutorPool::spawn(services).expect("pool");
    let mut sequence = 0u64;
    let mut run = |pool: &mut ThreadedBatchExecutorPool<OnnxBatchService>, until: Instant| {
        let (mut batches, mut pack, mut session, mut lane, mut split) =
            (0u64, 0u64, 0u64, 0u64, 0u64);
        while Instant::now() < until {
            while pool.in_flight() < pool.capacity() {
                pool.submit(InferenceBatch::new(sequence, inputs.clone()).expect("batch"))
                    .expect("submit");
                sequence += 1;
            }
            let done = pool.receive().expect("receive");
            let stages = done.stages();
            pack += stages.input_packing_nanoseconds;
            session += stages.session_run_nanoseconds;
            lane += done.elapsed().as_nanos() as u64;
            let (outputs, took) = done.finish().expect("finish");
            assert_eq!(outputs.len(), BATCH);
            split += took.as_nanos() as u64;
            batches += 1;
        }
        (batches, pack, session, lane, split)
    };
    run(&mut pool, Instant::now() + Duration::from_secs(5));
    let started = Instant::now();
    let (batches, pack, session, lane, split) =
        run(&mut pool, started + Duration::from_secs_f64(seconds));
    let elapsed = started.elapsed().as_secs_f64();
    let ms = |total: u64| total as f64 / batches as f64 / 1e6;
    println!(
        "pool lanes {lanes} staged={}: {:.0} positions/s | per batch: pack {:.3} ms, session {:.3} ms, lane busy {:.3} ms, split {:.3} ms",
        std::env::var("VGO_STAGED_PACKING").unwrap_or_else(|_| "1".into()),
        batches as f64 * BATCH as f64 / elapsed,
        ms(pack),
        ms(session),
        ms(lane),
        ms(split),
    );
}

/// The pinned path against the ordinary one: same inputs, outputs must match.
fn check(model: &std::path::Path, raster: RasterConfig) {
    let load = |pinned: bool| {
        // SAFETY: single-threaded here; the service reads the switch while loading.
        unsafe { std::env::set_var("VGO_PINNED_IO", if pinned { "1" } else { "0" }) };
        OnnxBatchService::load(&OnnxServiceConfig {
            model: model.to_path_buf(),
            raster,
            policy: Some(RasterConfig::square(128)),
            maximum_batch: BATCH,
            provider: OnnxProvider::TensorRt,
            device_id: 0,
            fp16: true,
            cache_directory: PathBuf::from("artifacts/onnx-cache"),
        })
        .expect("load")
    };
    let mut plain = load(false);
    let mut pinned = load(true);
    let mut stager = pinned
        .stager()
        .expect("a pinned packed service has a stager");
    let mut worst = (0.0f32, 0.0f64);
    for (seed, size) in [(1, 32), (2, 32), (3, 7), (4, 32), (5, 1), (6, 32), (7, 32)] {
        let inputs: Vec<InferenceInput> = positions(seed)[..size]
            .iter()
            .enumerate()
            .map(|(i, p)| {
                InferenceInput::packed(seed * 100 + i as u64, rasterize_packed(p, raster))
            })
            .collect();
        let expected = plain.infer(&inputs).expect("plain");
        let deferred = pinned.infer_deferred(&inputs).expect("pinned")().expect("finish");
        let ids = inputs.iter().map(InferenceInput::id).collect();
        let staged_inputs = stager(&inputs).expect("stage");
        let staged = pinned.infer_staged(staged_inputs, ids).expect("staged")().expect("finish");
        for got in [&deferred, &staged] {
            assert_eq!(got.len(), expected.len());
            for (a, b) in got.iter().zip(&expected) {
                assert_eq!(a.id(), b.id());
                let policy = a
                    .policy()
                    .iter()
                    .zip(b.policy())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f32::max);
                worst = (
                    worst.0.max(policy),
                    worst.1.max((a.current_value() - b.current_value()).abs()),
                );
            }
        }
        println!(
            "seed {seed} batch {size}: worst policy diff so far {:.2e}, value {:.2e}",
            worst.0, worst.1
        );
    }
    assert!(
        worst.0 < 1e-3 && worst.1 < 1e-3,
        "pinned outputs differ: {worst:?}"
    );
    println!("pinned and staged paths match");
}
