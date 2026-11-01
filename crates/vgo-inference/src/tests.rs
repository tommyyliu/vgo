use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, Position};
use vgo_raster::{RasterConfig, rasterize};
use vgo_search::{Action, EvaluationError, Evaluator};

use super::*;
use crate::broker::{Broker, Request};
use crate::metrics::AtomicMetrics;

struct ConstantService {
    wrong_id: bool,
}

#[test]
fn stalled_pool_fails_every_waiting_caller_and_shuts_down() {
    struct Blocked(mpsc::Receiver<()>);
    impl BatchService for Blocked {
        fn contract(&self) -> BatchContract {
            test_contract(1)
        }
        fn infer(&mut self, _: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
            let _ = self.0.recv();
            Err(EvaluationError::new("released test backend"))
        }
    }
    let (release, gate) = mpsc::channel();
    let service = Blocked(gate);
    let contract = service.contract();
    let executor =
        ThreadedBatchExecutorPool::spawn_with_timeout(vec![service], Duration::from_millis(30))
            .unwrap();
    let (sender, receiver) = mpsc::sync_channel(8);
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let mut waiting = Vec::new();
    for id in 1..=5 {
        let (response, completion) = mpsc::sync_channel(1);
        sender
            .send(Request {
                inputs: vec![encode_request(id, &position, contract)],
                encoding_nanoseconds: 0,
                queued_at: Instant::now(),
                response,
            })
            .unwrap();
        waiting.push(completion);
    }
    drop(sender);
    let metrics = Arc::new(AtomicMetrics::default());
    let lanes = vec![Arc::new(AtomicMetrics::default())];
    let started = Instant::now();
    Broker::new(
        BrokerConfig {
            maximum_delay: Duration::from_millis(1),
            queue_capacity: 8,
        },
        contract,
        executor,
        receiver,
        metrics,
        lanes,
    )
    .run();
    assert!(started.elapsed() < Duration::from_secs(1));
    for completion in waiting {
        assert!(
            completion
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("batch 0, request IDs 1..1")
        );
    }
    release.send(()).unwrap();
}

struct FailingService;

struct StagedService;

impl BatchService for FailingService {
    fn contract(&self) -> BatchContract {
        test_contract(2)
    }

    fn infer(
        &mut self,
        _batch: &[InferenceInput],
    ) -> Result<Vec<InferenceOutput>, EvaluationError> {
        Err(EvaluationError::new("synthetic inference failure"))
    }
}

impl BatchService for StagedService {
    fn contract(&self) -> BatchContract {
        test_contract(2)
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        batch
            .iter()
            .map(|input| InferenceOutput::new(input.id(), 0.25, vec![0.0; 5]))
            .collect()
    }

    fn last_inference_stages(&self) -> InferenceStageMetrics {
        InferenceStageMetrics {
            input_packing_nanoseconds: 11,
            session_run_nanoseconds: 13,
            output_materialization_nanoseconds: 17,
        }
    }
}

impl BatchService for ConstantService {
    fn contract(&self) -> BatchContract {
        BatchContract {
            raster: RasterConfig::square(2),
            policy: RasterConfig::square(2),
            maximum_batch: 2,
            input_layout: InputLayout::Dense,
        }
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        batch
            .iter()
            .map(|input| {
                InferenceOutput::new(input.id() + u64::from(self.wrong_id), 0.25, vec![0.0; 5])
            })
            .collect()
    }
}

struct RecordingService {
    maximum_batch: usize,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl BatchService for RecordingService {
    fn contract(&self) -> BatchContract {
        BatchContract {
            raster: RasterConfig::square(2),
            policy: RasterConfig::square(2),
            maximum_batch: self.maximum_batch,
            input_layout: InputLayout::Dense,
        }
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        self.batch_sizes
            .lock()
            .expect("batch-size lock")
            .push(batch.len());
        batch
            .iter()
            .map(|input| InferenceOutput::new(input.id(), 0.25, vec![0.0; 5]))
            .collect()
    }
}

struct TaggedService {
    contract: BatchContract,
    current_value: f64,
    delay: Duration,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    drops: Arc<AtomicUsize>,
}

impl BatchService for TaggedService {
    fn contract(&self) -> BatchContract {
        self.contract
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        thread::sleep(self.delay);
        self.batch_sizes
            .lock()
            .expect("tagged batch-size lock")
            .push(batch.len());
        batch
            .iter()
            .map(|input| {
                InferenceOutput::new(
                    input.id(),
                    self.current_value,
                    vec![0.0; self.contract.policy.pixels() + 1],
                )
            })
            .collect()
    }
}

impl Drop for TaggedService {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

fn tagged_service(
    current_value: f64,
    contract: BatchContract,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    drops: Arc<AtomicUsize>,
) -> TaggedService {
    TaggedService {
        contract,
        current_value,
        delay: Duration::ZERO,
        batch_sizes,
        drops,
    }
}

fn test_contract(maximum_batch: usize) -> BatchContract {
    BatchContract {
        raster: RasterConfig::square(2),
        policy: RasterConfig::square(2),
        maximum_batch,
        input_layout: InputLayout::Dense,
    }
}

fn broker_config() -> BrokerConfig {
    BrokerConfig {
        maximum_delay: Duration::ZERO,
        queue_capacity: 2,
    }
}

/// A short batch must not go on deck: it would take positions a full batch was about
/// to use, for a slot that has no need of it yet.
#[test]
fn a_deck_takes_only_a_full_batch() {
    struct Recording {
        gate: Option<mpsc::Receiver<()>>,
        sizes: Arc<Mutex<Vec<usize>>>,
    }
    impl BatchService for Recording {
        fn contract(&self) -> BatchContract {
            test_contract(4)
        }
        fn infer(
            &mut self,
            inputs: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
            self.sizes.lock().unwrap().push(inputs.len());
            if let Some(gate) = self.gate.take() {
                let _ = gate.recv();
            }
            ConstantService { wrong_id: false }.infer(inputs)
        }
    }
    let (release, gate) = mpsc::channel();
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let service = Recording {
        gate: Some(gate),
        sizes: Arc::clone(&sizes),
    };
    let contract = service.contract();
    let executor = ThreadedBatchExecutorPool::spawn(vec![service]).unwrap();
    let (sender, receiver) = mpsc::sync_channel(8);
    let broker = Broker::new(
        BrokerConfig {
            maximum_delay: Duration::from_millis(1),
            queue_capacity: 8,
        },
        contract,
        executor,
        receiver,
        Arc::new(AtomicMetrics::default()),
        vec![Arc::new(AtomicMetrics::default())],
    );
    let serving = thread::spawn(move || broker.run());

    let position = Position::new(0.1, Vec::new(), Color::Black);
    let mut next_id = 0;
    let mut send = |count: usize| {
        let inputs = (0..count)
            .map(|_| {
                next_id += 1;
                encode_request(next_id, &position, contract)
            })
            .collect();
        let (response, completion) = mpsc::sync_channel(1);
        sender
            .send(Request {
                inputs,
                encoding_nanoseconds: 0,
                queued_at: Instant::now(),
                response,
            })
            .unwrap();
        completion
    };

    // A full batch to hold the slot, then fewer than a batch behind it, with far
    // longer than the batching delay to go on deck if it were going to.
    let mut completions = vec![send(4)];
    while sizes.lock().unwrap().is_empty() {
        thread::sleep(Duration::from_millis(1));
    }
    completions.extend([send(1), send(1), send(1)]);
    thread::sleep(Duration::from_millis(50));
    completions.push(send(1));
    release.send(()).unwrap();
    for completion in completions {
        assert!(
            completion
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .is_ok()
        );
    }
    drop(sender);
    serving.join().unwrap();
    assert_eq!(*sizes.lock().unwrap(), [4, 4]);
}

#[test]
fn batched_evaluator_adapts_a_batch_service() {
    let evaluator =
        BatchedEvaluatorPool::spawn(broker_config(), vec![ConstantService { wrong_id: false }])
            .unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let evaluation = evaluator.evaluate(&position).unwrap();
    assert_eq!(evaluation.current_value, 0.25);
    assert_eq!(evaluation.policy_logit(Action::Pass), 0.0);
    assert_eq!(evaluator.metrics().requests, 1);
}

#[test]
fn evaluate_batch_reassembles_a_group_larger_than_the_backend_batch() {
    let evaluator =
        BatchedEvaluatorPool::spawn(broker_config(), vec![ConstantService { wrong_id: false }])
            .unwrap();
    let positions = [
        Position::new(0.1, Vec::new(), Color::Black),
        Position::new(0.1, Vec::new(), Color::White),
        Position::new(0.1, Vec::new(), Color::Black),
    ];

    let evaluations = evaluator.evaluate_batch(&positions).unwrap();

    assert_eq!(evaluations.len(), 3);
    assert!(
        evaluations
            .iter()
            .all(|evaluation| evaluation.current_value == 0.25)
    );
    let metrics = evaluator.metrics();
    assert_eq!(metrics.requests, 3);
    assert_eq!(metrics.batches, 2);
    assert_eq!(metrics.positions, 3);
    assert_eq!(metrics.maximum_batch, 2);
    assert_eq!(metrics.failures, 0);
}

#[test]
fn broker_splits_nine_item_groups_across_full_sixteen_item_batches() {
    const GROUPS: usize = 4;
    const GROUP_SIZE: usize = 9;
    const MAXIMUM_BATCH: usize = 16;

    let contract = BatchContract {
        raster: RasterConfig::square(2),
        policy: RasterConfig::square(2),
        maximum_batch: MAXIMUM_BATCH,
        input_layout: InputLayout::Dense,
    };
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let service = RecordingService {
        maximum_batch: MAXIMUM_BATCH,
        batch_sizes: Arc::clone(&batch_sizes),
    };
    let metrics = Arc::new(AtomicMetrics::default());
    let (sender, receiver) = mpsc::sync_channel(GROUPS);
    let mut completions = Vec::with_capacity(GROUPS);

    for group in 0..GROUPS {
        let first_id = (group * GROUP_SIZE) as u64;
        let inputs = (0..GROUP_SIZE)
            .map(|offset| {
                let position = Position::new(0.1, Vec::new(), Color::Black);
                InferenceInput::new(
                    first_id + offset as u64,
                    rasterize(&position, RasterConfig::square(2)),
                )
            })
            .collect();
        let (response, completion) = mpsc::sync_channel(1);
        sender
            .send(Request {
                inputs,
                encoding_nanoseconds: 0,
                queued_at: Instant::now(),
                response,
            })
            .unwrap();
        completions.push((first_id, completion));
    }
    drop(sender);

    Broker::new(
        BrokerConfig {
            maximum_delay: Duration::from_millis(10),
            queue_capacity: GROUPS,
        },
        contract,
        ThreadedBatchExecutorPool::spawn(vec![service]).unwrap(),
        receiver,
        Arc::clone(&metrics),
        vec![Arc::new(AtomicMetrics::default())],
    )
    .run();

    assert_eq!(
        *batch_sizes.lock().expect("batch-size lock"),
        vec![16, 16, 4],
        "four 9-item caller groups should flatten into two full batches and one tail"
    );
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.batches, 3);
    assert_eq!(snapshot.positions, GROUPS as u64 * GROUP_SIZE as u64);
    assert_eq!(snapshot.maximum_batch, MAXIMUM_BATCH);

    for (first_id, completion) in completions {
        let outputs = completion
            .recv()
            .expect("broker sends one completion")
            .expect("recording service succeeds");
        assert_eq!(outputs.len(), GROUP_SIZE);
        assert_eq!(
            outputs.iter().map(InferenceOutput::id).collect::<Vec<_>>(),
            (first_id..first_id + GROUP_SIZE as u64).collect::<Vec<_>>()
        );
        assert!(
            matches!(completion.try_recv(), Err(mpsc::TryRecvError::Disconnected)),
            "each caller group receives exactly one completion"
        );
    }
}

#[test]
fn broker_rejects_mismatched_request_ids() {
    let evaluator =
        BatchedEvaluatorPool::spawn(broker_config(), vec![ConstantService { wrong_id: true }])
            .unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let error = match evaluator.evaluate(&position) {
        Ok(_) => panic!("mismatched request id should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("expected"));
    assert_eq!(evaluator.metrics().failures, 1);
}

#[test]
fn evaluator_pool_rejects_mismatched_request_ids() {
    let pool = BatchedEvaluatorPool::spawn(
        broker_config(),
        vec![
            ConstantService { wrong_id: true },
            ConstantService { wrong_id: true },
        ],
    )
    .unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let error = match pool.evaluate(&position) {
        Ok(_) => panic!("mismatched request id should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("expected"));
    assert_eq!(pool.metrics().failures, 1);
    assert_eq!(
        pool.lane_metrics()
            .iter()
            .map(|lane| lane.failures)
            .sum::<u64>(),
        1
    );
}

#[test]
fn evaluator_pool_propagates_service_failures() {
    let pool =
        BatchedEvaluatorPool::spawn(broker_config(), vec![FailingService, FailingService]).unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let error = match pool.evaluate(&position) {
        Ok(_) => panic!("service failure should reach the caller"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("synthetic inference failure"));
    assert_eq!(pool.metrics().failures, 1);
}

#[test]
fn evaluator_pool_reports_backend_stage_timings() {
    let pool = BatchedEvaluatorPool::spawn(broker_config(), vec![StagedService]).unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);

    pool.evaluate(&position).unwrap();

    let metrics = pool.metrics();
    assert_eq!(metrics.input_packing_nanoseconds, 11);
    assert_eq!(metrics.session_run_nanoseconds, 13);
    assert_eq!(metrics.output_materialization_nanoseconds, 17);
    assert!(metrics.inference_nanoseconds >= 41);
    assert_eq!(
        pool.lane_metrics()[0].input_packing_nanoseconds,
        metrics.input_packing_nanoseconds
    );
}

#[test]
fn evaluator_pool_builds_full_batches_from_one_shared_queue() {
    const CALLERS: usize = 8;
    let drops = Arc::new(AtomicUsize::new(0));
    let first_batches = Arc::new(Mutex::new(Vec::new()));
    let second_batches = Arc::new(Mutex::new(Vec::new()));
    let pool = Arc::new(
        BatchedEvaluatorPool::spawn(
            BrokerConfig {
                maximum_delay: Duration::from_millis(100),
                queue_capacity: CALLERS,
            },
            vec![
                tagged_service(
                    -0.5,
                    test_contract(4),
                    Arc::clone(&first_batches),
                    Arc::clone(&drops),
                ),
                tagged_service(
                    0.5,
                    test_contract(4),
                    Arc::clone(&second_batches),
                    Arc::clone(&drops),
                ),
            ],
        )
        .unwrap(),
    );
    let ready = Arc::new(std::sync::Barrier::new(CALLERS));
    let position = Position::new(0.1, Vec::new(), Color::Black);

    thread::scope(|scope| {
        for _ in 0..CALLERS {
            let pool = Arc::clone(&pool);
            let ready = Arc::clone(&ready);
            let position = position.clone();
            scope.spawn(move || {
                ready.wait();
                let value = pool.evaluate(&position).unwrap().current_value;
                assert!(value == -0.5 || value == 0.5);
            });
        }
    });

    let mut batch_sizes = first_batches.lock().unwrap().clone();
    batch_sizes.extend(second_batches.lock().unwrap().iter().copied());
    batch_sizes.sort_unstable();
    assert_eq!(batch_sizes, vec![4, 4]);
    assert_eq!(pool.lane_count(), 2);
    let lanes = pool.lane_metrics();
    assert_eq!(lanes.iter().map(|lane| lane.positions).sum::<u64>(), 8);
    assert_eq!(lanes.iter().map(|lane| lane.batches).sum::<u64>(), 2);
    let total = pool.metrics();
    assert_eq!(total.requests, 8);
    assert_eq!(total.positions, 8);
    assert_eq!(total.batches, 2);
    assert_eq!(total.maximum_batch, 4);
    assert_eq!(total.full_batches, 2);
    assert_eq!(total.deadline_batches, 0);
    assert_eq!(total.drain_batches, 0);
    assert_eq!(
        total.queue_nanoseconds,
        total.channel_nanoseconds + total.broker_queue_nanoseconds
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.encoding_nanoseconds)
            .sum::<u64>(),
        total.encoding_nanoseconds
    );
    assert_eq!(
        lanes.iter().map(|lane| lane.queue_nanoseconds).sum::<u64>(),
        total.queue_nanoseconds
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.inference_nanoseconds)
            .sum::<u64>(),
        total.inference_nanoseconds
    );
    assert_eq!(
        lanes.iter().map(|lane| lane.full_batches).sum::<u64>(),
        total.full_batches
    );
}

#[test]
fn evaluator_pool_reassembles_groups_split_across_slots() {
    let drops = Arc::new(AtomicUsize::new(0));
    let first_batches = Arc::new(Mutex::new(Vec::new()));
    let second_batches = Arc::new(Mutex::new(Vec::new()));
    let mut slow = tagged_service(
        -0.25,
        test_contract(4),
        Arc::clone(&first_batches),
        Arc::clone(&drops),
    );
    slow.delay = Duration::from_millis(20);
    let pool = BatchedEvaluatorPool::spawn(
        broker_config(),
        vec![
            slow,
            tagged_service(
                0.75,
                test_contract(4),
                Arc::clone(&second_batches),
                Arc::clone(&drops),
            ),
        ],
    )
    .unwrap();
    let positions = (0..9)
        .map(|index| {
            Position::new(
                0.1,
                Vec::new(),
                if index % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            )
        })
        .collect::<Vec<_>>();

    let evaluations = pool.evaluate_batch(&positions).unwrap();

    assert_eq!(evaluations.len(), 9);
    assert_eq!(
        evaluations
            .iter()
            .map(|evaluation| evaluation.current_value)
            .collect::<Vec<_>>(),
        vec![-0.25, -0.25, -0.25, -0.25, 0.75, 0.75, 0.75, 0.75, 0.75]
    );
    let mut batch_sizes = first_batches.lock().unwrap().clone();
    batch_sizes.extend(second_batches.lock().unwrap().iter().copied());
    batch_sizes.sort_unstable();
    assert_eq!(batch_sizes, vec![1, 4, 4]);
    let total = pool.metrics();
    assert_eq!(total.requests, 9);
    assert_eq!(total.batches, 3);
    assert_eq!(total.positions, 9);
    assert_eq!(total.maximum_batch, 4);
    assert_eq!(total.failures, 0);
    assert_eq!(
        pool.lane_contracts()
            .iter()
            .map(|contract| contract.maximum_batch)
            .collect::<Vec<_>>(),
        vec![4, 4]
    );
}

#[test]
fn evaluator_pool_validates_lane_contracts() {
    let error = match BatchedEvaluatorPool::spawn(broker_config(), Vec::<TaggedService>::new()) {
        Ok(_) => panic!("an empty evaluator pool should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("at least one service"));

    let drops = Arc::new(AtomicUsize::new(0));
    let lane = |contract| {
        tagged_service(
            0.0,
            contract,
            Arc::new(Mutex::new(Vec::new())),
            Arc::clone(&drops),
        )
    };
    let raster_mismatch = BatchContract {
        raster: RasterConfig {
            width: 3,
            height: 2,
            kind: vgo_raster::RasterKind::Semantic,
        },
        ..test_contract(2)
    };
    let error = match BatchedEvaluatorPool::spawn(
        broker_config(),
        vec![lane(test_contract(2)), lane(raster_mismatch)],
    ) {
        Ok(_) => panic!("mismatched raster contracts should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("raster contract"));

    let policy_mismatch = BatchContract {
        policy: RasterConfig::square(3),
        ..test_contract(2)
    };
    let error = match BatchedEvaluatorPool::spawn(
        broker_config(),
        vec![lane(test_contract(2)), lane(policy_mismatch)],
    ) {
        Ok(_) => panic!("mismatched policy contracts should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("policy contract"));

    let error = match BatchedEvaluatorPool::spawn(
        broker_config(),
        vec![lane(test_contract(2)), lane(test_contract(3))],
    ) {
        Ok(_) => panic!("mismatched maximum batches should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("maximum batch"));
}

#[test]
fn evaluator_pool_drops_all_lanes_after_its_last_clone() {
    let drops = Arc::new(AtomicUsize::new(0));
    let services = (0..2)
        .map(|_| {
            tagged_service(
                0.0,
                test_contract(2),
                Arc::new(Mutex::new(Vec::new())),
                Arc::clone(&drops),
            )
        })
        .collect();
    let pool = BatchedEvaluatorPool::spawn(broker_config(), services).unwrap();
    let clone = pool.clone();

    drop(pool);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(clone);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}
