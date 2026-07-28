#![forbid(unsafe_code)]

mod executor;
mod onnx;
mod protocol;

pub use executor::{BatchExecutor, CompletedBatch, InferenceBatch, ThreadedBatchExecutor};
pub use onnx::{OnnxBatchService, OnnxProvider, OnnxServiceConfig};
pub use protocol::{
    InferenceInput, InferenceOutput, encode_request_frame, read_response_frame,
    read_response_frame_with_policy,
};

use std::{
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use vgo_core::Position;
use vgo_raster::{RasterConfig, RasterKind, action_pixel, rasterize};
use vgo_search::{Action, Evaluation, EvaluationError, Evaluator, FineGrid, Policy};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TorchDevice {
    Cpu,
    Cuda,
}

impl TorchDevice {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl std::str::FromStr for TorchDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unsupported torch device: {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PythonProcessConfig {
    pub python: PathBuf,
    pub working_directory: PathBuf,
    pub checkpoint: PathBuf,
    pub raster: RasterConfig,
    pub policy: Option<RasterConfig>,
    pub maximum_batch: usize,
    pub torch_threads: usize,
    pub device: TorchDevice,
    pub compile: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchContract {
    pub raster: RasterConfig,
    /// Placement grid the policy head emits. Equal to `raster` unless the model
    /// was exported with a decoupled (coarser) policy grid.
    pub policy: RasterConfig,
    pub maximum_batch: usize,
}

pub trait BatchService: Send {
    fn contract(&self) -> BatchContract;
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError>;
}

#[derive(Clone, Copy, Debug)]
pub struct BrokerConfig {
    pub maximum_delay: Duration,
    pub queue_capacity: usize,
}

pub struct PythonBatchService {
    child: Child,
    writer: Option<BufWriter<ChildStdin>>,
    reader: BufReader<ChildStdout>,
    contract: BatchContract,
}

impl PythonBatchService {
    pub fn spawn(config: &PythonProcessConfig) -> Result<Self, EvaluationError> {
        if config.maximum_batch == 0 || config.torch_threads == 0 {
            return Err(EvaluationError::new(
                "batch size and torch thread count must be positive",
            ));
        }
        let working_directory =
            std::path::absolute(&config.working_directory).map_err(|error| {
                EvaluationError::new(format!("resolve training directory: {error}"))
            })?;
        let checkpoint = std::path::absolute(&config.checkpoint)
            .map_err(|error| EvaluationError::new(format!("resolve checkpoint path: {error}")))?;
        let mut child = Command::new(&config.python)
            .current_dir(working_directory)
            .arg("-m")
            .arg("vgo_training.serve")
            .arg("--checkpoint")
            .arg(checkpoint)
            .arg("--threads")
            .arg(config.torch_threads.to_string())
            .arg("--device")
            .arg(config.device.as_str())
            .arg(if config.compile {
                "--compile"
            } else {
                "--no-compile"
            })
            .arg("--maximum-batch")
            .arg(config.maximum_batch.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| EvaluationError::new(format!("start Python service: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EvaluationError::new("Python service has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EvaluationError::new("Python service has no stdout"))?;
        Ok(Self {
            child,
            writer: Some(BufWriter::new(stdin)),
            reader: BufReader::new(stdout),
            contract: BatchContract {
                raster: config.raster,
                policy: config.policy.unwrap_or(config.raster),
                maximum_batch: config.maximum_batch,
            },
        })
    }
}

impl BatchService for PythonBatchService {
    fn contract(&self) -> BatchContract {
        self.contract
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        let frame = encode_request_frame(batch)?;
        let writer = self
            .writer
            .as_mut()
            .expect("writer exists while service is alive");
        writer.write_all(&frame).map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        read_response_frame_with_policy(&mut self.reader, batch, self.contract.policy)
    }
}

impl Drop for PythonBatchService {
    fn drop(&mut self) {
        self.writer.take();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BrokerMetrics {
    pub requests: u64,
    pub batches: u64,
    pub positions: u64,
    pub maximum_batch: usize,
    pub failures: u64,
    pub encoding_nanoseconds: u64,
    pub queue_nanoseconds: u64,
    pub inference_nanoseconds: u64,
}

#[derive(Default)]
struct AtomicMetrics {
    requests: AtomicU64,
    batches: AtomicU64,
    positions: AtomicU64,
    maximum_batch: AtomicUsize,
    failures: AtomicU64,
    encoding_nanoseconds: AtomicU64,
    queue_nanoseconds: AtomicU64,
    inference_nanoseconds: AtomicU64,
}

impl AtomicMetrics {
    fn snapshot(&self) -> BrokerMetrics {
        BrokerMetrics {
            requests: self.requests.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            positions: self.positions.load(Ordering::Relaxed),
            maximum_batch: self.maximum_batch.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            encoding_nanoseconds: self.encoding_nanoseconds.load(Ordering::Relaxed),
            queue_nanoseconds: self.queue_nanoseconds.load(Ordering::Relaxed),
            inference_nanoseconds: self.inference_nanoseconds.load(Ordering::Relaxed),
        }
    }
}

struct Request {
    inputs: Vec<InferenceInput>,
    queued_at: Instant,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

struct PendingRequest {
    inputs: std::vec::IntoIter<InferenceInput>,
    outputs: Vec<InferenceOutput>,
    queued_at: Instant,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

impl Request {
    fn into_pending(self) -> PendingRequest {
        let output_capacity = self.inputs.len();
        PendingRequest {
            inputs: self.inputs.into_iter(),
            outputs: Vec::with_capacity(output_capacity),
            queued_at: self.queued_at,
            response: self.response,
        }
    }
}

struct BatchPart {
    request: PendingRequest,
    count: usize,
}

struct Inner {
    sender: Option<SyncSender<Request>>,
    next_id: AtomicU64,
    contract: BatchContract,
    metrics: Arc<AtomicMetrics>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.join.lock().expect("broker join lock").take() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone)]
pub struct BatchedEvaluator {
    inner: Arc<Inner>,
}

impl BatchedEvaluator {
    pub fn spawn(
        config: BrokerConfig,
        service: impl BatchService + 'static,
    ) -> Result<Self, EvaluationError> {
        if config.queue_capacity == 0 {
            return Err(EvaluationError::new("queue capacity must be positive"));
        }
        let contract = service.contract();
        if contract.maximum_batch == 0
            || contract.raster.width == 0
            || contract.raster.height == 0
            || contract.policy.width == 0
            || contract.policy.height == 0
        {
            return Err(EvaluationError::new(
                "inference service contract dimensions must be positive",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let metrics = Arc::new(AtomicMetrics::default());
        let broker_metrics = Arc::clone(&metrics);
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-broker"))
            .spawn(move || run_broker(config, contract, service, receiver, broker_metrics))
            .map_err(|error| EvaluationError::new(format!("start inference broker: {error}")))?;
        Ok(Self {
            inner: Arc::new(Inner {
                sender: Some(sender),
                next_id: AtomicU64::new(1),
                contract,
                metrics,
                join: Mutex::new(Some(join)),
            }),
        })
    }

    #[must_use]
    pub fn contract(&self) -> BatchContract {
        self.inner.contract
    }

    #[must_use]
    pub fn metrics(&self) -> BrokerMetrics {
        self.inner.metrics.snapshot()
    }

    fn evaluate_positions(
        &self,
        positions: &[Position],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        let encoding_started = Instant::now();
        let inputs = positions
            .iter()
            .map(|position| {
                let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
                InferenceInput::new(id, rasterize(position, self.inner.contract.raster))
            })
            .collect::<Vec<_>>();
        self.inner.metrics.encoding_nanoseconds.fetch_add(
            encoding_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );

        let (response_sender, response_receiver) = mpsc::sync_channel(1);
        self.inner
            .metrics
            .requests
            .fetch_add(inputs.len() as u64, Ordering::Relaxed);
        self.inner
            .sender
            .as_ref()
            .expect("sender exists while evaluator is alive")
            .send(Request {
                inputs,
                queued_at: Instant::now(),
                response: response_sender,
            })
            .map_err(|_| EvaluationError::new("inference broker has stopped"))?;

        // The broker may split this group across backend batches, but it sends
        // exactly one completion after reassembling every output in input order.
        response_receiver
            .recv()
            .map_err(|_| EvaluationError::new("inference broker dropped the response"))??
            .into_iter()
            .map(|output| {
                let (_, current_value, policy) = output.into_parts();
                Ok(Evaluation::new(
                    current_value,
                    Box::new(DensePolicy {
                        config: self.inner.contract.policy,
                        logits: policy,
                    }),
                ))
            })
            .collect()
    }
}

impl Evaluator for BatchedEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        self.evaluate_positions(std::slice::from_ref(position))
            .map(|mut evaluations| {
                evaluations
                    .pop()
                    .expect("one position produces one evaluation")
            })
    }

    fn evaluate_batch(&self, positions: &[Position]) -> Result<Vec<Evaluation>, EvaluationError> {
        self.evaluate_positions(positions)
    }
}

/// A round-robin pool of independent inference brokers.
///
/// Every logical [`Evaluator::evaluate`] or [`Evaluator::evaluate_batch`] call
/// is submitted intact to one lane. The pool only routes the borrowed
/// positions; raster payloads are created once by the selected lane and are
/// never cloned between lanes.
#[derive(Clone)]
pub struct BatchedEvaluatorPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    lanes: Vec<BatchedEvaluator>,
    next_lane: AtomicUsize,
}

impl BatchedEvaluatorPool {
    pub fn new(lanes: Vec<BatchedEvaluator>) -> Result<Self, EvaluationError> {
        let Some(first) = lanes.first() else {
            return Err(EvaluationError::new(
                "inference evaluator pool must contain at least one lane",
            ));
        };
        let expected = first.contract();
        for (index, lane) in lanes.iter().enumerate().skip(1) {
            let actual = lane.contract();
            if actual.raster != expected.raster {
                return Err(EvaluationError::new(format!(
                    "inference evaluator lane {index} has raster contract {:?}, expected {:?}",
                    actual.raster, expected.raster
                )));
            }
            if actual.policy != expected.policy {
                return Err(EvaluationError::new(format!(
                    "inference evaluator lane {index} has policy contract {:?}, expected {:?}",
                    actual.policy, expected.policy
                )));
            }
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                lanes,
                next_lane: AtomicUsize::new(0),
            }),
        })
    }

    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.inner.lanes.len()
    }

    #[must_use]
    pub fn lane_contracts(&self) -> Vec<BatchContract> {
        self.inner
            .lanes
            .iter()
            .map(BatchedEvaluator::contract)
            .collect()
    }

    #[must_use]
    pub fn lane_metrics(&self) -> Vec<BrokerMetrics> {
        self.inner
            .lanes
            .iter()
            .map(BatchedEvaluator::metrics)
            .collect()
    }

    #[must_use]
    pub fn metrics(&self) -> BrokerMetrics {
        self.lane_metrics()
            .into_iter()
            .fold(BrokerMetrics::default(), |mut total, lane| {
                total.requests = total.requests.saturating_add(lane.requests);
                total.batches = total.batches.saturating_add(lane.batches);
                total.positions = total.positions.saturating_add(lane.positions);
                total.maximum_batch = total.maximum_batch.max(lane.maximum_batch);
                total.failures = total.failures.saturating_add(lane.failures);
                total.encoding_nanoseconds = total
                    .encoding_nanoseconds
                    .saturating_add(lane.encoding_nanoseconds);
                total.queue_nanoseconds = total
                    .queue_nanoseconds
                    .saturating_add(lane.queue_nanoseconds);
                total.inference_nanoseconds = total
                    .inference_nanoseconds
                    .saturating_add(lane.inference_nanoseconds);
                total
            })
    }

    fn next_lane(&self) -> &BatchedEvaluator {
        let index = self.inner.next_lane.fetch_add(1, Ordering::Relaxed) % self.inner.lanes.len();
        &self.inner.lanes[index]
    }
}

impl Evaluator for BatchedEvaluatorPool {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        self.next_lane().evaluate(position)
    }

    fn evaluate_batch(&self, positions: &[Position]) -> Result<Vec<Evaluation>, EvaluationError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        self.next_lane().evaluate_batch(positions)
    }
}

struct DensePolicy {
    /// Placement grid the logits are laid out on. This is the policy grid, which
    /// may be coarser than the rendered raster.
    config: RasterConfig,
    logits: Vec<f32>,
}

impl Policy for DensePolicy {
    fn logit(&self, action: Action) -> f64 {
        let index = match action {
            Action::Pass => self.config.pixels(),
            Action::Place(point) => action_pixel(point.x, point.y, self.config),
        };
        f64::from(self.logits[index])
    }

    fn fine_grid(&self, position: &vgo_core::Position, coarse: usize) -> Option<FineGrid> {
        let width = self.config.width;
        let height = self.config.height;
        Some(FineGrid::build(
            position,
            width,
            height,
            coarse,
            |row, col| self.logits[row * width + col],
        ))
    }
}

fn run_broker(
    config: BrokerConfig,
    contract: BatchContract,
    mut service: impl BatchService,
    receiver: Receiver<Request>,
    metrics: Arc<AtomicMetrics>,
) {
    let mut carry = None;
    loop {
        let mut current = if let Some(request) = carry.take() {
            request
        } else {
            let Ok(request) = receiver.recv() else {
                break;
            };
            request.into_pending()
        };

        let mut inputs = Vec::with_capacity(contract.maximum_batch);
        let mut parts = Vec::<BatchPart>::new();
        let deadline = Instant::now() + config.maximum_delay;
        loop {
            let available = contract.maximum_batch - inputs.len();
            let count = available.min(current.inputs.len());
            debug_assert!(count > 0, "pending requests always contain input");
            inputs.extend(current.inputs.by_ref().take(count));
            let has_remaining = current.inputs.len() != 0;
            parts.push(BatchPart {
                request: current,
                count,
            });

            if inputs.len() == contract.maximum_batch {
                break;
            }
            debug_assert!(
                !has_remaining,
                "a partially consumed request must fill the backend batch"
            );
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(request) => current = request.into_pending(),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }

        let position_count = inputs.len();
        let now = Instant::now();
        for part in &parts {
            let queued = now.duration_since(part.request.queued_at).as_nanos() as u64;
            metrics
                .queue_nanoseconds
                .fetch_add(queued.saturating_mul(part.count as u64), Ordering::Relaxed);
        }
        metrics.batches.fetch_add(1, Ordering::Relaxed);
        metrics
            .positions
            .fetch_add(position_count as u64, Ordering::Relaxed);
        metrics
            .maximum_batch
            .fetch_max(position_count, Ordering::Relaxed);
        let expected_ids = inputs.iter().map(InferenceInput::id).collect::<Vec<_>>();
        let started = Instant::now();
        let result = service.infer(&inputs);
        metrics
            .inference_nanoseconds
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let result = result.and_then(|outputs| {
            if outputs.len() != expected_ids.len() {
                return Err(EvaluationError::new(format!(
                    "inference backend returned {} outputs for {} inputs",
                    outputs.len(),
                    expected_ids.len()
                )));
            }
            for (index, (output, expected_id)) in outputs.iter().zip(&expected_ids).enumerate() {
                if output.id() != *expected_id {
                    return Err(EvaluationError::new(format!(
                        "inference output {index} has request id {}, expected {expected_id}",
                        output.id()
                    )));
                }
            }
            Ok(outputs)
        });
        match result {
            Ok(outputs) => {
                let mut outputs = outputs.into_iter();
                for mut part in parts {
                    let prior_outputs = part.request.outputs.len();
                    part.request
                        .outputs
                        .extend(outputs.by_ref().take(part.count));
                    debug_assert_eq!(
                        part.request.outputs.len(),
                        prior_outputs + part.count,
                        "backend validation guarantees one output per input"
                    );
                    if part.request.inputs.len() == 0 {
                        let _ = part.request.response.send(Ok(part.request.outputs));
                    } else {
                        debug_assert!(carry.is_none(), "only the final batch part can continue");
                        carry = Some(part.request);
                    }
                }
            }
            Err(error) => {
                metrics
                    .failures
                    .fetch_add(position_count as u64, Ordering::Relaxed);
                for part in parts {
                    let _ = part.request.response.send(Err(error.clone()));
                }
                break;
            }
        }
    }
}

fn io_error(error: std::io::Error) -> EvaluationError {
    EvaluationError::new(format!("inference transport: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgo_core::Color;

    struct ConstantService {
        wrong_id: bool,
    }

    impl BatchService for ConstantService {
        fn contract(&self) -> BatchContract {
            BatchContract {
                raster: RasterConfig::square(2),
                policy: RasterConfig::square(2),
                maximum_batch: 2,
            }
        }

        fn infer(
            &mut self,
            batch: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
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
            }
        }

        fn infer(
            &mut self,
            batch: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
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
        batch_sizes: Arc<Mutex<Vec<usize>>>,
        drops: Arc<AtomicUsize>,
    }

    impl BatchService for TaggedService {
        fn contract(&self) -> BatchContract {
            self.contract
        }

        fn infer(
            &mut self,
            batch: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
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

    fn tagged_evaluator(
        current_value: f64,
        contract: BatchContract,
        batch_sizes: Arc<Mutex<Vec<usize>>>,
        drops: Arc<AtomicUsize>,
    ) -> BatchedEvaluator {
        BatchedEvaluator::spawn(
            BrokerConfig {
                maximum_delay: Duration::ZERO,
                queue_capacity: 8,
            },
            TaggedService {
                contract,
                current_value,
                batch_sizes,
                drops,
            },
        )
        .unwrap()
    }

    fn test_contract(maximum_batch: usize) -> BatchContract {
        BatchContract {
            raster: RasterConfig::square(2),
            policy: RasterConfig::square(2),
            maximum_batch,
        }
    }

    fn broker_config() -> BrokerConfig {
        BrokerConfig {
            maximum_delay: Duration::ZERO,
            queue_capacity: 2,
        }
    }

    #[test]
    fn batched_evaluator_adapts_a_batch_service() {
        let evaluator =
            BatchedEvaluator::spawn(broker_config(), ConstantService { wrong_id: false }).unwrap();
        let position = Position::new(0.1, Vec::new(), Color::Black);
        let evaluation = evaluator.evaluate(&position).unwrap();
        assert_eq!(evaluation.current_value, 0.25);
        assert_eq!(evaluation.policy_logit(Action::Pass), 0.0);
        assert_eq!(evaluator.metrics().requests, 1);
    }

    #[test]
    fn evaluate_batch_reassembles_a_group_larger_than_the_backend_batch() {
        let evaluator =
            BatchedEvaluator::spawn(broker_config(), ConstantService { wrong_id: false }).unwrap();
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
                    queued_at: Instant::now(),
                    response,
                })
                .unwrap();
            completions.push((first_id, completion));
        }
        drop(sender);

        run_broker(
            BrokerConfig {
                maximum_delay: Duration::from_millis(10),
                queue_capacity: GROUPS,
            },
            contract,
            service,
            receiver,
            Arc::clone(&metrics),
        );

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
            BatchedEvaluator::spawn(broker_config(), ConstantService { wrong_id: true }).unwrap();
        let position = Position::new(0.1, Vec::new(), Color::Black);
        let error = match evaluator.evaluate(&position) {
            Ok(_) => panic!("mismatched request id should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected"));
        assert_eq!(evaluator.metrics().failures, 1);
    }

    #[test]
    fn evaluator_pool_routes_calls_round_robin() {
        let drops = Arc::new(AtomicUsize::new(0));
        let lanes = [-0.5, 0.0, 0.5]
            .into_iter()
            .map(|value| {
                tagged_evaluator(
                    value,
                    test_contract(2),
                    Arc::new(Mutex::new(Vec::new())),
                    Arc::clone(&drops),
                )
            })
            .collect();
        let pool = BatchedEvaluatorPool::new(lanes).unwrap();
        let position = Position::new(0.1, Vec::new(), Color::Black);

        let values = (0..6)
            .map(|_| pool.evaluate(&position).unwrap().current_value)
            .collect::<Vec<_>>();

        assert_eq!(values, vec![-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]);
        assert_eq!(pool.lane_count(), 3);
        assert_eq!(
            pool.lane_metrics()
                .iter()
                .map(|metrics| metrics.positions)
                .collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
    }

    #[test]
    fn evaluator_pool_keeps_each_group_on_one_lane() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first_batches = Arc::new(Mutex::new(Vec::new()));
        let second_batches = Arc::new(Mutex::new(Vec::new()));
        let pool = BatchedEvaluatorPool::new(vec![
            tagged_evaluator(
                -0.25,
                test_contract(2),
                Arc::clone(&first_batches),
                Arc::clone(&drops),
            ),
            tagged_evaluator(
                0.75,
                test_contract(2),
                Arc::clone(&second_batches),
                Arc::clone(&drops),
            ),
        ])
        .unwrap();
        let positions = [
            Position::new(0.1, Vec::new(), Color::Black),
            Position::new(0.1, Vec::new(), Color::White),
            Position::new(0.1, Vec::new(), Color::Black),
        ];

        let first = pool.evaluate_batch(&positions).unwrap();
        let second = pool.evaluate_batch(&positions).unwrap();

        assert!(
            first
                .iter()
                .all(|evaluation| evaluation.current_value == -0.25)
        );
        assert!(
            second
                .iter()
                .all(|evaluation| evaluation.current_value == 0.75)
        );
        assert_eq!(
            *first_batches.lock().expect("first batch-size lock"),
            vec![2, 1]
        );
        assert_eq!(
            *second_batches.lock().expect("second batch-size lock"),
            vec![2, 1]
        );
    }

    #[test]
    fn evaluator_pool_aggregates_and_exposes_lane_metrics() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = BatchedEvaluatorPool::new(vec![
            tagged_evaluator(
                -0.25,
                test_contract(2),
                Arc::new(Mutex::new(Vec::new())),
                Arc::clone(&drops),
            ),
            tagged_evaluator(
                0.75,
                test_contract(4),
                Arc::new(Mutex::new(Vec::new())),
                Arc::clone(&drops),
            ),
        ])
        .unwrap();
        let positions = [
            Position::new(0.1, Vec::new(), Color::Black),
            Position::new(0.1, Vec::new(), Color::White),
            Position::new(0.1, Vec::new(), Color::Black),
        ];
        pool.evaluate_batch(&positions).unwrap();
        pool.evaluate_batch(&positions).unwrap();

        let lanes = pool.lane_metrics();
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0].requests, 3);
        assert_eq!(lanes[0].batches, 2);
        assert_eq!(lanes[0].positions, 3);
        assert_eq!(lanes[0].maximum_batch, 2);
        assert_eq!(lanes[1].requests, 3);
        assert_eq!(lanes[1].batches, 1);
        assert_eq!(lanes[1].positions, 3);
        assert_eq!(lanes[1].maximum_batch, 3);

        let total = pool.metrics();
        assert_eq!(total.requests, 6);
        assert_eq!(total.batches, 3);
        assert_eq!(total.positions, 6);
        assert_eq!(total.maximum_batch, 3);
        assert_eq!(total.failures, 0);
        assert_eq!(
            pool.lane_contracts()
                .iter()
                .map(|contract| contract.maximum_batch)
                .collect::<Vec<_>>(),
            vec![2, 4]
        );
    }

    #[test]
    fn evaluator_pool_validates_lane_contracts() {
        let error = match BatchedEvaluatorPool::new(Vec::new()) {
            Ok(_) => panic!("an empty evaluator pool should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("at least one lane"));

        let drops = Arc::new(AtomicUsize::new(0));
        let lane = |contract| {
            tagged_evaluator(
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
                kind: RasterKind::Semantic,
            },
            ..test_contract(2)
        };
        let error =
            match BatchedEvaluatorPool::new(vec![lane(test_contract(2)), lane(raster_mismatch)]) {
                Ok(_) => panic!("mismatched raster contracts should fail"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("raster contract"));

        let policy_mismatch = BatchContract {
            policy: RasterConfig::square(3),
            ..test_contract(2)
        };
        let error =
            match BatchedEvaluatorPool::new(vec![lane(test_contract(2)), lane(policy_mismatch)]) {
                Ok(_) => panic!("mismatched policy contracts should fail"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("policy contract"));
    }

    #[test]
    fn evaluator_pool_drops_all_lanes_after_its_last_clone() {
        let drops = Arc::new(AtomicUsize::new(0));
        let lanes = (0..2)
            .map(|_| {
                tagged_evaluator(
                    0.0,
                    test_contract(2),
                    Arc::new(Mutex::new(Vec::new())),
                    Arc::clone(&drops),
                )
            })
            .collect();
        let pool = BatchedEvaluatorPool::new(lanes).unwrap();
        let clone = pool.clone();

        drop(pool);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(clone);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }
}
