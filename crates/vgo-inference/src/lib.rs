#![forbid(unsafe_code)]

mod executor;
mod onnx;
mod protocol;

pub use executor::{
    BatchExecutor, CompletedBatch, InferenceBatch, ThreadedBatchExecutor, ThreadedBatchExecutorPool,
};
pub use onnx::{OnnxBatchService, OnnxProvider, OnnxServiceConfig};
pub use protocol::{
    InferenceInput, InferenceOutput, encode_request_frame, read_response_frame,
    read_response_frame_with_policy,
};

use std::{
    collections::{HashMap, VecDeque},
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
use vgo_raster::{DensePolicy, RasterConfig, rasterize};
use vgo_search::{Evaluation, EvaluationError, Evaluator};

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

/// Time spent inside the host-visible stages of one backend call.
///
/// `session_run_nanoseconds` still includes runtime setup and any synchronous
/// host/device transfers performed by ONNX Runtime. Splitting those transfers
/// further requires I/O binding and explicit device buffers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceStageMetrics {
    pub input_packing_nanoseconds: u64,
    pub session_run_nanoseconds: u64,
    pub output_materialization_nanoseconds: u64,
}

pub trait BatchService: Send {
    fn contract(&self) -> BatchContract;
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError>;

    /// Return the stage timings for the most recent `infer` call. Backends
    /// without internal instrumentation leave the call unattributed.
    fn last_inference_stages(&self) -> InferenceStageMetrics {
        InferenceStageMetrics::default()
    }
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
    /// Position-weighted time between actor submission and broker receipt.
    pub channel_nanoseconds: u64,
    /// Position-weighted time between broker receipt and slot dispatch.
    pub broker_queue_nanoseconds: u64,
    /// Broker wall time waiting for the first request while every slot is idle.
    pub idle_request_wait_nanoseconds: u64,
    /// Broker wall time waiting for a request while at least one slot is busy.
    pub overlap_request_wait_nanoseconds: u64,
    /// Summed wall time spent constructing batches, including the deadline wait.
    pub batch_collection_nanoseconds: u64,
    /// Summed wall time spent transferring batch ownership to executor threads.
    pub batch_submission_nanoseconds: u64,
    /// Broker wall time blocked waiting for an executor completion.
    pub completion_wait_nanoseconds: u64,
    pub full_batches: u64,
    pub deadline_batches: u64,
    pub drain_batches: u64,
    pub inference_nanoseconds: u64,
    pub input_packing_nanoseconds: u64,
    pub session_run_nanoseconds: u64,
    pub output_materialization_nanoseconds: u64,
}

impl BrokerMetrics {
    #[must_use]
    pub fn inference_unattributed_nanoseconds(self) -> u64 {
        self.inference_nanoseconds.saturating_sub(
            self.input_packing_nanoseconds
                .saturating_add(self.session_run_nanoseconds)
                .saturating_add(self.output_materialization_nanoseconds),
        )
    }

    #[must_use]
    pub fn delta_since(self, earlier: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(earlier.requests),
            batches: self.batches.saturating_sub(earlier.batches),
            positions: self.positions.saturating_sub(earlier.positions),
            maximum_batch: self.maximum_batch,
            failures: self.failures.saturating_sub(earlier.failures),
            encoding_nanoseconds: self
                .encoding_nanoseconds
                .saturating_sub(earlier.encoding_nanoseconds),
            queue_nanoseconds: self
                .queue_nanoseconds
                .saturating_sub(earlier.queue_nanoseconds),
            channel_nanoseconds: self
                .channel_nanoseconds
                .saturating_sub(earlier.channel_nanoseconds),
            broker_queue_nanoseconds: self
                .broker_queue_nanoseconds
                .saturating_sub(earlier.broker_queue_nanoseconds),
            idle_request_wait_nanoseconds: self
                .idle_request_wait_nanoseconds
                .saturating_sub(earlier.idle_request_wait_nanoseconds),
            overlap_request_wait_nanoseconds: self
                .overlap_request_wait_nanoseconds
                .saturating_sub(earlier.overlap_request_wait_nanoseconds),
            batch_collection_nanoseconds: self
                .batch_collection_nanoseconds
                .saturating_sub(earlier.batch_collection_nanoseconds),
            batch_submission_nanoseconds: self
                .batch_submission_nanoseconds
                .saturating_sub(earlier.batch_submission_nanoseconds),
            completion_wait_nanoseconds: self
                .completion_wait_nanoseconds
                .saturating_sub(earlier.completion_wait_nanoseconds),
            full_batches: self.full_batches.saturating_sub(earlier.full_batches),
            deadline_batches: self
                .deadline_batches
                .saturating_sub(earlier.deadline_batches),
            drain_batches: self.drain_batches.saturating_sub(earlier.drain_batches),
            inference_nanoseconds: self
                .inference_nanoseconds
                .saturating_sub(earlier.inference_nanoseconds),
            input_packing_nanoseconds: self
                .input_packing_nanoseconds
                .saturating_sub(earlier.input_packing_nanoseconds),
            session_run_nanoseconds: self
                .session_run_nanoseconds
                .saturating_sub(earlier.session_run_nanoseconds),
            output_materialization_nanoseconds: self
                .output_materialization_nanoseconds
                .saturating_sub(earlier.output_materialization_nanoseconds),
        }
    }

    pub fn accumulate(&mut self, other: Self) {
        self.requests = self.requests.saturating_add(other.requests);
        self.batches = self.batches.saturating_add(other.batches);
        self.positions = self.positions.saturating_add(other.positions);
        self.maximum_batch = self.maximum_batch.max(other.maximum_batch);
        self.failures = self.failures.saturating_add(other.failures);
        self.encoding_nanoseconds = self
            .encoding_nanoseconds
            .saturating_add(other.encoding_nanoseconds);
        self.queue_nanoseconds = self
            .queue_nanoseconds
            .saturating_add(other.queue_nanoseconds);
        self.channel_nanoseconds = self
            .channel_nanoseconds
            .saturating_add(other.channel_nanoseconds);
        self.broker_queue_nanoseconds = self
            .broker_queue_nanoseconds
            .saturating_add(other.broker_queue_nanoseconds);
        self.idle_request_wait_nanoseconds = self
            .idle_request_wait_nanoseconds
            .saturating_add(other.idle_request_wait_nanoseconds);
        self.overlap_request_wait_nanoseconds = self
            .overlap_request_wait_nanoseconds
            .saturating_add(other.overlap_request_wait_nanoseconds);
        self.batch_collection_nanoseconds = self
            .batch_collection_nanoseconds
            .saturating_add(other.batch_collection_nanoseconds);
        self.batch_submission_nanoseconds = self
            .batch_submission_nanoseconds
            .saturating_add(other.batch_submission_nanoseconds);
        self.completion_wait_nanoseconds = self
            .completion_wait_nanoseconds
            .saturating_add(other.completion_wait_nanoseconds);
        self.full_batches = self.full_batches.saturating_add(other.full_batches);
        self.deadline_batches = self.deadline_batches.saturating_add(other.deadline_batches);
        self.drain_batches = self.drain_batches.saturating_add(other.drain_batches);
        self.inference_nanoseconds = self
            .inference_nanoseconds
            .saturating_add(other.inference_nanoseconds);
        self.input_packing_nanoseconds = self
            .input_packing_nanoseconds
            .saturating_add(other.input_packing_nanoseconds);
        self.session_run_nanoseconds = self
            .session_run_nanoseconds
            .saturating_add(other.session_run_nanoseconds);
        self.output_materialization_nanoseconds = self
            .output_materialization_nanoseconds
            .saturating_add(other.output_materialization_nanoseconds);
    }
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
    channel_nanoseconds: AtomicU64,
    broker_queue_nanoseconds: AtomicU64,
    idle_request_wait_nanoseconds: AtomicU64,
    overlap_request_wait_nanoseconds: AtomicU64,
    batch_collection_nanoseconds: AtomicU64,
    batch_submission_nanoseconds: AtomicU64,
    completion_wait_nanoseconds: AtomicU64,
    full_batches: AtomicU64,
    deadline_batches: AtomicU64,
    drain_batches: AtomicU64,
    inference_nanoseconds: AtomicU64,
    input_packing_nanoseconds: AtomicU64,
    session_run_nanoseconds: AtomicU64,
    output_materialization_nanoseconds: AtomicU64,
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
            channel_nanoseconds: self.channel_nanoseconds.load(Ordering::Relaxed),
            broker_queue_nanoseconds: self.broker_queue_nanoseconds.load(Ordering::Relaxed),
            idle_request_wait_nanoseconds: self
                .idle_request_wait_nanoseconds
                .load(Ordering::Relaxed),
            overlap_request_wait_nanoseconds: self
                .overlap_request_wait_nanoseconds
                .load(Ordering::Relaxed),
            batch_collection_nanoseconds: self.batch_collection_nanoseconds.load(Ordering::Relaxed),
            batch_submission_nanoseconds: self.batch_submission_nanoseconds.load(Ordering::Relaxed),
            completion_wait_nanoseconds: self.completion_wait_nanoseconds.load(Ordering::Relaxed),
            full_batches: self.full_batches.load(Ordering::Relaxed),
            deadline_batches: self.deadline_batches.load(Ordering::Relaxed),
            drain_batches: self.drain_batches.load(Ordering::Relaxed),
            inference_nanoseconds: self.inference_nanoseconds.load(Ordering::Relaxed),
            input_packing_nanoseconds: self.input_packing_nanoseconds.load(Ordering::Relaxed),
            session_run_nanoseconds: self.session_run_nanoseconds.load(Ordering::Relaxed),
            output_materialization_nanoseconds: self
                .output_materialization_nanoseconds
                .load(Ordering::Relaxed),
        }
    }

    fn record_stages(&self, stages: InferenceStageMetrics) {
        self.input_packing_nanoseconds
            .fetch_add(stages.input_packing_nanoseconds, Ordering::Relaxed);
        self.session_run_nanoseconds
            .fetch_add(stages.session_run_nanoseconds, Ordering::Relaxed);
        self.output_materialization_nanoseconds
            .fetch_add(stages.output_materialization_nanoseconds, Ordering::Relaxed);
    }
}

struct Request {
    inputs: Vec<InferenceInput>,
    encoding_nanoseconds: u64,
    queued_at: Instant,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

struct PendingRequest {
    inputs: std::vec::IntoIter<InferenceInput>,
    outputs: Vec<InferenceOutput>,
    queued_at: Instant,
    brokered_at: Instant,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

impl Request {
    fn into_pending(self, brokered_at: Instant) -> PendingRequest {
        let output_capacity = self.inputs.len();
        PendingRequest {
            inputs: self.inputs.into_iter(),
            outputs: Vec::with_capacity(output_capacity),
            queued_at: self.queued_at,
            brokered_at,
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
        let encoding_nanoseconds = encoding_started.elapsed().as_nanos() as u64;
        self.inner
            .metrics
            .encoding_nanoseconds
            .fetch_add(encoding_nanoseconds, Ordering::Relaxed);

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
                encoding_nanoseconds,
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
                    Box::new(DensePolicy::new(self.inner.contract.policy, policy)),
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

/// One shared batching broker backed by independent inference execution slots.
///
/// All callers feed the same FIFO queue. The broker constructs each backend
/// batch before assigning it to a free slot, so adding execution concurrency
/// does not fragment arrivals across independent collection deadlines.
#[derive(Clone)]
pub struct BatchedEvaluatorPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    sender: Option<SyncSender<Request>>,
    next_id: AtomicU64,
    contract: BatchContract,
    lane_contracts: Vec<BatchContract>,
    metrics: Arc<AtomicMetrics>,
    lane_metrics: Vec<Arc<AtomicMetrics>>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.join.lock().expect("pool broker join lock").take() {
            let _ = handle.join();
        }
    }
}

impl BatchedEvaluatorPool {
    pub fn spawn<S: BatchService + 'static>(
        config: BrokerConfig,
        services: Vec<S>,
    ) -> Result<Self, EvaluationError> {
        if config.queue_capacity == 0 {
            return Err(EvaluationError::new("queue capacity must be positive"));
        }
        let Some(first) = services.first() else {
            return Err(EvaluationError::new(
                "inference evaluator pool must contain at least one service",
            ));
        };
        let expected = first.contract();
        if expected.maximum_batch == 0
            || expected.raster.width == 0
            || expected.raster.height == 0
            || expected.policy.width == 0
            || expected.policy.height == 0
        {
            return Err(EvaluationError::new(
                "inference service contract dimensions must be positive",
            ));
        }
        let lane_contracts = services
            .iter()
            .map(BatchService::contract)
            .collect::<Vec<_>>();
        for (index, actual) in lane_contracts.iter().copied().enumerate().skip(1) {
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
            if actual.maximum_batch != expected.maximum_batch {
                return Err(EvaluationError::new(format!(
                    "inference evaluator lane {index} has maximum batch {}, expected {}",
                    actual.maximum_batch, expected.maximum_batch
                )));
            }
        }
        let executor = ThreadedBatchExecutorPool::spawn(services)?;
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let metrics = Arc::new(AtomicMetrics::default());
        let lane_metrics = (0..lane_contracts.len())
            .map(|_| Arc::new(AtomicMetrics::default()))
            .collect::<Vec<_>>();
        let broker_metrics = Arc::clone(&metrics);
        let broker_lane_metrics = lane_metrics.clone();
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-pool-broker"))
            .spawn(move || {
                run_pool_broker(
                    config,
                    expected,
                    executor,
                    receiver,
                    broker_metrics,
                    broker_lane_metrics,
                );
            })
            .map_err(|error| EvaluationError::new(format!("start inference broker: {error}")))?;
        Ok(Self {
            inner: Arc::new(PoolInner {
                sender: Some(sender),
                next_id: AtomicU64::new(1),
                contract: expected,
                lane_contracts,
                metrics,
                lane_metrics,
                join: Mutex::new(Some(join)),
            }),
        })
    }

    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.inner.lane_metrics.len()
    }

    #[must_use]
    pub fn lane_contracts(&self) -> Vec<BatchContract> {
        self.inner.lane_contracts.clone()
    }

    #[must_use]
    pub fn lane_metrics(&self) -> Vec<BrokerMetrics> {
        self.inner
            .lane_metrics
            .iter()
            .map(|metrics| metrics.snapshot())
            .collect()
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
        let encoding_nanoseconds = encoding_started.elapsed().as_nanos() as u64;
        self.inner
            .metrics
            .encoding_nanoseconds
            .fetch_add(encoding_nanoseconds, Ordering::Relaxed);
        self.inner
            .metrics
            .requests
            .fetch_add(inputs.len() as u64, Ordering::Relaxed);

        let (response, completion) = mpsc::sync_channel(1);
        self.inner
            .sender
            .as_ref()
            .expect("sender exists while evaluator pool is alive")
            .send(Request {
                inputs,
                encoding_nanoseconds,
                queued_at: Instant::now(),
                response,
            })
            .map_err(|_| EvaluationError::new("inference broker has stopped"))?;

        completion
            .recv()
            .map_err(|_| EvaluationError::new("inference broker dropped the response"))??
            .into_iter()
            .map(|output| {
                let (_, current_value, policy) = output.into_parts();
                Ok(Evaluation::new(
                    current_value,
                    Box::new(DensePolicy::new(self.inner.contract.policy, policy)),
                ))
            })
            .collect()
    }
}

impl Evaluator for BatchedEvaluatorPool {
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
            let wait_started = Instant::now();
            let Ok(request) = receiver.recv() else {
                break;
            };
            metrics
                .idle_request_wait_nanoseconds
                .fetch_add(wait_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            request.into_pending(Instant::now())
        };

        let collection_started = Instant::now();
        let mut inputs = Vec::with_capacity(contract.maximum_batch);
        let mut parts = Vec::<BatchPart>::new();
        let deadline = collection_started + config.maximum_delay;
        let mut deadline_expired = false;
        let mut disconnected = false;
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
                deadline_expired = true;
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(request) => current = request.into_pending(Instant::now()),
                Err(RecvTimeoutError::Timeout) => {
                    deadline_expired = true;
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        let position_count = inputs.len();
        let dispatched = Instant::now();
        let collection_nanoseconds = collection_started.elapsed().as_nanos() as u64;
        let mut queue_nanoseconds = 0_u64;
        let mut channel_nanoseconds = 0_u64;
        let mut broker_queue_nanoseconds = 0_u64;
        for part in &parts {
            let count = part.count as u64;
            let queued = dispatched.duration_since(part.request.queued_at).as_nanos() as u64;
            let channel = part
                .request
                .brokered_at
                .duration_since(part.request.queued_at)
                .as_nanos() as u64;
            let broker = dispatched
                .duration_since(part.request.brokered_at)
                .as_nanos() as u64;
            queue_nanoseconds = queue_nanoseconds.saturating_add(queued.saturating_mul(count));
            channel_nanoseconds = channel_nanoseconds.saturating_add(channel.saturating_mul(count));
            broker_queue_nanoseconds =
                broker_queue_nanoseconds.saturating_add(broker.saturating_mul(count));
        }
        metrics.batches.fetch_add(1, Ordering::Relaxed);
        metrics
            .positions
            .fetch_add(position_count as u64, Ordering::Relaxed);
        metrics
            .maximum_batch
            .fetch_max(position_count, Ordering::Relaxed);
        metrics
            .queue_nanoseconds
            .fetch_add(queue_nanoseconds, Ordering::Relaxed);
        metrics
            .channel_nanoseconds
            .fetch_add(channel_nanoseconds, Ordering::Relaxed);
        metrics
            .broker_queue_nanoseconds
            .fetch_add(broker_queue_nanoseconds, Ordering::Relaxed);
        metrics
            .batch_collection_nanoseconds
            .fetch_add(collection_nanoseconds, Ordering::Relaxed);
        if position_count == contract.maximum_batch {
            metrics.full_batches.fetch_add(1, Ordering::Relaxed);
        } else if deadline_expired {
            metrics.deadline_batches.fetch_add(1, Ordering::Relaxed);
        } else if disconnected {
            metrics.drain_batches.fetch_add(1, Ordering::Relaxed);
        }
        let expected_ids = inputs.iter().map(InferenceInput::id).collect::<Vec<_>>();
        let started = Instant::now();
        let result = service.infer(&inputs);
        let stages = service.last_inference_stages();
        metrics
            .inference_nanoseconds
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        metrics.record_stages(stages);
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

struct QueuedPoolRequest {
    key: u64,
    inputs: std::vec::IntoIter<InferenceInput>,
    next_offset: usize,
    encoding_nanoseconds: u64,
    queued_at: Instant,
    brokered_at: Instant,
}

struct PendingPoolResponse {
    outputs: Vec<Option<InferenceOutput>>,
    remaining: usize,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

struct PoolBatchPart {
    key: u64,
    offset: usize,
    count: usize,
    encoding_nanoseconds: u64,
    queued_at: Instant,
    brokered_at: Instant,
}

struct InFlightPoolBatch {
    expected_ids: Vec<u64>,
    parts: Vec<PoolBatchPart>,
    position_count: usize,
    slot: usize,
}

fn enqueue_pool_request(
    request: Request,
    key: u64,
    queued: &mut VecDeque<QueuedPoolRequest>,
    responses: &mut HashMap<u64, PendingPoolResponse>,
) {
    let brokered_at = Instant::now();
    let count = request.inputs.len();
    debug_assert!(count > 0, "evaluators do not submit empty requests");
    responses.insert(
        key,
        PendingPoolResponse {
            outputs: (0..count).map(|_| None).collect(),
            remaining: count,
            response: request.response,
        },
    );
    queued.push_back(QueuedPoolRequest {
        key,
        inputs: request.inputs.into_iter(),
        next_offset: 0,
        encoding_nanoseconds: request.encoding_nanoseconds,
        queued_at: request.queued_at,
        brokered_at,
    });
}

fn fail_pool_requests(
    error: &EvaluationError,
    responses: &mut HashMap<u64, PendingPoolResponse>,
    receiver: &Receiver<Request>,
) {
    for (_, pending) in responses.drain() {
        let _ = pending.response.send(Err(error.clone()));
    }
    for request in receiver.try_iter() {
        let _ = request.response.send(Err(error.clone()));
    }
}

fn run_pool_broker<S: BatchService + 'static>(
    config: BrokerConfig,
    contract: BatchContract,
    mut executor: ThreadedBatchExecutorPool<S>,
    receiver: Receiver<Request>,
    metrics: Arc<AtomicMetrics>,
    lane_metrics: Vec<Arc<AtomicMetrics>>,
) {
    let mut queued = VecDeque::<QueuedPoolRequest>::new();
    let mut responses = HashMap::<u64, PendingPoolResponse>::new();
    let mut in_flight = HashMap::<u64, InFlightPoolBatch>::new();
    let mut next_request = 0_u64;
    let mut next_batch = 0_u64;
    let mut disconnected = false;

    loop {
        while executor.in_flight() < executor.capacity() {
            if queued.is_empty() && !disconnected {
                let idle = executor.in_flight() == 0;
                let wait_started = Instant::now();
                let received = if idle {
                    receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
                } else {
                    receiver.recv_timeout(config.maximum_delay)
                };
                let waited = wait_started.elapsed().as_nanos() as u64;
                if idle {
                    metrics
                        .idle_request_wait_nanoseconds
                        .fetch_add(waited, Ordering::Relaxed);
                } else {
                    metrics
                        .overlap_request_wait_nanoseconds
                        .fetch_add(waited, Ordering::Relaxed);
                }
                match received {
                    Ok(request) => {
                        enqueue_pool_request(request, next_request, &mut queued, &mut responses);
                        next_request = next_request.wrapping_add(1);
                    }
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => disconnected = true,
                }
            }
            if queued.is_empty() {
                break;
            }

            let collection_started = Instant::now();
            let deadline = collection_started + config.maximum_delay;
            let mut deadline_expired = false;
            let mut inputs = Vec::with_capacity(contract.maximum_batch);
            let mut parts = Vec::<PoolBatchPart>::new();
            loop {
                while inputs.len() < contract.maximum_batch {
                    let Some(current) = queued.front_mut() else {
                        break;
                    };
                    let remaining = current.inputs.len();
                    let count = remaining.min(contract.maximum_batch - inputs.len());
                    let offset = current.next_offset;
                    inputs.extend(current.inputs.by_ref().take(count));
                    current.next_offset += count;
                    let encoded = if count == remaining {
                        current.encoding_nanoseconds
                    } else {
                        ((u128::from(current.encoding_nanoseconds) * count as u128)
                            / remaining as u128) as u64
                    };
                    current.encoding_nanoseconds -= encoded;
                    parts.push(PoolBatchPart {
                        key: current.key,
                        offset,
                        count,
                        encoding_nanoseconds: encoded,
                        queued_at: current.queued_at,
                        brokered_at: current.brokered_at,
                    });
                    if current.inputs.len() == 0 {
                        queued.pop_front();
                    }
                }
                if inputs.len() == contract.maximum_batch || !queued.is_empty() || disconnected {
                    break;
                }
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    deadline_expired = true;
                    break;
                };
                match receiver.recv_timeout(remaining) {
                    Ok(request) => {
                        enqueue_pool_request(request, next_request, &mut queued, &mut responses);
                        next_request = next_request.wrapping_add(1);
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        deadline_expired = true;
                        break;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }

            let position_count = inputs.len();
            let collection_nanoseconds = collection_started.elapsed().as_nanos() as u64;
            let expected_ids = inputs.iter().map(InferenceInput::id).collect::<Vec<_>>();
            let batch = match InferenceBatch::new(next_batch, inputs) {
                Ok(batch) => batch,
                Err(error) => {
                    fail_pool_requests(&error, &mut responses, &receiver);
                    return;
                }
            };
            let submission_started = Instant::now();
            let slot = match executor.submit(batch) {
                Ok(slot) => slot,
                Err(error) => {
                    fail_pool_requests(&error, &mut responses, &receiver);
                    return;
                }
            };
            let submission_nanoseconds = submission_started.elapsed().as_nanos() as u64;
            let dispatched = Instant::now();
            let lane = &lane_metrics[slot];
            let mut queue_nanoseconds = 0_u64;
            let mut channel_nanoseconds = 0_u64;
            let mut broker_queue_nanoseconds = 0_u64;
            let mut encoding_nanoseconds = 0_u64;
            for part in &parts {
                let count = part.count as u64;
                let queued_for = dispatched.duration_since(part.queued_at).as_nanos() as u64;
                let channel_for = part.brokered_at.duration_since(part.queued_at).as_nanos() as u64;
                let broker_for = dispatched.duration_since(part.brokered_at).as_nanos() as u64;
                queue_nanoseconds =
                    queue_nanoseconds.saturating_add(queued_for.saturating_mul(count));
                channel_nanoseconds =
                    channel_nanoseconds.saturating_add(channel_for.saturating_mul(count));
                broker_queue_nanoseconds =
                    broker_queue_nanoseconds.saturating_add(broker_for.saturating_mul(count));
                encoding_nanoseconds =
                    encoding_nanoseconds.saturating_add(part.encoding_nanoseconds);
            }
            metrics.batches.fetch_add(1, Ordering::Relaxed);
            metrics
                .positions
                .fetch_add(position_count as u64, Ordering::Relaxed);
            metrics
                .maximum_batch
                .fetch_max(position_count, Ordering::Relaxed);
            metrics
                .queue_nanoseconds
                .fetch_add(queue_nanoseconds, Ordering::Relaxed);
            metrics
                .channel_nanoseconds
                .fetch_add(channel_nanoseconds, Ordering::Relaxed);
            metrics
                .broker_queue_nanoseconds
                .fetch_add(broker_queue_nanoseconds, Ordering::Relaxed);
            metrics
                .batch_collection_nanoseconds
                .fetch_add(collection_nanoseconds, Ordering::Relaxed);
            metrics
                .batch_submission_nanoseconds
                .fetch_add(submission_nanoseconds, Ordering::Relaxed);
            lane.requests
                .fetch_add(position_count as u64, Ordering::Relaxed);
            lane.batches.fetch_add(1, Ordering::Relaxed);
            lane.positions
                .fetch_add(position_count as u64, Ordering::Relaxed);
            lane.maximum_batch
                .fetch_max(position_count, Ordering::Relaxed);
            lane.encoding_nanoseconds
                .fetch_add(encoding_nanoseconds, Ordering::Relaxed);
            lane.queue_nanoseconds
                .fetch_add(queue_nanoseconds, Ordering::Relaxed);
            lane.channel_nanoseconds
                .fetch_add(channel_nanoseconds, Ordering::Relaxed);
            lane.broker_queue_nanoseconds
                .fetch_add(broker_queue_nanoseconds, Ordering::Relaxed);
            lane.batch_collection_nanoseconds
                .fetch_add(collection_nanoseconds, Ordering::Relaxed);
            lane.batch_submission_nanoseconds
                .fetch_add(submission_nanoseconds, Ordering::Relaxed);
            if position_count == contract.maximum_batch {
                metrics.full_batches.fetch_add(1, Ordering::Relaxed);
                lane.full_batches.fetch_add(1, Ordering::Relaxed);
            } else if deadline_expired {
                metrics.deadline_batches.fetch_add(1, Ordering::Relaxed);
                lane.deadline_batches.fetch_add(1, Ordering::Relaxed);
            } else if disconnected {
                metrics.drain_batches.fetch_add(1, Ordering::Relaxed);
                lane.drain_batches.fetch_add(1, Ordering::Relaxed);
            }
            in_flight.insert(
                next_batch,
                InFlightPoolBatch {
                    expected_ids,
                    parts,
                    position_count,
                    slot,
                },
            );
            next_batch = next_batch.wrapping_add(1);
        }

        if executor.in_flight() == 0 {
            if disconnected && queued.is_empty() {
                break;
            }
            continue;
        }

        let completion_wait_started = Instant::now();
        let completion = match executor.receive() {
            Ok(completion) => completion,
            Err(error) => {
                for batch in in_flight.values() {
                    metrics
                        .failures
                        .fetch_add(batch.position_count as u64, Ordering::Relaxed);
                    lane_metrics[batch.slot]
                        .failures
                        .fetch_add(batch.position_count as u64, Ordering::Relaxed);
                }
                fail_pool_requests(&error, &mut responses, &receiver);
                return;
            }
        };
        metrics.completion_wait_nanoseconds.fetch_add(
            completion_wait_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        let sequence = completion.sequence();
        let slot = completion.slot();
        let elapsed = completion.elapsed().as_nanos() as u64;
        let stages = completion.stages();
        let Some(batch) = in_flight.remove(&sequence) else {
            let error = EvaluationError::new(format!(
                "inference executor returned unknown batch sequence {sequence}"
            ));
            fail_pool_requests(&error, &mut responses, &receiver);
            return;
        };
        debug_assert_eq!(slot, batch.slot);
        metrics
            .inference_nanoseconds
            .fetch_add(elapsed, Ordering::Relaxed);
        lane_metrics[slot]
            .inference_nanoseconds
            .fetch_add(elapsed, Ordering::Relaxed);
        metrics.record_stages(stages);
        lane_metrics[slot].record_stages(stages);
        let result = (|| {
            let outputs = completion.into_outputs();
            if outputs.len() != batch.expected_ids.len() {
                return Err(EvaluationError::new(format!(
                    "inference backend returned {} outputs for {} inputs",
                    outputs.len(),
                    batch.expected_ids.len()
                )));
            }
            for (index, (output, expected_id)) in
                outputs.iter().zip(&batch.expected_ids).enumerate()
            {
                if output.id() != *expected_id {
                    return Err(EvaluationError::new(format!(
                        "inference output {index} has request id {}, expected {expected_id}",
                        output.id()
                    )));
                }
            }
            Ok(outputs)
        })();
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                metrics
                    .failures
                    .fetch_add(batch.position_count as u64, Ordering::Relaxed);
                lane_metrics[slot]
                    .failures
                    .fetch_add(batch.position_count as u64, Ordering::Relaxed);
                fail_pool_requests(&error, &mut responses, &receiver);
                return;
            }
        };

        let mut outputs = outputs.into_iter();
        let mut completed_requests = Vec::new();
        for part in batch.parts {
            let pending = responses
                .get_mut(&part.key)
                .expect("in-flight batch references a pending response");
            for (target, output) in pending.outputs[part.offset..part.offset + part.count]
                .iter_mut()
                .zip(outputs.by_ref().take(part.count))
            {
                debug_assert!(target.is_none(), "each request position completes once");
                *target = Some(output);
            }
            pending.remaining -= part.count;
            if pending.remaining == 0 {
                completed_requests.push(part.key);
            }
        }
        debug_assert!(outputs.next().is_none());
        for key in completed_requests {
            let pending = responses
                .remove(&key)
                .expect("completed response remains registered");
            let ordered = pending
                .outputs
                .into_iter()
                .map(|output| output.expect("all request positions completed"))
                .collect();
            let _ = pending.response.send(Ok(ordered));
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
    use vgo_search::Action;

    struct ConstantService {
        wrong_id: bool,
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

        fn infer(
            &mut self,
            batch: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
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
        delay: Duration,
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
                    encoding_nanoseconds: 0,
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
            BatchedEvaluatorPool::spawn(broker_config(), vec![FailingService, FailingService])
                .unwrap();
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
        let error = match BatchedEvaluatorPool::spawn(broker_config(), Vec::<TaggedService>::new())
        {
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
}
