#![forbid(unsafe_code)]

mod executor;
mod onnx;
mod protocol;

pub use executor::{BatchExecutor, CompletedBatch, InferenceBatch, ThreadedBatchExecutor};
pub use onnx::{OnnxBatchService, OnnxProvider, OnnxServiceConfig};
pub use protocol::{InferenceInput, InferenceOutput, encode_request_frame, read_response_frame};

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
use vgo_raster::{RasterConfig, action_pixel, rasterize};
use vgo_search::{Action, Evaluation, EvaluationError, Evaluator, Policy};

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
    pub maximum_batch: usize,
    pub maximum_delay: Duration,
    pub queue_capacity: usize,
    pub torch_threads: usize,
    pub device: TorchDevice,
    pub compile: bool,
}

pub trait BatchService: Send {
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError>;
}

#[derive(Clone, Copy, Debug)]
pub struct BrokerConfig {
    pub raster: RasterConfig,
    pub maximum_batch: usize,
    pub maximum_delay: Duration,
    pub queue_capacity: usize,
}

impl From<&PythonProcessConfig> for BrokerConfig {
    fn from(config: &PythonProcessConfig) -> Self {
        Self {
            raster: config.raster,
            maximum_batch: config.maximum_batch,
            maximum_delay: config.maximum_delay,
            queue_capacity: config.queue_capacity,
        }
    }
}

pub struct PythonBatchService {
    child: Child,
    writer: Option<BufWriter<ChildStdin>>,
    reader: BufReader<ChildStdout>,
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
        })
    }
}

impl BatchService for PythonBatchService {
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        let frame = encode_request_frame(batch)?;
        let writer = self
            .writer
            .as_mut()
            .expect("writer exists while service is alive");
        writer.write_all(&frame).map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        read_response_frame(&mut self.reader, batch)
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
    input: InferenceInput,
    queued_at: Instant,
    response: mpsc::Sender<Result<InferenceOutput, EvaluationError>>,
}

struct Inner {
    sender: Option<SyncSender<Request>>,
    next_id: AtomicU64,
    raster: RasterConfig,
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
        if config.maximum_batch == 0 || config.queue_capacity == 0 {
            return Err(EvaluationError::new(
                "batch size and queue capacity must be positive",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let metrics = Arc::new(AtomicMetrics::default());
        let broker_metrics = Arc::clone(&metrics);
        let raster = config.raster;
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-broker"))
            .spawn(move || run_broker(config, service, receiver, broker_metrics))
            .map_err(|error| EvaluationError::new(format!("start inference broker: {error}")))?;
        Ok(Self {
            inner: Arc::new(Inner {
                sender: Some(sender),
                next_id: AtomicU64::new(1),
                raster,
                metrics,
                join: Mutex::new(Some(join)),
            }),
        })
    }

    #[must_use]
    pub fn metrics(&self) -> BrokerMetrics {
        self.inner.metrics.snapshot()
    }
}

impl Evaluator for BatchedEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let encoding_started = Instant::now();
        let raster = rasterize(position, self.inner.raster);
        self.inner.metrics.encoding_nanoseconds.fetch_add(
            encoding_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        let (response_sender, response_receiver) = mpsc::channel();
        self.inner.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.inner
            .sender
            .as_ref()
            .expect("sender exists while evaluator is alive")
            .send(Request {
                input: InferenceInput::new(id, raster),
                queued_at: Instant::now(),
                response: response_sender,
            })
            .map_err(|_| EvaluationError::new("inference broker has stopped"))?;
        let output = response_receiver
            .recv()
            .map_err(|_| EvaluationError::new("inference broker dropped the response"))??;
        let (_, current_value, policy) = output.into_parts();
        Ok(Evaluation::new(
            current_value,
            Box::new(DensePolicy {
                config: self.inner.raster,
                logits: policy,
            }),
        ))
    }
}

#[derive(Clone)]
pub struct PythonEvaluator {
    evaluator: BatchedEvaluator,
}

impl PythonEvaluator {
    pub fn spawn(config: PythonProcessConfig) -> Result<Self, EvaluationError> {
        let service = PythonBatchService::spawn(&config)?;
        Ok(Self {
            evaluator: BatchedEvaluator::spawn(BrokerConfig::from(&config), service)?,
        })
    }

    #[must_use]
    pub fn metrics(&self) -> BrokerMetrics {
        self.evaluator.metrics()
    }
}

impl Evaluator for PythonEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        self.evaluator.evaluate(position)
    }
}

struct DensePolicy {
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
}

fn run_broker(
    config: BrokerConfig,
    mut service: impl BatchService,
    receiver: Receiver<Request>,
    metrics: Arc<AtomicMetrics>,
) {
    while let Ok(first) = receiver.recv() {
        let mut batch = Vec::with_capacity(config.maximum_batch);
        batch.push(first);
        let deadline = Instant::now() + config.maximum_delay;
        while batch.len() < config.maximum_batch {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(request) => batch.push(request),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }
        let now = Instant::now();
        for request in &batch {
            metrics.queue_nanoseconds.fetch_add(
                now.duration_since(request.queued_at).as_nanos() as u64,
                Ordering::Relaxed,
            );
        }
        metrics.batches.fetch_add(1, Ordering::Relaxed);
        metrics
            .positions
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        metrics
            .maximum_batch
            .fetch_max(batch.len(), Ordering::Relaxed);
        let mut inputs = Vec::with_capacity(batch.len());
        let mut responses = Vec::with_capacity(batch.len());
        for request in batch {
            inputs.push(request.input);
            responses.push(request.response);
        }
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
                for (response, output) in responses.into_iter().zip(outputs) {
                    let _ = response.send(Ok(output));
                }
            }
            Err(error) => {
                metrics
                    .failures
                    .fetch_add(responses.len() as u64, Ordering::Relaxed);
                for response in responses {
                    let _ = response.send(Err(error.clone()));
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

    fn broker_config() -> BrokerConfig {
        BrokerConfig {
            raster: RasterConfig::square(2),
            maximum_batch: 2,
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
}
