#![forbid(unsafe_code)]

use std::{
    collections::{HashMap, HashSet},
    io::{BufReader, BufWriter, Read, Write},
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
use vgo_raster::{CHANNEL_COUNT, RasterConfig, action_pixel, rasterize};
use vgo_search::{Action, Evaluation, EvaluationError, Evaluator, Policy};

const REQUEST_MAGIC: [u8; 8] = *b"VGOIFR01";
const RESPONSE_MAGIC: [u8; 8] = *b"VGOOFR01";
const PROTOCOL_VERSION: u32 = 1;

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
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BrokerMetrics {
    pub requests: u64,
    pub batches: u64,
    pub positions: u64,
    pub maximum_batch: usize,
    pub failures: u64,
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
            queue_nanoseconds: self.queue_nanoseconds.load(Ordering::Relaxed),
            inference_nanoseconds: self.inference_nanoseconds.load(Ordering::Relaxed),
        }
    }
}

struct Request {
    id: u64,
    position: Position,
    queued_at: Instant,
    response: mpsc::Sender<Result<ModelOutput, EvaluationError>>,
}

struct ModelOutput {
    current_value: f64,
    policy: Vec<f32>,
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
pub struct PythonEvaluator {
    inner: Arc<Inner>,
}

impl PythonEvaluator {
    pub fn spawn(config: PythonProcessConfig) -> Result<Self, EvaluationError> {
        if config.maximum_batch == 0 || config.queue_capacity == 0 {
            return Err(EvaluationError::new(
                "batch size and queue capacity must be positive",
            ));
        }
        let mut child = Command::new(&config.python)
            .current_dir(&config.working_directory)
            .arg("-m")
            .arg("vgo_training.serve")
            .arg("--checkpoint")
            .arg(&config.checkpoint)
            .arg("--threads")
            .arg(config.torch_threads.to_string())
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
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let metrics = Arc::new(AtomicMetrics::default());
        let broker_metrics = Arc::clone(&metrics);
        let raster = config.raster;
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-broker"))
            .spawn(move || run_broker(config, child, stdin, stdout, receiver, broker_metrics))
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

impl Evaluator for PythonEvaluator {
    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (response_sender, response_receiver) = mpsc::channel();
        self.inner.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.inner
            .sender
            .as_ref()
            .expect("sender exists while evaluator is alive")
            .send(Request {
                id,
                position: position.clone(),
                queued_at: Instant::now(),
                response: response_sender,
            })
            .map_err(|_| EvaluationError::new("inference broker has stopped"))?;
        let output = response_receiver
            .recv()
            .map_err(|_| EvaluationError::new("inference broker dropped the response"))??;
        Ok(Evaluation::new(
            output.current_value,
            Box::new(DensePolicy {
                config: self.inner.raster,
                logits: output.policy,
            }),
        ))
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
    config: PythonProcessConfig,
    mut child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    receiver: Receiver<Request>,
    metrics: Arc<AtomicMetrics>,
) {
    let mut writer = BufWriter::new(stdin);
    let mut reader = BufReader::new(stdout);
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
        let started = Instant::now();
        let result = exchange_batch(&config, &batch, &mut writer, &mut reader);
        metrics
            .inference_nanoseconds
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match result {
            Ok(outputs) => {
                for (request, output) in batch.into_iter().zip(outputs) {
                    let _ = request.response.send(Ok(output));
                }
            }
            Err(error) => {
                metrics
                    .failures
                    .fetch_add(batch.len() as u64, Ordering::Relaxed);
                for request in batch {
                    let _ = request.response.send(Err(error.clone()));
                }
                break;
            }
        }
    }
    drop(writer);
    let _ = child.wait();
}

fn exchange_batch(
    config: &PythonProcessConfig,
    batch: &[Request],
    writer: &mut impl Write,
    reader: &mut impl Read,
) -> Result<Vec<ModelOutput>, EvaluationError> {
    writer
        .write_all(&REQUEST_MAGIC)
        .and_then(|()| write_u32(writer, PROTOCOL_VERSION))
        .and_then(|()| write_u32(writer, batch.len() as u32))
        .and_then(|()| write_u32(writer, CHANNEL_COUNT as u32))
        .and_then(|()| write_u32(writer, config.raster.height as u32))
        .and_then(|()| write_u32(writer, config.raster.width as u32))
        .map_err(io_error)?;
    for request in batch {
        writer
            .write_all(&request.id.to_le_bytes())
            .map_err(io_error)?;
        let raster = rasterize(&request.position, config.raster);
        for value in raster.data() {
            writer.write_all(&value.to_le_bytes()).map_err(io_error)?;
        }
    }
    writer.flush().map_err(io_error)?;

    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic).map_err(io_error)?;
    if magic != RESPONSE_MAGIC {
        return Err(EvaluationError::new("invalid inference response magic"));
    }
    let version = read_u32(reader)?;
    let count = read_u32(reader)? as usize;
    let policy_size = read_u32(reader)? as usize;
    if version != PROTOCOL_VERSION || count != batch.len() {
        return Err(EvaluationError::new("inference response header mismatch"));
    }
    if policy_size != config.raster.pixels() + 1 {
        return Err(EvaluationError::new("inference policy size mismatch"));
    }
    let expected_ids = batch
        .iter()
        .map(|request| request.id)
        .collect::<HashSet<_>>();
    let mut outputs = HashMap::with_capacity(count);
    for _ in 0..count {
        let id = read_u64(reader)?;
        if !expected_ids.contains(&id) {
            return Err(EvaluationError::new("unexpected inference response ID"));
        }
        let current_value = f64::from(read_f32(reader)?);
        if !current_value.is_finite() || !(-1.0..=1.0).contains(&current_value) {
            return Err(EvaluationError::new("invalid inference value"));
        }
        let mut policy = Vec::with_capacity(policy_size);
        for _ in 0..policy_size {
            let logit = read_f32(reader)?;
            if !logit.is_finite() {
                return Err(EvaluationError::new("non-finite policy logit"));
            }
            policy.push(logit);
        }
        if outputs
            .insert(
                id,
                ModelOutput {
                    current_value,
                    policy,
                },
            )
            .is_some()
        {
            return Err(EvaluationError::new("duplicate inference response ID"));
        }
    }
    batch
        .iter()
        .map(|request| {
            outputs
                .remove(&request.id)
                .ok_or_else(|| EvaluationError::new("missing inference response ID"))
        })
        .collect()
}

fn io_error(error: std::io::Error) -> EvaluationError {
    EvaluationError::new(format!("inference transport: {error}"))
}

fn write_u32(writer: &mut impl Write, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> Result<u32, EvaluationError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, EvaluationError> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32, EvaluationError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::PathBuf, sync::mpsc, time::Duration};

    use vgo_core::{Color, Position};
    use vgo_raster::RasterConfig;

    use super::{PROTOCOL_VERSION, PythonProcessConfig, RESPONSE_MAGIC, Request, exchange_batch};

    fn config() -> PythonProcessConfig {
        PythonProcessConfig {
            python: PathBuf::new(),
            working_directory: PathBuf::new(),
            checkpoint: PathBuf::new(),
            raster: RasterConfig::square(2),
            maximum_batch: 2,
            maximum_delay: Duration::ZERO,
            queue_capacity: 2,
            torch_threads: 1,
        }
    }

    #[test]
    fn responses_are_routed_by_identifier() {
        let (sender, _receiver) = mpsc::channel();
        let position = Position::new(0.1, Vec::new(), Color::Black);
        let batch = vec![
            Request {
                id: 10,
                position: position.clone(),
                queued_at: std::time::Instant::now(),
                response: sender.clone(),
            },
            Request {
                id: 20,
                position,
                queued_at: std::time::Instant::now(),
                response: sender,
            },
        ];
        let policy_size = 5_u32;
        let mut response = Vec::new();
        response.extend_from_slice(&RESPONSE_MAGIC);
        response.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        response.extend_from_slice(&2_u32.to_le_bytes());
        response.extend_from_slice(&policy_size.to_le_bytes());
        for (id, value, logit) in [(20_u64, -0.25_f32, 2.0_f32), (10, 0.5, 1.0)] {
            response.extend_from_slice(&id.to_le_bytes());
            response.extend_from_slice(&value.to_le_bytes());
            for _ in 0..policy_size {
                response.extend_from_slice(&logit.to_le_bytes());
            }
        }
        let mut request_bytes = Vec::new();
        let outputs = exchange_batch(
            &config(),
            &batch,
            &mut request_bytes,
            &mut Cursor::new(response),
        )
        .expect("valid framed response");

        assert_eq!(&request_bytes[..8], b"VGOIFR01");
        assert_eq!(outputs[0].current_value, 0.5);
        assert_eq!(outputs[0].policy, vec![1.0; policy_size as usize]);
        assert_eq!(outputs[1].current_value, -0.25);
        assert_eq!(outputs[1].policy, vec![2.0; policy_size as usize]);
    }
}
