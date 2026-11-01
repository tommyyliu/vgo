use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use vgo_core::Position;
use vgo_raster::DensePolicy;
use vgo_search::{Evaluation, EvaluationError, Evaluator};

use crate::broker::{Broker, Request};
use crate::metrics::AtomicMetrics;
use crate::{
    BatchContract, BatchService, BrokerConfig, BrokerMetrics, ThreadedBatchExecutorPool,
    encode_request,
};

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
            .name(String::from("vgo-inference-broker"))
            .spawn(move || {
                Broker::new(
                    config,
                    expected,
                    executor,
                    receiver,
                    broker_metrics,
                    broker_lane_metrics,
                )
                .run();
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
                encode_request(id, position, self.inner.contract)
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
