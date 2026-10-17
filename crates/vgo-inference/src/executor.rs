use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use vgo_search::EvaluationError;

use crate::{BatchService, InferenceInput, InferenceOutput, InferenceStageMetrics};

#[derive(Debug)]
pub struct InferenceBatch {
    sequence: u64,
    inputs: Vec<InferenceInput>,
}

impl InferenceBatch {
    pub fn new(sequence: u64, inputs: Vec<InferenceInput>) -> Result<Self, EvaluationError> {
        if inputs.is_empty() {
            return Err(EvaluationError::new("inference batch must not be empty"));
        }
        Ok(Self { sequence, inputs })
    }

    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub fn inputs(&self) -> &[InferenceInput] {
        &self.inputs
    }

    #[must_use]
    pub fn into_parts(self) -> (u64, Vec<InferenceInput>) {
        (self.sequence, self.inputs)
    }
}

#[derive(Debug)]
pub struct CompletedBatch {
    sequence: u64,
    slot: usize,
    elapsed: Duration,
    stages: InferenceStageMetrics,
    outputs: Vec<InferenceOutput>,
}

impl CompletedBatch {
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub const fn slot(&self) -> usize {
        self.slot
    }

    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    #[must_use]
    pub const fn stages(&self) -> InferenceStageMetrics {
        self.stages
    }

    #[must_use]
    pub fn outputs(&self) -> &[InferenceOutput] {
        &self.outputs
    }

    #[must_use]
    pub fn into_outputs(self) -> Vec<InferenceOutput> {
        self.outputs
    }
}

/// Owns one or more inference execution slots.
///
/// `submit` transfers batch ownership and returns once a slot accepts it.
/// `receive` may return completed batches out of submission order, keyed by
/// `sequence`. Actors therefore do not need to know whether an executor uses
/// threads, processes, pinned buffers, CUDA streams, or remote workers.
pub trait BatchExecutor: Send {
    fn capacity(&self) -> usize;
    fn in_flight(&self) -> usize;
    /// Submit to an available execution slot and return that slot's index.
    fn submit(&mut self, batch: InferenceBatch) -> Result<usize, EvaluationError>;
    fn receive(&mut self) -> Result<CompletedBatch, EvaluationError>;
}

pub struct ThreadedBatchExecutor<S: BatchService + 'static> {
    sender: Option<SyncSender<InferenceBatch>>,
    receiver: Receiver<Result<CompletedBatch, EvaluationError>>,
    in_flight: usize,
    join: Option<JoinHandle<()>>,
    service: PhantomData<S>,
    timeout: Duration,
    pending: Option<(u64, Instant, u64, u64)>,
    description: String,
    fatal_native_failure: bool,
}

impl<S: BatchService + 'static> ThreadedBatchExecutor<S> {
    pub fn spawn(service: S) -> Result<Self, EvaluationError> {
        Self::spawn_with_timeout(
            service,
            crate::deadline_from_env("VGO_INFERENCE_TIMEOUT_SECONDS", 120)?,
        )
    }

    pub fn spawn_with_timeout(mut service: S, timeout: Duration) -> Result<Self, EvaluationError> {
        if timeout.is_zero() {
            return Err(EvaluationError::new("inference timeout must be positive"));
        }
        let description = service.description();
        let fatal_native_failure = service.fatal_native_failure();
        let worker_description = description.clone();
        let (batch_sender, batch_receiver) = mpsc::sync_channel::<InferenceBatch>(1);
        let (completion_sender, completion_receiver) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-executor"))
            .spawn(move || {
                while let Ok(batch) = batch_receiver.recv() {
                    let sequence = batch.sequence;
                    let started = Instant::now();
                    let result = service.infer(&batch.inputs).map_err(|error| {
                        EvaluationError::new(format!(
                            "{worker_description} slot 0 batch {sequence}: {error}"
                        ))
                    });
                    let stages = service.last_inference_stages();
                    let completion = result.map(|outputs| CompletedBatch {
                        sequence,
                        slot: 0,
                        elapsed: started.elapsed(),
                        stages,
                        outputs,
                    });
                    let failed = completion.is_err();
                    if completion_sender.send(completion).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|error| EvaluationError::new(format!("start batch executor: {error}")))?;
        Ok(Self {
            sender: Some(batch_sender),
            receiver: completion_receiver,
            in_flight: 0,
            join: Some(join),
            service: PhantomData,
            timeout,
            pending: None,
            description,
            fatal_native_failure,
        })
    }
}

impl<S: BatchService + 'static> BatchExecutor for ThreadedBatchExecutor<S> {
    fn capacity(&self) -> usize {
        1
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn submit(&mut self, batch: InferenceBatch) -> Result<usize, EvaluationError> {
        if self.in_flight >= self.capacity() {
            return Err(EvaluationError::new("batch executor has no available slot"));
        }
        let pending = (
            batch.sequence(),
            Instant::now(),
            batch.inputs()[0].id(),
            batch.inputs().last().unwrap().id(),
        );
        self.sender
            .as_ref()
            .ok_or_else(|| EvaluationError::new("inference executor stopped after a fatal error"))?
            .send(batch)
            .map_err(|_| EvaluationError::new("batch executor has stopped"))?;
        self.in_flight += 1;
        self.pending = Some(pending);
        Ok(0)
    }

    fn receive(&mut self) -> Result<CompletedBatch, EvaluationError> {
        if self.in_flight == 0 {
            return Err(EvaluationError::new("batch executor has no in-flight work"));
        }
        let (sequence, started, first, last) = self.pending.expect("in-flight batch metadata");
        let completion = match self
            .receiver
            .recv_timeout(self.timeout.saturating_sub(started.elapsed()))
        {
            Ok(completion) => completion,
            Err(error) => {
                if self.fatal_native_failure {
                    crate::deadline::arm_fatal_watchdog();
                }
                // A native call cannot be interrupted safely. Disconnect and
                // detach it so callers and process shutdown can finish. Fatal:
                // the owner must exit rather than reopen GPU sessions here.
                self.sender.take();
                self.join.take();
                self.pending = None;
                self.in_flight = 0;
                return Err(EvaluationError::new(format!(
                    "{} slot 0 batch {sequence}, request IDs {first}..{last}, elapsed {:.3}s (limit {:.3}s): {error}",
                    self.description,
                    started.elapsed().as_secs_f64(),
                    self.timeout.as_secs_f64()
                )));
            }
        };
        self.in_flight -= 1;
        self.pending = None;
        if completion.is_err() && self.fatal_native_failure {
            crate::deadline::arm_fatal_watchdog();
        }
        completion
    }
}

struct ServiceWorker {
    sender: Option<SyncSender<InferenceBatch>>,
    join: Option<JoinHandle<()>>,
    description: String,
    fatal_native_failure: bool,
}

impl Drop for ServiceWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// A set of independent synchronous services exposed as one asynchronous
/// executor. Batches are assigned only to idle slots and completions share one
/// receiver, so a slower slot cannot head-of-line block a faster one.
pub struct ThreadedBatchExecutorPool<S: BatchService + 'static> {
    workers: Vec<ServiceWorker>,
    completion_receiver: Receiver<Result<CompletedBatch, EvaluationError>>,
    available: Vec<usize>,
    in_flight: usize,
    service: PhantomData<S>,
    timeout: Duration,
    pending: HashMap<u64, (usize, Instant, u64, u64)>,
}

impl<S: BatchService + 'static> ThreadedBatchExecutorPool<S> {
    pub fn spawn(services: Vec<S>) -> Result<Self, EvaluationError> {
        Self::spawn_with_timeout(
            services,
            crate::deadline_from_env("VGO_INFERENCE_TIMEOUT_SECONDS", 120)?,
        )
    }

    pub fn spawn_with_timeout(
        services: Vec<S>,
        timeout: Duration,
    ) -> Result<Self, EvaluationError> {
        if timeout.is_zero() {
            return Err(EvaluationError::new("inference timeout must be positive"));
        }
        if services.is_empty() {
            return Err(EvaluationError::new(
                "inference executor pool must contain at least one service",
            ));
        }
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut workers = Vec::with_capacity(services.len());
        for (slot, mut service) in services.into_iter().enumerate() {
            let description = service.description();
            let fatal_native_failure = service.fatal_native_failure();
            let worker_description = description.clone();
            let (sender, receiver) = mpsc::sync_channel::<InferenceBatch>(1);
            let completions = completion_sender.clone();
            let join = thread::Builder::new()
                .name(format!("vgo-inference-executor-{slot}"))
                .spawn(move || {
                    while let Ok(batch) = receiver.recv() {
                        let sequence = batch.sequence;
                        let started = Instant::now();
                        let result = service.infer(&batch.inputs).map_err(|error| {
                            EvaluationError::new(format!(
                                "{worker_description} slot {slot} batch {sequence}: {error}"
                            ))
                        });
                        let stages = service.last_inference_stages();
                        let completion = result.map(|outputs| CompletedBatch {
                            sequence,
                            slot,
                            elapsed: started.elapsed(),
                            stages,
                            outputs,
                        });
                        let failed = completion.is_err();
                        if completions.send(completion).is_err() || failed {
                            break;
                        }
                    }
                })
                .map_err(|error| {
                    EvaluationError::new(format!("start inference executor {slot}: {error}"))
                })?;
            workers.push(ServiceWorker {
                sender: Some(sender),
                join: Some(join),
                description,
                fatal_native_failure,
            });
        }
        drop(completion_sender);
        let available = (0..workers.len()).rev().collect();
        Ok(Self {
            workers,
            completion_receiver,
            available,
            in_flight: 0,
            service: PhantomData,
            timeout,
            pending: HashMap::new(),
        })
    }
}

impl<S: BatchService + 'static> BatchExecutor for ThreadedBatchExecutorPool<S> {
    fn capacity(&self) -> usize {
        self.workers.len()
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn submit(&mut self, batch: InferenceBatch) -> Result<usize, EvaluationError> {
        let Some(slot) = self.available.pop() else {
            return Err(EvaluationError::new("batch executor has no available slot"));
        };
        let pending = (
            slot,
            Instant::now(),
            batch.inputs()[0].id(),
            batch.inputs().last().unwrap().id(),
        );
        let sequence = batch.sequence();
        let result = self.workers[slot]
            .sender
            .as_ref()
            .expect("sender exists while executor pool is alive")
            .send(batch)
            .map_err(|_| EvaluationError::new(format!("inference executor {slot} has stopped")));
        if let Err(error) = result {
            self.available.push(slot);
            return Err(error);
        }
        self.in_flight += 1;
        self.pending.insert(sequence, pending);
        Ok(slot)
    }

    fn receive(&mut self) -> Result<CompletedBatch, EvaluationError> {
        if self.in_flight == 0 {
            return Err(EvaluationError::new("batch executor has no in-flight work"));
        }
        let (&sequence, &(slot, started, first, last)) = self
            .pending
            .iter()
            .min_by_key(|(_, (_, started, _, _))| *started)
            .expect("in-flight metadata");
        let result = if started.elapsed() >= self.timeout {
            Err(mpsc::RecvTimeoutError::Timeout)
        } else {
            self.completion_receiver
                .recv_timeout(self.timeout.saturating_sub(started.elapsed()))
        };
        let completion = match result {
            Ok(Ok(completion)) => completion,
            error => {
                if self
                    .workers
                    .iter()
                    .any(|worker| worker.fatal_native_failure)
                {
                    crate::deadline::arm_fatal_watchdog();
                }
                for worker in &mut self.workers {
                    worker.sender.take();
                    worker.join.take();
                }
                self.pending.clear();
                self.available.clear();
                self.in_flight = 0;
                return Err(match error {
                    Ok(Err(error)) => error,
                    Err(error) => EvaluationError::new(format!(
                        "{} slot {slot} batch {sequence}, request IDs {first}..{last}, elapsed {:.3}s (limit {:.3}s): {error}",
                        self.workers[slot].description,
                        started.elapsed().as_secs_f64(),
                        self.timeout.as_secs_f64()
                    )),
                    Ok(Ok(_)) => unreachable!(),
                });
            }
        };
        self.in_flight -= 1;
        self.pending.remove(&completion.sequence);
        self.available.push(completion.slot);
        Ok(completion)
    }
}

impl<S: BatchService + 'static> Drop for ThreadedBatchExecutorPool<S> {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.sender.take();
        }
        for worker in &mut self.workers {
            if let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }
}

impl<S: BatchService + 'static> Drop for ThreadedBatchExecutor<S> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };
    use vgo_core::{Color, Position};
    use vgo_raster::{RasterConfig, rasterize};

    use super::{
        BatchExecutor, CompletedBatch, InferenceBatch, ThreadedBatchExecutor,
        ThreadedBatchExecutorPool,
    };
    use crate::{BatchContract, BatchService, InferenceInput, InferenceOutput, InputLayout};
    use vgo_search::EvaluationError;

    struct EchoService;

    struct GatedService(Option<mpsc::Receiver<()>>);
    impl BatchService for GatedService {
        fn contract(&self) -> BatchContract {
            EchoService.contract()
        }
        fn description(&self) -> String {
            "model=gated-test provider=test".into()
        }
        fn infer(
            &mut self,
            inputs: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
            if let Some(gate) = self.0.take() {
                let _ = gate.recv();
            }
            EchoService.infer(inputs)
        }
    }

    #[test]
    fn single_slot_timeout_and_drop_do_not_wait_for_a_stalled_native_call() {
        let (release, gate) = mpsc::channel();
        let mut executor = ThreadedBatchExecutor::spawn_with_timeout(
            GatedService(Some(gate)),
            Duration::from_millis(30),
        )
        .unwrap();
        executor.submit(batch(7, 11)).unwrap();
        let started = Instant::now();
        let error = executor.receive().unwrap_err().to_string();
        assert!(
            error.contains("model=gated-test provider=test slot 0 batch 7, request IDs 11..11")
        );
        assert!(executor.submit(batch(8, 12)).is_err());
        drop(executor);
        assert!(started.elapsed() < Duration::from_secs(1));
        release.send(()).unwrap();
    }

    #[test]
    fn fast_pool_slot_cannot_hide_an_old_stalled_batch() {
        let (release, gate) = mpsc::channel();
        let mut executor = ThreadedBatchExecutorPool::spawn_with_timeout(
            vec![GatedService(Some(gate)), GatedService(None)],
            Duration::from_millis(100),
        )
        .unwrap();
        executor.submit(batch(7, 11)).unwrap();
        executor.submit(batch(8, 12)).unwrap();
        assert_eq!(executor.receive().unwrap().sequence(), 8);
        let started = Instant::now();
        let error = executor.receive().unwrap_err().to_string();
        assert!(error.contains("slot 0 batch 7, request IDs 11..11"));
        assert!(executor.submit(batch(9, 13)).is_err());
        drop(executor);
        assert!(started.elapsed() < Duration::from_secs(1));
        release.send(()).unwrap();
    }

    impl BatchService for EchoService {
        fn contract(&self) -> BatchContract {
            BatchContract {
                raster: RasterConfig::square(2),
                policy: RasterConfig::square(2),
                maximum_batch: 1,
                input_layout: InputLayout::Dense,
            }
        }

        fn infer(
            &mut self,
            batch: &[InferenceInput],
        ) -> Result<Vec<InferenceOutput>, EvaluationError> {
            batch
                .iter()
                .map(|input| InferenceOutput::new(input.id(), 0.25, vec![1.0]))
                .collect()
        }
    }

    fn batch(sequence: u64, request: u64) -> InferenceBatch {
        let position = Position::new(0.1, Vec::new(), Color::Black);
        let raster = rasterize(&position, RasterConfig::square(2));
        InferenceBatch::new(sequence, vec![InferenceInput::new(request, raster)]).unwrap()
    }

    #[test]
    fn executor_tracks_capacity_and_sequence() {
        let mut executor = ThreadedBatchExecutor::spawn(EchoService).unwrap();
        assert_eq!(executor.submit(batch(7, 11)).unwrap(), 0);
        assert_eq!(executor.in_flight(), 1);
        assert!(executor.submit(batch(8, 12)).is_err());
        let completed = executor.receive().unwrap();
        assert_eq!(completed.sequence(), 7);
        assert_eq!(completed.slot(), 0);
        assert_eq!(completed.outputs()[0].id(), 11);
        assert_eq!(executor.in_flight(), 0);
    }

    #[test]
    fn executor_pool_uses_each_available_slot() {
        let mut executor =
            ThreadedBatchExecutorPool::spawn(vec![EchoService, EchoService]).unwrap();
        let first = executor.submit(batch(7, 11)).unwrap();
        let second = executor.submit(batch(8, 12)).unwrap();
        assert_ne!(first, second);
        assert_eq!(executor.capacity(), 2);
        assert_eq!(executor.in_flight(), 2);

        let mut completed = [executor.receive().unwrap(), executor.receive().unwrap()];
        completed.sort_by_key(CompletedBatch::sequence);
        assert_eq!(completed[0].outputs()[0].id(), 11);
        assert_eq!(completed[1].outputs()[0].id(), 12);
        assert_eq!(executor.in_flight(), 0);
    }
}
