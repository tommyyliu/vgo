use std::{
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
}

impl<S: BatchService + 'static> ThreadedBatchExecutor<S> {
    pub fn spawn(mut service: S) -> Result<Self, EvaluationError> {
        let (batch_sender, batch_receiver) = mpsc::sync_channel::<InferenceBatch>(1);
        let (completion_sender, completion_receiver) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name(String::from("vgo-inference-executor"))
            .spawn(move || {
                while let Ok(batch) = batch_receiver.recv() {
                    let sequence = batch.sequence;
                    let started = Instant::now();
                    let result = service.infer(&batch.inputs);
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
        self.sender
            .as_ref()
            .expect("sender exists while executor is alive")
            .send(batch)
            .map_err(|_| EvaluationError::new("batch executor has stopped"))?;
        self.in_flight += 1;
        Ok(0)
    }

    fn receive(&mut self) -> Result<CompletedBatch, EvaluationError> {
        if self.in_flight == 0 {
            return Err(EvaluationError::new("batch executor has no in-flight work"));
        }
        let completion = self
            .receiver
            .recv()
            .map_err(|_| EvaluationError::new("batch executor dropped its completion"))?;
        self.in_flight -= 1;
        completion
    }
}

struct ServiceWorker {
    sender: Option<SyncSender<InferenceBatch>>,
    join: Option<JoinHandle<()>>,
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
}

impl<S: BatchService + 'static> ThreadedBatchExecutorPool<S> {
    pub fn spawn(services: Vec<S>) -> Result<Self, EvaluationError> {
        if services.is_empty() {
            return Err(EvaluationError::new(
                "inference executor pool must contain at least one service",
            ));
        }
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut workers = Vec::with_capacity(services.len());
        for (slot, mut service) in services.into_iter().enumerate() {
            let (sender, receiver) = mpsc::sync_channel::<InferenceBatch>(1);
            let completions = completion_sender.clone();
            let join = thread::Builder::new()
                .name(format!("vgo-inference-executor-{slot}"))
                .spawn(move || {
                    while let Ok(batch) = receiver.recv() {
                        let sequence = batch.sequence;
                        let started = Instant::now();
                        let result = service.infer(&batch.inputs);
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
        Ok(slot)
    }

    fn receive(&mut self) -> Result<CompletedBatch, EvaluationError> {
        if self.in_flight == 0 {
            return Err(EvaluationError::new("batch executor has no in-flight work"));
        }
        let completion = self
            .completion_receiver
            .recv()
            .map_err(|_| EvaluationError::new("inference executor pool stopped"))??;
        self.in_flight -= 1;
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
    use vgo_core::{Color, Position};
    use vgo_raster::{RasterConfig, rasterize};

    use super::{
        BatchExecutor, CompletedBatch, InferenceBatch, ThreadedBatchExecutor,
        ThreadedBatchExecutorPool,
    };
    use crate::{BatchContract, BatchService, InferenceInput, InferenceOutput};
    use vgo_search::EvaluationError;

    struct EchoService;

    impl BatchService for EchoService {
        fn contract(&self) -> BatchContract {
            BatchContract {
                raster: RasterConfig::square(2),
                policy: RasterConfig::square(2),
                maximum_batch: 1,
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
