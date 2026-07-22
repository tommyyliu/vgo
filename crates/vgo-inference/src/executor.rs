use std::{
    marker::PhantomData,
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
};

use vgo_search::EvaluationError;

use crate::{BatchService, InferenceInput, InferenceOutput};

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
    outputs: Vec<InferenceOutput>,
}

impl CompletedBatch {
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
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
    fn submit(&mut self, batch: InferenceBatch) -> Result<(), EvaluationError>;
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
                    let completion = service
                        .infer(&batch.inputs)
                        .map(|outputs| CompletedBatch { sequence, outputs });
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

    fn submit(&mut self, batch: InferenceBatch) -> Result<(), EvaluationError> {
        if self.in_flight >= self.capacity() {
            return Err(EvaluationError::new("batch executor has no available slot"));
        }
        self.sender
            .as_ref()
            .expect("sender exists while executor is alive")
            .send(batch)
            .map_err(|_| EvaluationError::new("batch executor has stopped"))?;
        self.in_flight += 1;
        Ok(())
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

    use super::{BatchExecutor, InferenceBatch, ThreadedBatchExecutor};
    use crate::{BatchContract, BatchService, InferenceInput, InferenceOutput};
    use vgo_search::EvaluationError;

    struct EchoService;

    impl BatchService for EchoService {
        fn contract(&self) -> BatchContract {
            BatchContract {
                raster: RasterConfig::square(2),
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
        executor.submit(batch(7, 11)).unwrap();
        assert_eq!(executor.in_flight(), 1);
        assert!(executor.submit(batch(8, 12)).is_err());
        let completed = executor.receive().unwrap();
        assert_eq!(completed.sequence(), 7);
        assert_eq!(completed.outputs()[0].id(), 11);
        assert_eq!(executor.in_flight(), 0);
    }
}
