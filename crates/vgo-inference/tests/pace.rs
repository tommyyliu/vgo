//! A paced lane rests after each batch. A process of its own, because the duty
//! is one setting per process and would slow every other test's lanes.

use std::{
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, Position};
use vgo_inference::{
    BatchContract, BatchService, InferenceBatch, InferenceInput, InferenceOutput, InputLayout,
    ThreadedBatchExecutorPool, pace,
};
use vgo_raster::{RasterConfig, rasterize};
use vgo_search::EvaluationError;

const BATCH_TIME: Duration = Duration::from_millis(20);

/// `staged` packs on a stager thread, as the GPU service does.
struct SlowService {
    staged: bool,
}
impl BatchService for SlowService {
    fn contract(&self) -> BatchContract {
        BatchContract {
            raster: RasterConfig::square(2),
            policy: RasterConfig::square(2),
            maximum_batch: 1,
            input_layout: InputLayout::Dense,
        }
    }
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        thread::sleep(BATCH_TIME);
        batch
            .iter()
            .map(|input| InferenceOutput::new(input.id(), 0.0, vec![1.0]))
            .collect()
    }
    fn stager(&mut self) -> Option<vgo_inference::Stager> {
        let stager: vgo_inference::Stager = Box::new(|_| Ok(Box::new(())));
        self.staged.then_some(stager)
    }
    fn infer_staged(
        &mut self,
        _staged: vgo_inference::StagedBatch,
        ids: Vec<u64>,
    ) -> Result<vgo_inference::FinishBatch, EvaluationError> {
        thread::sleep(BATCH_TIME);
        let outputs = ids
            .into_iter()
            .map(|id| InferenceOutput::new(id, 0.0, vec![1.0]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(move || Ok(outputs)))
    }
}

/// Run batches back to back through one lane and time them.
fn time_batches(count: u64, staged: bool) -> Duration {
    let mut executor = ThreadedBatchExecutorPool::spawn(vec![SlowService { staged }]).unwrap();
    let position = Position::new(0.1, Vec::new(), Color::Black);
    let started = Instant::now();
    for sequence in 0..count {
        let input = InferenceInput::new(sequence, rasterize(&position, RasterConfig::square(2)));
        executor
            .submit(InferenceBatch::new(sequence, vec![input]).unwrap())
            .unwrap();
        let completed = executor.receive().unwrap();
        // The reply is not held back by the rest that follows it.
        assert!(completed.elapsed() < BATCH_TIME * 2);
    }
    started.elapsed()
}

#[test]
fn a_lane_at_half_duty_takes_twice_as_long() {
    let batches = 6;
    for staged in [false, true] {
        pace::set_duty_percent(100).unwrap();
        let full = time_batches(batches, staged);
        assert!(
            full < BATCH_TIME * 9,
            "staged {staged}: full speed took {full:?}"
        );

        pace::set_duty_percent(50).unwrap();
        let half = time_batches(batches, staged);
        // Each batch but the last is followed by a rest as long as itself.
        assert!(
            half >= BATCH_TIME * 11,
            "staged {staged}: half duty took {half:?}"
        );
    }

    assert!(pace::set_duty_percent(5).is_err());
    assert!(pace::set_duty_percent(101).is_err());
    assert_eq!(pace::duty_percent(), 50);
    pace::set_duty_percent(100).unwrap();
}
