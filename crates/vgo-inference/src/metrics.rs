use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::InferenceStageMetrics;

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
pub(crate) struct AtomicMetrics {
    pub(crate) requests: AtomicU64,
    pub(crate) batches: AtomicU64,
    pub(crate) positions: AtomicU64,
    pub(crate) maximum_batch: AtomicUsize,
    pub(crate) failures: AtomicU64,
    pub(crate) encoding_nanoseconds: AtomicU64,
    pub(crate) queue_nanoseconds: AtomicU64,
    pub(crate) channel_nanoseconds: AtomicU64,
    pub(crate) broker_queue_nanoseconds: AtomicU64,
    pub(crate) idle_request_wait_nanoseconds: AtomicU64,
    pub(crate) overlap_request_wait_nanoseconds: AtomicU64,
    pub(crate) batch_collection_nanoseconds: AtomicU64,
    pub(crate) batch_submission_nanoseconds: AtomicU64,
    pub(crate) completion_wait_nanoseconds: AtomicU64,
    pub(crate) full_batches: AtomicU64,
    pub(crate) deadline_batches: AtomicU64,
    pub(crate) drain_batches: AtomicU64,
    pub(crate) inference_nanoseconds: AtomicU64,
    pub(crate) input_packing_nanoseconds: AtomicU64,
    pub(crate) session_run_nanoseconds: AtomicU64,
    pub(crate) output_materialization_nanoseconds: AtomicU64,
}

impl AtomicMetrics {
    pub(crate) fn snapshot(&self) -> BrokerMetrics {
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

    pub(crate) fn record_stages(&self, stages: InferenceStageMetrics) {
        self.input_packing_nanoseconds
            .fetch_add(stages.input_packing_nanoseconds, Ordering::Relaxed);
        self.session_run_nanoseconds
            .fetch_add(stages.session_run_nanoseconds, Ordering::Relaxed);
        self.output_materialization_nanoseconds
            .fetch_add(stages.output_materialization_nanoseconds, Ordering::Relaxed);
    }

    /// What the whole broker and each lane both count for a dispatched batch.
    pub(crate) fn record_batch(&self, record: &BatchRecord) {
        let positions = record.positions as u64;
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.positions.fetch_add(positions, Ordering::Relaxed);
        self.maximum_batch
            .fetch_max(record.positions, Ordering::Relaxed);
        self.queue_nanoseconds
            .fetch_add(record.queue_nanoseconds, Ordering::Relaxed);
        self.channel_nanoseconds
            .fetch_add(record.channel_nanoseconds, Ordering::Relaxed);
        self.broker_queue_nanoseconds
            .fetch_add(record.broker_queue_nanoseconds, Ordering::Relaxed);
        self.batch_collection_nanoseconds
            .fetch_add(record.collection_nanoseconds, Ordering::Relaxed);
        self.batch_submission_nanoseconds
            .fetch_add(record.submission_nanoseconds, Ordering::Relaxed);
        let reason = if record.full {
            &self.full_batches
        } else if record.deadline_expired {
            &self.deadline_batches
        } else if record.drained {
            &self.drain_batches
        } else {
            return;
        };
        reason.fetch_add(1, Ordering::Relaxed);
    }

    /// What only a lane counts here. The broker counts these per caller instead.
    pub(crate) fn record_lane_dispatch(&self, record: &BatchRecord) {
        self.requests
            .fetch_add(record.positions as u64, Ordering::Relaxed);
        self.encoding_nanoseconds
            .fetch_add(record.encoding_nanoseconds, Ordering::Relaxed);
    }
}

/// The measurements for one dispatched batch.
#[derive(Default)]
pub(crate) struct BatchRecord {
    pub(crate) positions: usize,
    pub(crate) queue_nanoseconds: u64,
    pub(crate) channel_nanoseconds: u64,
    pub(crate) broker_queue_nanoseconds: u64,
    pub(crate) encoding_nanoseconds: u64,
    pub(crate) collection_nanoseconds: u64,
    pub(crate) submission_nanoseconds: u64,
    pub(crate) full: bool,
    pub(crate) deadline_expired: bool,
    pub(crate) drained: bool,
}
