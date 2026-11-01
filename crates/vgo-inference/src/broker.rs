//! One FIFO queue of requests, batched for a pool of execution slots.
//!
//! Each backend batch is built before a slot is chosen, so adding slots adds
//! execution concurrency without changing how batches form. A caller's group may
//! span several batches. It gets exactly one reply, with outputs in input order.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::Ordering,
        mpsc::{Receiver, RecvTimeoutError, SyncSender},
    },
    time::Instant,
};

use vgo_search::EvaluationError;

use crate::metrics::{AtomicMetrics, BatchRecord};
use crate::{
    BatchContract, BatchService, BrokerConfig, CompletedBatch, InferenceBatch, InferenceInput,
    InferenceOutput, ThreadedBatchExecutorPool,
};

pub(crate) struct Request {
    pub(crate) inputs: Vec<InferenceInput>,
    pub(crate) encoding_nanoseconds: u64,
    pub(crate) queued_at: Instant,
    pub(crate) response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

/// The part of a request not yet placed in a batch.
struct Queued {
    key: u64,
    inputs: std::vec::IntoIter<InferenceInput>,
    next_offset: usize,
    encoding_nanoseconds: u64,
    queued_at: Instant,
    brokered_at: Instant,
}

/// Outputs collected so far for one caller.
struct PendingResponse {
    outputs: Vec<Option<InferenceOutput>>,
    remaining: usize,
    response: SyncSender<Result<Vec<InferenceOutput>, EvaluationError>>,
}

/// A run of one request's positions inside a batch.
struct BatchPart {
    key: u64,
    offset: usize,
    count: usize,
    encoding_nanoseconds: u64,
    queued_at: Instant,
    brokered_at: Instant,
}

struct CollectedBatch {
    inputs: Vec<InferenceInput>,
    parts: Vec<BatchPart>,
    deadline_expired: bool,
    collection_nanoseconds: u64,
}

struct InFlight {
    expected_ids: Vec<u64>,
    parts: Vec<BatchPart>,
    position_count: usize,
    slot: usize,
}

pub(crate) struct Broker<S: BatchService + 'static> {
    config: BrokerConfig,
    contract: BatchContract,
    executor: ThreadedBatchExecutorPool<S>,
    receiver: Receiver<Request>,
    metrics: Arc<AtomicMetrics>,
    lane_metrics: Vec<Arc<AtomicMetrics>>,
    queued: VecDeque<Queued>,
    /// Positions in `queued` that no batch has taken yet.
    queued_positions: usize,
    responses: HashMap<u64, PendingResponse>,
    in_flight: HashMap<u64, InFlight>,
    next_request: u64,
    next_batch: u64,
    disconnected: bool,
}

impl<S: BatchService + 'static> Broker<S> {
    pub(crate) fn new(
        config: BrokerConfig,
        contract: BatchContract,
        executor: ThreadedBatchExecutorPool<S>,
        receiver: Receiver<Request>,
        metrics: Arc<AtomicMetrics>,
        lane_metrics: Vec<Arc<AtomicMetrics>>,
    ) -> Self {
        Self {
            config,
            contract,
            executor,
            receiver,
            metrics,
            lane_metrics,
            queued: VecDeque::new(),
            queued_positions: 0,
            responses: HashMap::new(),
            in_flight: HashMap::new(),
            next_request: 0,
            next_batch: 0,
            disconnected: false,
        }
    }

    /// Serve until every caller has gone and all work has drained. Any failure
    /// stops the broker and is reported to every waiting caller.
    pub(crate) fn run(mut self) {
        if let Err(error) = self.serve() {
            for (_, pending) in self.responses.drain() {
                let _ = pending.response.send(Err(error.clone()));
            }
            for request in self.receiver.try_iter() {
                let _ = request.response.send(Err(error.clone()));
            }
        }
    }

    fn serve(&mut self) -> Result<(), EvaluationError> {
        loop {
            self.fill_free_slots()?;
            if self.executor.in_flight() > 0 {
                self.complete_one()?;
            } else if self.disconnected && self.queued.is_empty() {
                return Ok(());
            }
        }
    }

    fn fill_free_slots(&mut self) -> Result<(), EvaluationError> {
        while self.executor.in_flight() < self.executor.capacity() {
            // A deck is for a batch that is ready. With every slot running, a short
            // one would take positions a full batch was about to use, and nothing is
            // waiting for it to start.
            if self.executor.in_flight() >= self.executor.slots() && !self.full_batch_ready() {
                break;
            }
            if self.queued.is_empty() && !self.disconnected && !self.wait_for_request() {
                break;
            }
            if self.queued.is_empty() {
                break;
            }
            let batch = self.collect_batch();
            self.dispatch(batch)?;
        }
        Ok(())
    }

    /// Whether a full batch is queued, taking whatever has already arrived.
    fn full_batch_ready(&mut self) -> bool {
        while self.queued_positions < self.contract.maximum_batch {
            match self.receiver.try_recv() {
                Ok(request) => self.enqueue(request),
                Err(_) => return false,
            }
        }
        true
    }

    /// Block until a request arrives. With work in flight, only wait the batching
    /// delay, and return false so a completion can be handled instead.
    fn wait_for_request(&mut self) -> bool {
        let idle = self.executor.in_flight() == 0;
        let started = Instant::now();
        let received = if idle {
            self.receiver
                .recv()
                .map_err(|_| RecvTimeoutError::Disconnected)
        } else {
            self.receiver.recv_timeout(self.config.maximum_delay)
        };
        let waited = started.elapsed().as_nanos() as u64;
        let counter = if idle {
            &self.metrics.idle_request_wait_nanoseconds
        } else {
            &self.metrics.overlap_request_wait_nanoseconds
        };
        counter.fetch_add(waited, Ordering::Relaxed);
        self.accept(received)
    }

    /// Returns false when nothing arrived before a timeout.
    fn accept(&mut self, received: Result<Request, RecvTimeoutError>) -> bool {
        match received {
            Ok(request) => {
                self.enqueue(request);
                true
            }
            Err(RecvTimeoutError::Timeout) => false,
            Err(RecvTimeoutError::Disconnected) => {
                self.disconnected = true;
                true
            }
        }
    }

    fn enqueue(&mut self, request: Request) {
        let key = self.next_request;
        self.next_request = key.wrapping_add(1);
        let count = request.inputs.len();
        debug_assert!(count > 0, "evaluators do not submit empty requests");
        self.queued_positions += count;
        self.responses.insert(
            key,
            PendingResponse {
                outputs: (0..count).map(|_| None).collect(),
                remaining: count,
                response: request.response,
            },
        );
        self.queued.push_back(Queued {
            key,
            inputs: request.inputs.into_iter(),
            next_offset: 0,
            encoding_nanoseconds: request.encoding_nanoseconds,
            queued_at: request.queued_at,
            brokered_at: Instant::now(),
        });
    }

    /// Pack queued work into one batch, waiting up to the batching delay for more
    /// if the batch is not yet full.
    fn collect_batch(&mut self) -> CollectedBatch {
        let started = Instant::now();
        let deadline = started + self.config.maximum_delay;
        let mut inputs = Vec::with_capacity(self.contract.maximum_batch);
        let mut parts = Vec::new();
        let mut deadline_expired = false;
        loop {
            self.take_queued(&mut inputs, &mut parts);
            let full = inputs.len() == self.contract.maximum_batch;
            if full || !self.queued.is_empty() || self.disconnected {
                break;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                deadline_expired = true;
                break;
            };
            let received = self.receiver.recv_timeout(remaining);
            if !self.accept(received) {
                deadline_expired = true;
                break;
            }
        }
        CollectedBatch {
            inputs,
            parts,
            deadline_expired,
            collection_nanoseconds: started.elapsed().as_nanos() as u64,
        }
    }

    fn take_queued(&mut self, inputs: &mut Vec<InferenceInput>, parts: &mut Vec<BatchPart>) {
        while inputs.len() < self.contract.maximum_batch {
            let Some(current) = self.queued.front_mut() else {
                break;
            };
            let remaining = current.inputs.len();
            let count = remaining.min(self.contract.maximum_batch - inputs.len());
            let offset = current.next_offset;
            inputs.extend(current.inputs.by_ref().take(count));
            self.queued_positions -= count;
            current.next_offset += count;
            // A split request charges its encoding time pro rata.
            let encoded = if count == remaining {
                current.encoding_nanoseconds
            } else {
                ((u128::from(current.encoding_nanoseconds) * count as u128) / remaining as u128)
                    as u64
            };
            current.encoding_nanoseconds -= encoded;
            parts.push(BatchPart {
                key: current.key,
                offset,
                count,
                encoding_nanoseconds: encoded,
                queued_at: current.queued_at,
                brokered_at: current.brokered_at,
            });
            if current.inputs.len() == 0 {
                self.queued.pop_front();
            }
        }
    }

    fn dispatch(&mut self, collected: CollectedBatch) -> Result<(), EvaluationError> {
        let CollectedBatch {
            inputs,
            parts,
            deadline_expired,
            collection_nanoseconds,
        } = collected;
        let position_count = inputs.len();
        let expected_ids = inputs.iter().map(InferenceInput::id).collect::<Vec<_>>();
        let sequence = self.next_batch;
        let batch = InferenceBatch::new(sequence, inputs)?;
        let submission_started = Instant::now();
        let slot = self.executor.submit(batch)?;
        let submission_nanoseconds = submission_started.elapsed().as_nanos() as u64;

        let dispatched = Instant::now();
        let mut record = BatchRecord {
            positions: position_count,
            collection_nanoseconds,
            submission_nanoseconds,
            full: position_count == self.contract.maximum_batch,
            deadline_expired,
            drained: self.disconnected,
            ..BatchRecord::default()
        };
        for part in &parts {
            let count = part.count as u64;
            let weighted = |from: Instant, to: Instant| {
                (to.duration_since(from).as_nanos() as u64).saturating_mul(count)
            };
            record.queue_nanoseconds = record
                .queue_nanoseconds
                .saturating_add(weighted(part.queued_at, dispatched));
            record.channel_nanoseconds = record
                .channel_nanoseconds
                .saturating_add(weighted(part.queued_at, part.brokered_at));
            record.broker_queue_nanoseconds = record
                .broker_queue_nanoseconds
                .saturating_add(weighted(part.brokered_at, dispatched));
            record.encoding_nanoseconds = record
                .encoding_nanoseconds
                .saturating_add(part.encoding_nanoseconds);
        }
        self.metrics.record_batch(&record);
        let lane = &self.lane_metrics[slot];
        lane.record_batch(&record);
        lane.record_lane_dispatch(&record);

        self.in_flight.insert(
            sequence,
            InFlight {
                expected_ids,
                parts,
                position_count,
                slot,
            },
        );
        self.next_batch = sequence.wrapping_add(1);
        Ok(())
    }

    fn complete_one(&mut self) -> Result<(), EvaluationError> {
        let started = Instant::now();
        let completion = self.executor.receive().inspect_err(|_| {
            let batches = self.in_flight.values();
            for batch in batches {
                self.record_failure(batch.slot, batch.position_count);
            }
        })?;
        self.metrics
            .completion_wait_nanoseconds
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let sequence = completion.sequence();
        let slot = completion.slot();
        let Some(batch) = self.in_flight.remove(&sequence) else {
            return Err(EvaluationError::new(format!(
                "inference executor returned unknown batch sequence {sequence}"
            )));
        };
        debug_assert_eq!(slot, batch.slot);
        let elapsed = completion.elapsed().as_nanos() as u64;
        let stages = completion.stages();
        for metrics in [&self.metrics, &self.lane_metrics[slot]] {
            metrics
                .inference_nanoseconds
                .fetch_add(elapsed, Ordering::Relaxed);
            metrics.record_stages(stages);
        }
        match validated_outputs(completion, &batch.expected_ids) {
            Ok(outputs) => {
                self.deliver(batch.parts, outputs);
                Ok(())
            }
            Err(error) => {
                self.record_failure(slot, batch.position_count);
                Err(error)
            }
        }
    }

    fn record_failure(&self, slot: usize, positions: usize) {
        for metrics in [&self.metrics, &self.lane_metrics[slot]] {
            metrics
                .failures
                .fetch_add(positions as u64, Ordering::Relaxed);
        }
    }

    /// Hand each part's outputs to its caller, replying once a group is whole.
    fn deliver(&mut self, parts: Vec<BatchPart>, outputs: Vec<InferenceOutput>) {
        let mut outputs = outputs.into_iter();
        for part in parts {
            let pending = self
                .responses
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
                let pending = self
                    .responses
                    .remove(&part.key)
                    .expect("completed response remains registered");
                let ordered = pending
                    .outputs
                    .into_iter()
                    .map(|output| output.expect("all request positions completed"))
                    .collect();
                let _ = pending.response.send(Ok(ordered));
            }
        }
        debug_assert!(outputs.next().is_none());
    }
}

/// A batch's outputs, checked against the request IDs it was sent with.
fn validated_outputs(
    completion: CompletedBatch,
    expected_ids: &[u64],
) -> Result<Vec<InferenceOutput>, EvaluationError> {
    let outputs = completion.into_outputs();
    if outputs.len() != expected_ids.len() {
        return Err(EvaluationError::new(format!(
            "inference backend returned {} outputs for {} inputs",
            outputs.len(),
            expected_ids.len()
        )));
    }
    for (index, (output, expected_id)) in outputs.iter().zip(expected_ids).enumerate() {
        if output.id() != *expected_id {
            return Err(EvaluationError::new(format!(
                "inference output {index} has request id {}, expected {expected_id}",
                output.id()
            )));
        }
    }
    Ok(outputs)
}
