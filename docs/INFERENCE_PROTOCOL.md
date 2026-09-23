# Inference Boundaries

## Ownership

`vgo-search` defines an evaluator interface. A nonterminal node requests one
player-relative value and a policy function; terminal nodes always use exact
Rust scoring and never call a model. `vgo-inference` implements that interface
with a shared broker over native ONNX Runtime. The built-in `NaiveEvaluator`
needs no model and seeds a loop that starts from nothing.

Rust owns rasterization, request IDs, batching, backpressure, response routing,
and model execution. Python owns training and ONNX export.

## Native ONNX contract

`OnnxBatchService` loads a model once in the Rust process and implements
`BatchService`. Before serving it validates the
raster schema, channel count, spatial dimensions, dense policy size, maximum
batch, tensor names, and source-checkpoint digest embedded by the exporter.
CUDA and TensorRT are explicit providers; unavailable requested acceleration is
an error rather than a CPU fallback.

A model exported with `--packed-input` takes three tensors (`bits`, `dense`,
`scalars`) instead of one dense `states` raster; the layout is read from the
model, so a binary and a model that disagree fail at load. Each session packs
into one maximum-batch host allocation reused for its lifetime. TensorRT engine
and timing caches are separated by model digest, precision, raster shape, and
maximum batch. Pinned buffers and device I/O binding remain later throughput
work.

## Broker

Each actor rasterizes its leaf round before submitting one ordered group through
a bounded synchronous channel. The broker takes the first group, waits up to a
short deadline for peers, and flattens groups across backend batch boundaries up
to the declared contract. A group larger than the backend ceiling may span
several calls; its caller still receives exactly one ordered completion after
every output is reassembled. Encoding therefore scales with actors rather than
being partitioned across per-slot broker queues.

The implementation exposes three independent contracts:

- `InferenceInput` and `InferenceOutput` are the encoded request boundary;
- `BatchService` synchronously evaluates an already-encoded batch;
- `BatchExecutor` separates batch submission from completion, declares its slot
  capacity, and permits out-of-order completion by sequence number.

Each `BatchService` remains synchronous, but generation can run multiple
session-owned execution slots with `--inference-slots` (default `2`). One
broker builds batches from the shared actor queue, then assigns each complete
batch to a free slot; requests are no longer partitioned before batching. Each
slot retains its own session, execution context, and reusable input storage, so
tune the slot count against device memory and end-to-end throughput. The
`BatchExecutor` contract also permits a future backend to use multiple
pinned-memory/stream slots inside one session without changing actors,
encoding, batching, or response routing.

Metrics count submitted request positions (the compatibility `requests`
counter), executed batches and positions, maximum batch occupancy, and
failures. Timing is split at the ownership boundaries:

- summed parallel actor encoding;
- position-weighted channel time before the broker receives a request;
- position-weighted broker time from receipt through slot dispatch;
- per-batch collection and executor submission;
- broker waits for an entirely idle executor, a partially idle executor, and
  executor completion;
- input packing, ONNX `Session::run`, and output materialization inside each
  instrumented native slot.

The original total queue and inference counters remain for compatibility.
Batch counters distinguish full, deadline-expired, and shutdown-drain
dispatches. `Session::run` still contains runtime overhead and synchronous
host/device copies; explicit I/O binding is required to separate those.
Summed parallel and position-weighted durations may exceed wall time.

## Open work

- Pinned reusable slabs and I/O binding, to remove the remaining pageable host
  packing and allow explicit asynchronous transfers per slot.
- `--inference-slots` above 2 livelocks the inference threads (100% CPU, GPU at
  idle power, no results). Understand why before raising it.
