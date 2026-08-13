# Inference Boundaries

## Ownership

`vgo-search` defines an evaluator interface. A nonterminal node requests one
player-relative value and a policy function; terminal nodes always use exact
Rust scoring and never call a model. `vgo-inference` implements that interface
with a shared broker over either native ONNX Runtime or a long-lived Python
subprocess. The built-in `NaiveEvaluator` preserves the model-free canary.

Rust owns rasterization, request IDs, batching, backpressure, response routing,
and production model execution. Python owns training and ONNX export. Its
subprocess service is retained for output parity and transport benchmarks.

## Native ONNX contract

`OnnxBatchService` loads a model once in the Rust process and exposes the same
`BatchService` contract as the Python backend. Before serving it validates the
raster schema, channel count, spatial dimensions, dense policy size, maximum
batch, tensor names, and source-checkpoint digest embedded by the exporter.
CUDA and TensorRT are explicit providers; unavailable requested acceleration is
an error rather than a CPU fallback.

The current implementation packs into one maximum-batch host allocation reused
for the lifetime of the ONNX session and collects owned outputs. TensorRT engine
and timing caches are separated by model digest, precision, raster shape, and
maximum batch. Pinned buffers and device I/O binding remain later throughput
work.

## Transport

Version 1 uses binary frames over child-process stdin/stdout. All integers and
floats are little-endian. Python stderr remains attached to the parent so logs
cannot corrupt response frames.

Request header:

```text
8 bytes  VGOIFR01
u32      protocol version
u32      batch size
u32      channels
u32      height
u32      width
```

Each request item contains:

```text
u64      request ID
f32[]    contiguous [channels, height, width] semantic raster
```

Response header:

```text
8 bytes  VGOOFR01
u32      protocol version
u32      batch size
u32      policy size
```

Each response item contains:

```text
u64      request ID
f32      current-player value in [-1, 1]
f32[]    dense placement logits followed by pass logit
```

Responses are routed by ID and may arrive in any item order. Unknown, duplicate,
or missing IDs fail the batch. Shape mismatches, non-finite outputs, invalid
values, malformed frames, subprocess termination, and closed queues propagate
as evaluator errors; search never substitutes neutral predictions silently.

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
counter), executed batches and positions, maximum batch occupancy, failures,
summed parallel encoding nanoseconds, total queue nanoseconds, and backend
inference nanoseconds. Summed durations may exceed wall time.

## Verified smoke path

The initial 48x48 CPU test used one Python service, a queue capacity of 64, a
five-millisecond batch window, and a maximum batch size of 16:

- direct Python and framed Rust results matched for value and sampled logits;
- 16 synchronized calls formed a single batch;
- 16 concurrent MCTS actors completed 16 games and 187 plies;
- 1,493 evaluations were served in 194 batches, averaging 7.70 positions;
- maximum occupancy reached 16 and no request failed;
- actor wall time was 2.94 seconds.

Raw metrics are retained in
[`../benchmarks/results/2026-07-21-inference-smoke.json`](../benchmarks/results/2026-07-21-inference-smoke.json).

## CPU scaling

The parameterized release benchmark runs with:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-model-smoke -- `
  --actors 64 --games 256 --simulations 8 `
  --maximum-batch 16 --delay-ms 1 --torch-threads 16
```

On a 16-core/32-thread Ryzen 9 9950X using PyTorch 2.13 CPU inference,
32 actors sustained 2,279 evaluations per second and 64 actors sustained
2,319. Increasing to 128 actors reduced throughput to 2,196. The efficient
default is therefore 32 actors; 64 actors are useful when maximum throughput is
more important than actor count and scheduling overhead.

Three additional 64-actor runs sustained 2,474 to 2,485 evaluations per second,
with a mean of 2,481 and no failures. The sweep values are the conservative
comparison between configurations; the repeated range is the best estimate of
steady-state capacity on this machine.

For this 59,555-parameter CNN and 10x48x48 input, batches larger than 16 are
counterproductive on CPU. At 64 actors, a batch cap of 16 sustained 2,317
evaluations per second, compared with 2,127 at 32 and 1,655 at 64. A one- or
two-millisecond collection window fills batches without meaningful idle time;
zero delay produced average batches of only 2.2 and 1,302 evaluations per
second.

These are end-to-end actor results: each evaluation includes Rust state
analysis and rasterization, framed process transport, Python tensor assembly,
model inference, and MCTS consumption. At eight simulations per move the tuned
run completed about 29 short 3x3-board games per second. Games per second will
fall approximately in proportion to the simulation budget once inference is
the bottleneck; evaluations per second is the portable capacity measure.

All sweep results and exact parameters are retained in
[`../benchmarks/results/2026-07-21-selfplay-scaling.json`](../benchmarks/results/2026-07-21-selfplay-scaling.json).

## GPU stages

The CUDA environment uses PyTorch 2.13, CUDA 13.0, and `torch.compile` on an RTX
5070 Ti. Windows installs `triton-windows` 3.7 to match PyTorch's compiler
version. CUDA is explicit: a missing accelerator fails service startup instead
of silently selecting CPU.

The release stage benchmarks isolate identical 10x48x48 inputs:

| Stage | Batch | Throughput |
|---|---:|---:|
| Rust raster, one thread | 1 | 16,510 positions/s |
| Rust raster, 16 threads | 1 | 192,357 positions/s |
| Contiguous request framing | 16 | 50,197 positions/s |
| Compiled GPU model only | 16 | 52,135 positions/s |
| Warm framed subprocess | 16 | 10,385 positions/s |
| Warm framed subprocess | 32 | 12,330 positions/s |
| Warm framed subprocess | 64 | 12,754 positions/s |

Compiled model throughput at batch 16 increases from 56,461 positions/s with
one execution slot to 71,816 with two and 78,248 with four. Two slots capture
most of the concurrency benefit. The framed service does not yet exploit those
slots: it reads, copies, executes, copies back, and writes each batch in
lockstep.

After moving rasterization to actors and constructing one contiguous frame per
batch, 64 actors with batch 32 sustained **7,965 evaluations/s** over 512 games.
Batch 16 sustained 7,695; 128 actors or batch 64 regressed. On all 96 retained
semantic examples, compiled CUDA preserved the CPU and eager-CUDA top policy
action. Raw stage data is retained in
[`../benchmarks/results/2026-07-21-gpu-inference-stages.json`](../benchmarks/results/2026-07-21-gpu-inference-stages.json).

### Resolution scaling

The active small-game canary now separates game and representation scale: stone
radius remains `1/6`, while the raster is 128x128. One 10-channel `f32` input is
655,360 bytes, and batch 8 is 5,242,880 bytes before framing. Relative to 48x48,
the tensor is 7.11 times larger.

At 128x128, 16 Rust raster threads sustained 27,662 to 28,833 positions/s and
request framing sustained 7,473 to 7,605 positions/s at 4.56 to 4.64 GiB/s.
The compiled GPU model peaked at 8,714 positions/s with batch 8, while the warm
subprocess service reached 1,880 positions/s. Two simultaneous batch-16 model
slots regressed from 7,887 to 7,186 positions/s, indicating that model compute
is already saturated at this resolution. End-to-end self-play with 64 actors
and batch 8 sustained 1,452 evaluations/s across 61,440 requests with no
failures. Raw results are retained in
[`../benchmarks/results/2026-07-21-raster-resolution-scaling.json`](../benchmarks/results/2026-07-21-raster-resolution-scaling.json).

The subprocess path performs several avoidable copies around the one necessary
host-to-device transfer: it materializes a byte frame, crosses an OS pipe,
copies into NumPy, and uses pageable host memory. The native ONNX/TensorRT path
removes Python and pipe overhead. At 128x128 and batch 16 it sustained 5,732
positions/s in isolation and 4,109 evaluations/s through 64 self-play actors,
2.83 times the prior subprocess result. Raw data is retained in
[`../benchmarks/results/2026-07-21-onnx-tensorrt.json`](../benchmarks/results/2026-07-21-onnx-tensorrt.json).

The native path still gathers actor rasters into a contiguous host tensor and
returns owned output tensors. The pageable packing allocation is now retained
across calls, but reusable pinned input/output slabs and I/O binding would leave
one asynchronous host-to-device and device-to-host transfer per batch.
`vgo-raster::rasterize_into` provides the caller-owned destination needed for
direct writes into assigned slab rows.

## Remaining production work

- Implement pinned reusable slabs and I/O binding to remove the remaining
  pageable host packing and enable explicit asynchronous transfers per slot.
- Keep actor workers and the ONNX session alive across same-model shard
  boundaries when measurements show the residual process/session startup is
  material; model-digest-scoped TensorRT warmup already removes engine build
  from the next shard's critical path.
- Benchmark the learned production model across batch windows and actor counts.
- Retain the Python protocol as a parity oracle; optimize it only if a measured
  workflow still depends on it.
