# Batched Inference Protocol

## Ownership

`vgo-search` defines an evaluator interface. A nonterminal node requests one
player-relative value and a policy function; terminal nodes always use exact
Rust scoring and never call a model. The built-in `NaiveEvaluator` preserves the
original canary path, while `vgo-inference` implements the same interface with a
long-lived Python subprocess.

Rust owns rasterization, request IDs, batching, backpressure, and response
routing. Python owns checkpoint loading and batched neural-network execution.

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

Each actor rasterizes its position before submitting through a bounded
synchronous channel. The broker takes the first encoded request, waits up to a
short deadline for peers, caps the batch, performs one subprocess exchange, and
returns each output to its waiting search thread. Encoding therefore scales
with actors rather than occupying the single broker.

The implementation exposes three independent contracts:

- `InferenceInput` and `InferenceOutput` are the encoded request boundary;
- `BatchService` synchronously evaluates an already-encoded batch;
- `BatchExecutor` separates batch submission from completion, declares its slot
  capacity, and permits out-of-order completion by sequence number.

The current subprocess is a capacity-one `BatchService`. A threaded executor
adapter demonstrates the asynchronous contract; a future CUDA implementation
can provide multiple pinned-memory/stream slots without changing actors,
encoding, batching, or response routing.

Metrics count requests, batches, positions, maximum batch occupancy, failures,
summed parallel encoding nanoseconds, total queue nanoseconds, and total
subprocess-exchange nanoseconds. Summed durations may exceed wall time.

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
cargo run --release -p vgo-inference --bin vgo-model-smoke -- `
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

The current subprocess path performs several avoidable copies around the one
necessary host-to-device transfer: it materializes a byte frame, crosses an OS
pipe, copies each payload into a NumPy batch, copies pageable host memory to the
GPU, and mirrors the process for outputs. An in-process ONNX executor should use
reusable pinned input/output slabs and I/O binding. Initially, individual actor
rasters still require one host gather into the contiguous input slab. A later
batched raster encoder can write directly into assigned slab rows, leaving one
asynchronous host-to-device transfer and one device-to-host transfer per batch.
`vgo-raster::rasterize_into` provides the caller-owned destination required for
that direct-write path.

## Remaining production work

- Add an explicit startup handshake carrying schema and model versions.
- Support checkpoint replacement only at a documented actor synchronization
  point.
- Implement a two-slot pinned-memory CUDA executor and overlap service transfer,
  execution, and response work.
- Evaluate shared memory or another bulk local transport after pipelining.
- Benchmark larger models across batch windows and actor counts.
- Move immutable trajectory writing from the demo generator into self-play.
- Attach per-game model versions and protocol failures to replay metadata.
