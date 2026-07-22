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

Callers submit through a bounded synchronous channel. The broker takes the first
request, waits up to a short deadline for peers, caps the batch, rasterizes the
positions, performs one subprocess exchange, and returns each output to its
waiting search thread.

Metrics count requests, batches, positions, maximum batch occupancy, failures,
total queue nanoseconds, and total subprocess-exchange nanoseconds. Queue and
inference durations are sums, so divide by positions or batches as appropriate.

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

## Remaining production work

- Add an explicit startup handshake carrying schema and model versions.
- Support checkpoint replacement only at a documented actor synchronization
  point.
- Benchmark accelerator services and larger models across batch windows and
  actor counts.
- Move immutable trajectory writing from the demo generator into self-play.
- Attach per-game model versions and protocol failures to replay metadata.
