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

## Remaining production work

- Add an explicit startup handshake carrying schema and model versions.
- Support checkpoint replacement only at a documented actor synchronization
  point.
- Benchmark CPU and accelerator services across batch windows and actor counts.
- Move immutable trajectory writing from the demo generator into self-play.
- Attach per-game model versions and protocol failures to replay metadata.
