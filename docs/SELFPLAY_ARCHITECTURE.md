# Self-Play Architecture

## Current milestone

The first two benchmark layers are implemented:

- `vgo-core` owns exact Voronoi geometry, legal placement, global capture and
  self-capture, pass/pass termination, and scoring;
- `vgo-search` owns deterministic prefix candidates, a naive spread evaluator,
  progressive-widening MCTS, and an explicit fallible evaluator boundary;
- `vgo-inference` owns a bounded batching broker and framed Python subprocess
  transport;
- `vgo-canary` runs parallel color-swapped matches and emits machine-readable
  search and gameplay metrics.

The original canary deliberately uses no Python process. Its neutral
nonterminal value and exact terminal value continue to isolate simulator and
search behavior. A separate model smoke path now runs the trained raster CNN
through the broker described in [`INFERENCE_PROTOCOL.md`](INFERENCE_PROTOCOL.md).
Acceptance results are retained under [`../benchmarks/results/`](../benchmarks/results/).

## Ownership boundary

Rust owns the complete data-generation loop:

```text
game actors -> MCTS -> candidate widening -> exact transitions
                   \-> inference broker -> Python model service
game actors -> completed trajectories -> replay shard writer
```

Python owns model execution and optimization:

```text
inference service <- versioned batches from Rust
trainer <- replay shards written by Rust
trainer -> atomic model checkpoints -> inference service
```

Python does not import the simulator. Rust does not implement neural-network
layers. Their contracts are versioned data formats and a batch inference
protocol.

## Rust self-play process

One process may host many actor threads. Each actor owns a game and search tree.
When a leaf needs evaluation, it submits a request to a shared inference broker
and yields rather than calling Python directly.

The broker:

- combines requests until a maximum batch size or latency deadline;
- applies backpressure when too many batches are in flight;
- preserves request IDs so results return to the correct trees;
- records model version, queue delay, encoding time, and inference time;
- may later coalesce duplicate position evaluations.

Search, simulation, state encoding, symmetry transforms, trajectory assembly,
and replay serialization remain native and parallelizable.

## Python inference service

The service accepts contiguous batched features and candidate metadata. It
returns value predictions and policy outputs without knowing game rules.

The first protocol should support:

```text
request:  protocol version, model version, request IDs,
          state tensor, candidate offsets and features
response: request IDs, value outputs, proposal outputs, candidate logits
```

Tensor shapes, dtypes, channel meanings, coordinate conventions, and player
perspective must be part of a checked model-interface schema. We should begin
with a simple local transport and preserve the ability to replace it with
shared memory after profiling.

Model updates occur between games or at another explicit synchronization point.
A trajectory records the model version that generated every search target.

## Replay boundary

Rust writes immutable, chunked replay shards. Python reads them without a Rust
extension. Each record contains enough data to reproduce training targets and
audit sampling behavior:

```text
position or canonical state features
candidate coordinates, sources, and proposal probabilities
root visit counts and selected action
player perspective and terminal score
search budget, RNG seed, and model version
```

The precise storage format should be selected after representative records are
available. Selection criteria are streaming writes, Python and Rust support,
schema evolution, partial recovery, and efficient variable-length candidates.

## Failure semantics

- A model-service disconnect stops or drains actors; it never substitutes
  arbitrary values silently.
- Timeouts and dropped evaluations are counted and attached to run metadata.
- A protocol or model-schema mismatch fails before self-play begins.
- A replay shard is published atomically only after its checksum and footer are
  complete.

## Benchmark layers

Measure the system at four boundaries:

1. Engine analysis and transitions in one Rust thread.
2. Parallel Rust search with a deterministic fake evaluator.
3. Rust-to-Python batching with a trivial tensor model.
4. End-to-end self-play with the real model and replay writer.

This separates simulator speed from batching, transport, GPU inference, and
storage bottlenecks.
