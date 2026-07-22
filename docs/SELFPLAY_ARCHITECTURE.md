# Self-Play Architecture

## Current system

Rust owns the complete gameplay and inference-serving loop:

- `vgo-core` owns geometry, legal placement, global opponent capture,
  self-capture, pass/pass termination, and scoring;
- `vgo-raster` owns the semantic tensor and human-readable diagnostics;
- `vgo-search` owns deterministic candidate generation, evaluators, and
  progressive-widening MCTS;
- `vgo-inference` owns bounded batching and interchangeable native ONNX or
  framed Python batch services;
- `vgo-selfplay` owns the one canonical complete-game playout, repetition
  avoidance, arenas, model smoke tests, and trajectory generation.

The naive canary uses no model runtime and isolates engine/search behavior. The
model smoke path uses the same playout with either ONNX Runtime/TensorRT or the
Python parity service.

## Ownership boundary

```text
actor threads -> shared playout -> MCTS -> inference broker -> BatchService
                       |                              |          |-- ONNX/TensorRT
                       |                              |          `-- Python debug
                       `-> completed trajectory -> replay writer

Python trainer -> atomic .pt checkpoint -> ONNX export -> Rust worker startup
```

Python never imports the simulator. Rust never implements neural-network
layers. Their durable boundaries are replay files and a self-describing ONNX
artifact. The framed subprocess protocol is retained for parity checks and
stage benchmarks rather than required for deployment.

## Playout contract

`vgo-selfplay::play_game` is the sole owner of whole-game progression. Callers
provide a search function and an optional per-ply observer. The playout owns:

- exact position fingerprints and repetition avoidance;
- preferred-action fallback to pass;
- transition application and event accounting;
- pass/pass termination and the maximum-ply bound;
- accumulated search and gameplay statistics.

The canary, model smoke test, and demo dataset generator therefore cannot drift
on game semantics. MCTS still owns tree-local simulation; that is a different
scope from an externally visible game trajectory.

## Inference contract

Actors rasterize positions before submitting them to a shared bounded broker.
Every `BatchService` declares the raster shape and maximum batch it accepts, so
the broker cannot be configured inconsistently with its backend. The broker
collects requests until the batch cap or latency deadline, invokes the service,
and routes validated outputs by request ID.

The current broker has one synchronous service slot. Actor-side search and
rasterization are parallel, but a submitted evaluation blocks its actor until
that shared batch completes. Multiple in-flight GPU slots, pinned slabs, and I/O
binding remain performance work; their executor interface already separates
submission from sequence-keyed completion.

The native ONNX loader validates model metadata, raster schema, tensor names and
shapes, maximum batch, source digest, and finite outputs before or during use.
TensorRT engine and timing caches are namespaced by model and execution
configuration. See [`INFERENCE_PROTOCOL.md`](INFERENCE_PROTOCOL.md).

## Training and replay

Python owns optimization and export. Rust will write immutable, chunked replay
shards containing enough information to train and audit each decision:

```text
semantic raster and sampled policy mask
root visit target and selected action
player perspective and terminal utility
search budget, RNG seed, schema version, and model digest
```

The current demo dataset proves the representation and loader contract, but it
is not the production replay format: it lacks atomic shard publication,
checksums, model identity, and full trajectory metadata.

## Failure semantics

- Backend startup or schema mismatch fails before games begin.
- A disconnect, malformed output, wrong request ID, or non-finite prediction is
  an evaluator error; search never substitutes a neutral result silently.
- Model replacement occurs only between explicit generation or arena runs.
- A production replay shard will be published only after its checksum and
  footer are complete.

## Benchmark layers

1. Core analysis and transitions in one Rust thread.
2. Parallel search with a deterministic in-process evaluator.
3. Rasterization, input packing, and backend inference as separate stages.
4. End-to-end model self-play through the shared broker.
5. Replay serialization and trainer consumption once production shards exist.

These boundaries distinguish game-state cost from search, batching, GPU
execution, and storage throughput.
