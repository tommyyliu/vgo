# Self-play architecture

## Ownership

Rust owns gameplay, search, native model inference, arenas, and replay
serialization:

- `vgo-core`: exact rules, geometry, transitions, termination, and scoring;
- `vgo-raster`: the model's input tensors, and the shard renderer the Python
  loader calls;
- `vgo-search`: evaluators, progressive-widening MCTS, spatial proposals, and
  visit-based move selection;
- `vgo-inference`: bounded request grouping over ONNX Runtime/TensorRT; and
- `vgo-selfplay`: complete-game playouts, actor pools, arenas, and immutable
  per-game dataset publication.

Python owns replay preparation, optimization, checkpoints and ONNX export.
`scripts/bulk-loop.sh` orchestrates. Python never imports the
simulator, and Rust never implements neural-network layers. Replay shards and
self-describing model artifacts are the durable boundary.

## Hot path

```text
actor game
  -> MCTS leaf round
  -> rasterize unique pending leaves
  -> shared bounded inference broker
  -> one ONNX/TensorRT batch
  -> validated ordered evaluations
  -> MCTS backup
  -> completed labelled trajectory
  -> bounded writer queue
  -> streaming replay-v3 writer
```

MCTS submits a leaf round through `Evaluator::evaluate_batch`. Simple
evaluators inherit a sequential default; `BatchedEvaluator` sends a grouped
request directly to the model broker. This avoids creating scoped operating
system threads and one response channel per leaf.

Before inference, the search coalesces repeated descents to the same pending
tree path and resolves terminal child transitions exactly. Backups still occur
in descent order. Spatial `FineGrid` construction, including the fact that a
policy has no spatial grid, is cached on each node. The legacy candidate
sequence is constructed lazily only when a spatial policy is unavailable.

`leaf_batch = 1` is test-pinned to the original sequential action and visit
counts. Larger values deliberately alter exploration through virtual loss.

## Inference broker

Every backend declares its raster grid, policy grid, and maximum batch. The
broker:

- bounds the number of queued request groups;
- preserves a grouped leaf request rather than splitting it across unrelated
  completions;
- packs compatible groups until the batch ceiling or latency deadline;
- validates output count and every request ID; and
- reports per-position encoding, queue, inference, failure, and observed-batch
  metrics.

Generation feeds every grouped call into one shared broker queue. The broker
packs across all actors until the batch ceiling or deadline, then dispatches
the batch to one of `--inference-slots` session slots (default `2`). Each slot
has its own reusable host buffer, ONNX session, and TensorRT execution context,
while actor search and rasterization continue in parallel. Additional slots
trade session and execution-context memory for overlapping inference latency,
so tune the count from end-to-end throughput on the target GPU.

Generation and arenas receive an explicit `device_id`; the pipeline exposes it
as `--inference-device-id` and forwards the same value to self-play, promotion,
and telemetry. This is intentionally separate from the Python learner's
`--training-device`, so a multi-GPU run can keep Rust inference on (for example)
device 0 and PyTorch on `cuda:1`.

For TensorRT, the coordinator runs a full configured inference batch after each
export by default. That primes the shared, model-digest-scoped engine cache
during the current actor tail instead of delaying the next shard's first model
load. `--no-warm-inference` disables the operational warmup; CUDA and CPU
providers skip it.

## Playout contract

`vgo-selfplay::play_game` is the sole owner of whole-game progression:

- exact position fingerprints and repetition avoidance;
- preferred-action fallback;
- transition application and event accounting;
- pass/pass termination and maximum-ply bounds; and
- accumulated search and gameplay statistics.

The canary, model smoke tests, arenas, and replay generator therefore share game
semantics. MCTS owns only tree-local simulation.

## Streaming replay

Actor threads send only complete, terminally labelled games through a bounded
`sync_channel`. The consumer serializes replay-v3 records immediately. It never
holds a full shard of semantic rasters in memory.

The header advertises the exact requested record count. Whole completed games
are admitted until that count is reached; only excess records in the final
complete game are omitted. The writer hashes bytes while writing, flushes,
fsyncs, atomically renames `dataset.vgo.tmp`, and syncs the containing
directory. A failed or incomplete stream removes its private temporary file and
never publishes a shard.

The manifest identifies:

- replay schema and dataset digest/size;
- behavior-model digest and immutable shard ID;
- first/last serialized game IDs and per-record game/seed identity;
- search, actor, queue, raster, and policy configuration;
- attempted, completed, discarded, failed, tail, writer, and wall timings; and
- broker batch/utilization timings.

Only requested example rasters survive for image output. At the exact boundary,
the collector signals cancellation and closes the completed-game receiver.
Actors finish at most their current search, then every handle is joined before
the inference brokers and native sessions are destroyed. This bounds the tail
without allowing TensorRT process-exit cleanup to race an in-flight inference.

## Loop boundary

`vgo-generate-continuous` writes each finished game as its own one-game
dataset under `games/gen-N-<sha>/`, where the directory names the model that
played it. A generator holds one model for its whole life. When the loop
exports a new model it starts a second generator on it and touches the first
one's stop file, so the old generator finishes the games it holds while the new
one is already producing. Policy lag is therefore explicit and bounded by one
game per actor, and nothing swaps an evaluator on the per-leaf hot path.

`scripts/train-once.py` selects the newest games totalling the window by game
number, not directory order, and trains a fresh learner process. Nothing
persists between rounds except files.

## Failure semantics

- Model schema or backend startup errors fail before games begin.
- A disconnect, malformed output, wrong ID, output-count mismatch, or non-finite
  prediction aborts search; no neutral fallback is synthesized.
- Evaluator identity is fixed for an entire generator or arena.
- A game appears only after complete serialization and an atomic rename, so a
  killed generator loses only the games in flight.

## Measurement layers

1. Core analysis and transition cost in one Rust thread.
2. Tree search with an in-process deterministic evaluator.
3. Rasterization, request grouping, backend inference, and output extraction.
4. End-to-end actor games through the shared broker.
5. Completed-game queue occupancy, writer backpressure, and durable
   publication.

Each generator reports broker batch fill and stage timings in `generate.log`.
Keeping these boundaries separate keeps CPU search, GPU execution and storage
costs separately attributable.
