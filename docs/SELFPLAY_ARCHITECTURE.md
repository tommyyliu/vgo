# Self-play architecture

## Ownership

Rust owns gameplay, search, native model inference, arenas, and replay
serialization:

- `vgo-core`: exact rules, geometry, transitions, termination, and scoring;
- `vgo-raster`: canonical semantic tensors and visual diagnostics;
- `vgo-search`: evaluators, progressive-widening MCTS, spatial proposals, and
  visit-based move selection;
- `vgo-inference`: bounded request grouping plus ONNX Runtime/TensorRT and
  diagnostic Python backends; and
- `vgo-selfplay`: complete-game playouts, actor pools, arenas, and immutable
  replay-shard publication.

Python owns replay preparation, optimization, checkpoints, ONNX export,
publication orchestration, and telemetry scheduling. Python never imports the
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

## Pipeline boundary

The Python coordinator consumes immutable shards, not actor-owned buffers. It
may train and run the next actor shard concurrently. Every actor captures an
incumbent before starting, and the shard records that model digest, so a
publication during generation creates explicit bounded policy lag rather than
ambiguous mixed-policy data. `--maximum-prefetch-shards` bounds the active and
completed shards ahead of the learner; zero restores barriered scheduling.

One supervised learner process owns the model, optimizer, optional compiled
graph, prepared replay-window cache, and pinned staging buffers across every
update in one coordinator invocation. Training defaults to BF16 autocast on
supported CUDA devices. A coordinator restart creates a new learner process and
reloads the authoritative accepted checkpoint; persistent state never
substitutes for the immutable checkpoint/replay boundary.

The run configuration separates learning identity from operational placement
and concurrency. Search/replay/model/optimizer semantics remain fixed, while
the update target, prefetch depth, actor counts, device placement, compilation,
TensorRT warmup, and telemetry capacity may change at a restart and are
recorded in config history.

See [`RL_LOOP.md`](RL_LOOP.md) for replay-window scheduling, persistent learner
ownership, promotion, recovery, and off-path telemetry.

## Failure semantics

- Model schema or backend startup errors fail before games begin.
- A disconnect, malformed output, wrong ID, output-count mismatch, or non-finite
  prediction aborts search; no neutral fallback is synthesized.
- Evaluator identity is fixed for an entire shard or arena.
- Replay appears only after exact-size serialization and durable publication.
- On POSIX, coordinator cancellation targets the full subprocess group so the
  binary launched through Cargo cannot remain as an orphan GPU consumer. On
  Windows, the current supervisor guarantees only direct Cargo-process
  termination, not descendant-tree cleanup.
- A single run-directory lease prevents two coordinators from publishing the
  same sequence.
- State is reloaded only after that lease is acquired; recovered replay,
  checkpoint, ONNX, and publication identities are checked before reuse.
- Telemetry is queued outside candidate cadence. A drain groups all selected
  opponents for one candidate into one arena process, amortizing model/provider
  startup while publishing an atomic result for each match.

## Measurement layers

1. Core analysis and transition cost in one Rust thread.
2. Tree search with an in-process deterministic evaluator.
3. Rasterization, request grouping, backend inference, and output extraction.
4. End-to-end actor games through the shared broker.
5. Completed-game queue occupancy, writer backpressure, tail waste, and durable
   replay publication.
6. Steady-state actor/learner overlap and candidate-publication cadence.

The coordinator aggregates these boundaries in `run.json.utilization`:
generation/update sample counts, stage and optimizer wall time, overlap factor,
active-game occupancy, inference batch fill, writer backpressure, optimization
fraction, and prepared-replay cache reuse. See
[`RL_LOOP.md`](RL_LOOP.md#utilization-feedback-loop) before changing operational
concurrency controls.

These boundaries keep CPU search, GPU execution, storage, and orchestration
costs separately attributable.
