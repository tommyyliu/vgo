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
  -> actor-owned streaming position-replay writer
  -> atomic game-directory publication
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

The continuous generator loads its TensorRT slots sequentially so builders do
not race over the cache. Model-digest-scoped engine caches and a shared timing
cache reduce subsequent startup cost. A change to precision, shape, or batch
limits can require an engine rebuild.

## Playout contract

`vgo-selfplay::play_game` is the sole owner of whole-game progression:

- selection of the search's preferred playable action, or pass if none exists;
- transition application and event accounting;
- pass/pass termination and maximum-ply bounds; and
- accumulated search and gameplay statistics.

The arena and replay generator share this progression. The current Vgo rules
permit repetition; this driver adds no repetition prohibition. MCTS owns only
tree-local simulation.

## Streaming replay

Each continuous actor generates and writes its own complete game; there is no
collector or completed-game queue. Replay stores positions and sparse search
records, not a shard of rendered model inputs. The stream hashes and publishes
the dataset inside a private staging directory; the game directory is renamed
into place after its manifest and sidecars have been written.

The manifest identifies the dataset digest, model digest when present, sample
count, nominal raster dimensions, policy size, and simulation budget.
`games.jsonl` records the actual per-game radius/komi, final scores and stones,
cap status, and soft resignation. Full resolved run configuration and stage
timings live in `performance.jsonl`, rather than being repeated in every game
manifest. `resign-calibration.jsonl` stores counterfactual threshold outcomes.

A stop file drains games already in flight. Cancellation aborts searches on
failure or at a profiling deadline. Actor handles are joined before inference
sessions are released; this avoids teardown racing active calls.

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
5. Per-actor writing time and successfully published games/samples.

The continuous generator appends startup, periodic, and final records to each
generation's `performance.jsonl`, with brief summaries in `generate.log`.
Records distinguish work in flight from published data, include process memory,
and group search work by rounded board size, thirds of the game cap, and soft
resignation status. Neural runs include interval and cumulative broker counters.
The bulk loop's `events.jsonl` marks training and generator handoffs. See
[`RUNNING.md`](RUNNING.md#continuous-run-performance) for fields and interpretation.

Search wall time includes inference waits, parallel duration sums overlap, and
queue durations are position-weighted. These metrics expose stage costs; their
sum is not a decomposition of CPU time. Use a flamegraph to locate CPU work.
