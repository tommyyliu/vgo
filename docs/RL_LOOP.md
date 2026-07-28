# Pipelined reinforcement-learning loop

The RL loop is a queue-driven actor/replay/learner pipeline. It no longer runs
generation, training, export, and Elo as one serial barrier:

```text
incumbent N ──> actor shard N ──> bounded replay snapshot ──> learner update
                    │                                      │
                    └── actor shard N (bounded prefetch)    └──> atomic model N+1
                                                                  │
                                                                  └──> Elo job queue
```

An actor captures one immutable model before starting a shard. Publication can
advance while that actor is still running; its replay remains valid because the
manifest records the behavior-model SHA-256. At most
`--maximum-prefetch-shards` completed or active shards can sit ahead of the
learner. This bounds policy lag and the pending disk/replay backlog; historical
immutable shards remain on disk. Set it to `0` for barriered
generation/training.

The default overlaps self-play with learning. Self-play alternates CPU-heavy
tree search with short GPU inference batches, while training is mostly GPU
work, so the two processes can fill each other's bubbles. Use
`--no-overlap-actor-learner` when profiling shows destructive kernel or context
contention. This serializes their execution; it does not unload the persistent
learner's CUDA allocations. Put inference and training on different GPUs when
their combined resident memory does not fit.

After each export, the default TensorRT path runs one full-batch inference
warmup. This builds the model-digest-scoped engine cache while the current actor
tail is still running, so the next shard does not pay engine-build latency on
its critical path. Pass `--no-warm-inference` to disable it. CUDA and CPU
providers skip this step.

## Run

From `training/`:

```powershell
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME `
  --updates 20 `
  --samples-per-shard 1024 --shards-per-update 1 --replay-window 8 `
  --maximum-prefetch-shards 1 `
  --resolution 96 --policy-resolution 32 --radius 0.05555555555555555 `
  --generation-simulations 256 --maximum-plies 256 `
  --actors 64 --leaf-batch 1 --inference-batch 16 --inference-slots 2 `
  --architecture ddrnet --model-width 64 --blocks 8 `
  --training-epochs 10 --training-batch 64 `
  --training-device cuda --training-precision bfloat16 `
  --provider tensorrt --inference-device-id 0 --fp16 --warm-inference
```

The command above changes only the update horizon from its default of 10.
Other important defaults are one 1,024-sample shard per update, an eight-shard
replay window, one prefetched shard, 256 simulations, a 96x96 raster with a
32x32 policy, DDRNet width 64 with 8 blocks, 10 learner epochs at batch 64,
TensorRT FP16 inference through two batch-16 lanes on device 0, and BF16 CUDA
training.
TensorRT engine warmup is enabled, promotion is disabled, and each accepted
model queues up to two 16-pair telemetry comparisons.

Important resource controls are:

- `--actors`: concurrent games feeding the shared inference broker. The local
  32-thread benchmark found 32 efficient and 64 maximum-throughput actors.
- `--leaf-batch`: unique MCTS leaves submitted together from one game. `1`
  preserves sequential MCTS; larger values change the search trajectory and
  help workloads with too few concurrent games to fill a model batch.
- `--inference-batch` and `--inference-delay-ms`: backend batch ceiling and
  broker collection window. The batch ceiling is part of the exported ONNX
  contract and cannot change when resuming a run.
- `--inference-slots`: independent inference lanes available to generation.
  Each lane owns its TensorRT session, execution context, and reusable input
  storage. It defaults to `2` and may be changed when resuming a run.
- `--inference-device-id`: CUDA device number used by Rust generation,
  promotion, and telemetry. It defaults to `0` and is independent of PyTorch's
  `--training-device`.
- `--warm-inference` / `--no-warm-inference`: enable or disable the post-export
  TensorRT warmup. It uses the configured inference batch, FP16 mode, and
  inference device. CUDA and CPU providers do not run it.
- `--writer-queue-games`: bounded completed-trajectory queue between actors and
  the replay writer. Actors backpressure instead of building a multi-gigabyte
  shard in memory.
- `--training-batch`: learner batch size. The learner uses two pinned host
  buffers and a separate CUDA copy stream.
- `--training-precision`: `bfloat16` by default. CUDA forward, loss, and metric
  evaluation use BF16 autocast while model parameters and optimizer state remain
  checkpoint-compatible. Use `float32` on a CUDA device without BF16 support.
  CPU training executes in FP32.

For two GPUs, a typical placement is
`--inference-device-id 0 --training-device cuda:1`. On one GPU, the defaults put
both workloads on device 0. TensorRT `--fp16` controls inference engine
precision; it is separate from learner `--training-precision`.

Bootstrap without a model uses the naive evaluator. To continue from a
published model, both artifacts are required:

```powershell
  --initial-checkpoint ../artifacts/PREV/updates/update-000019/candidate.pt `
  --initial-onnx ../artifacts/PREV/updates/update-000019/candidate.onnx `
  --initial-replay ../artifacts/PREV/replay/shard-000019/dataset.vgo
```

Initial replay participates in the training window but does not count as the
new replay quantum needed to trigger an update.

## Stage ownership

### Actors and inference

Rust actors run complete games and feed bounded inference lanes. A leaf
round is submitted as one ordered group rather than spawning an operating-system
thread per leaf. Repeated descents to the same pending tree edge share an
evaluation, terminal children bypass inference, and each node caches its spatial
fine grid.

Each broker coalesces groups up to the model's declared maximum batch and
validates output count and request IDs before routing results. Generation uses
two lanes by default; each owns a synchronous ONNX/TensorRT session so transfer
and execution latency can overlap without cloning request tensors. More lanes
increase host and device memory residency and should be selected from measured
end-to-end throughput rather than batch fill alone.

### Streaming replay

Only terminally labelled games enter the writer. Replay-v3 records stream
directly to `dataset.vgo.tmp` through a bounded completed-game channel. The
writer:

1. writes the final record count in the header;
2. hashes every byte as it is written;
3. truncates only the serialized tail of the final complete trajectory to hit
   the exact shard size;
4. flushes and fsyncs the file;
5. atomically renames it to `dataset.vgo`; and
6. publishes a manifest containing shard, model, game-range, queue, tail, and
   broker metrics.

The generator retains only `--examples` rasters. Once the exact boundary is
reached it signals cancellation, closes the completed-game queue, and joins
every actor after at most its current search. Only then are the inference lanes
and native sessions destroyed. This removes the former full-shard RAM copy
while keeping TensorRT teardown deterministic.

### Persistent learner

`vgo_training.learner` is one supervised JSON-lines process for the whole run.
Across updates it retains:

- model weights, Adam moments, and the compiled graph when enabled;
- prepared replay shards still present in the active window;
- game-stable train/validation assignments;
- two pinned host staging buffers and device buffers; and
- a background gather/augmentation worker plus an asynchronous CUDA copy
  stream.

Shards are prepared once, reused on the next update, and evicted as soon as they
leave the replay window. Metrics reduce on the device and synchronize once per
evaluation pass instead of once per batch and metric. BF16 autocast is the
pipeline default on supported CUDA devices; `--training-precision float32`
selects full-precision execution.

Every update names an authoritative parent checkpoint. If promotion rejected
the learner's resident candidate, or the service restarted, a path/signature
mismatch forces a reload of the accepted parent. The report includes the parent
SHA-256 and the coordinator verifies it before export. “Persistent” means one
learner process across updates in one coordinator invocation. After a full
coordinator restart, the service is recreated, reloads the authoritative
checkpoint, and rebuilds only the in-memory caches it needs.

### Publication and telemetry

Checkpoint, ONNX, update specification, publication decision, replay state, and
run state are written atomically. A candidate is visible to new actors only
after its complete publication record exists.

Before publication, TensorRT candidates are loaded once at the full configured
inference batch to populate the shared, model-digest-scoped engine cache. With
the default actor/learner overlap, this runs during the old actor's tail. The
warmup report is stored in `publication.json`; disabled and non-TensorRT paths
record no warmup report.

Promotion gating is optional. Enable it only with a meaningful score:

```powershell
  --promotion-arena --promotion-score 0.52 `
  --arena-pairs 40 --maximum-truncation-rate 0.02
```

Elo comparisons are telemetry, not a dependency of the next update. Accepted
models enqueue deterministic, idempotent comparison jobs. When drained, jobs
for one candidate are sent to one multi-opponent arena process, so candidate
model and TensorRT startup are amortized across every selected opponent. Each
match still publishes its own atomic result and can be recovered independently.
Drain the queue after the learning run:

```powershell
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME --telemetry-only
```

Alternatively add `--drain-telemetry` to the original invocation. These are
invocation controls and do not alter the saved pipeline configuration. The run
directory lease deliberately prevents a telemetry-only process from racing the
active learning coordinator; “off path” means telemetry is deferred, not that
it competes with training for the same device. Changing telemetry opponent or
pair counts on restart affects jobs enqueued by later publications; already
queued jobs retain their recorded opponents, seeds, and pair counts.

## Recovery and artifacts

Only one coordinator can hold an output directory's lease. State is reloaded
after acquiring that lease, so even a previously constructed coordinator cannot
overwrite newer progress. Repeating a command reads `pipeline-config.json` and
`pipeline-state.json`, verifies recovered replay/checkpoint/ONNX digests, and
continues from the first incomplete durable boundary.

Configuration is divided into two classes:

- **Learning identity** is immutable within a run. It includes replay/search
  semantics, raster and policy shapes, model and optimizer hyperparameters,
  training precision, provider/FP16 mode, inference batch contract, promotion
  policy, seeds, and bootstrap artifacts. Changing one requires a new output
  directory.
- **Operational controls** may change on restart: `--updates`,
  `--maximum-prefetch-shards`, `--actors`, `--writer-queue-games`,
  `--inference-delay-ms`, `--inference-slots`, `--inference-device-id`,
  `--training-threads`, `--warm-inference`, `--training-device`, `--compile`,
  `--overlap-actor-learner`,
  `--arena-actors`, `--telemetry-opponents`, and `--telemetry-pairs`.
  `--updates` may increase but cannot be reduced below completed work.

Rerun the original command with the desired operational overrides; all
non-default learning-identity flags must still match the saved run. The current
effective configuration is written to `pipeline-config.json`, and every change
is appended to `pipeline-config-history.json` with its replay/update boundary.
`--telemetry-only` instead loads the saved configuration directly.

Each run contains:

```text
pipeline-config.json
pipeline-config-history.json
pipeline-state.json
replay/shard-NNNNNN/
    dataset.vgo
    manifest.json
updates/update-NNNNNN/
    update-spec.json
    candidate.pt
    candidate.pt.json
    candidate.onnx
    candidate.onnx.json
    publication.json
telemetry/
logs/
run.json
```

TensorRT engine plans live in the shared `artifacts/onnx-cache/`, keyed by model
digest and runtime contract rather than by update number.

If a process stops after publishing an artifact but before advancing state, the
next run validates and reconciles the publication idempotently. Incomplete
replay staging is private and regenerated. On POSIX, cancelling an external
stage terminates its complete process group, including the binary launched by
Cargo, before staging is reused. On Windows, only termination of the direct
Cargo process is currently guaranteed. `run.json` reports active coordinator
wall time for the learning pipeline separately from elapsed calendar time, so
downtime between restarts is visible without being counted as compute.

## Utilization feedback loop

`run.json.utilization` aggregates durable replay manifests and publication
reports. It is a pipeline-level accounting view, not a replacement for GPU
profiler traces:

| Field | Meaning and tuning signal |
| --- | --- |
| `measured_generation_shards`, `measured_updates` | Number of artifacts that contributed valid metrics; use these to judge whether the aggregate has enough steady-state data. |
| `generation_wall_seconds`, `learning_wall_seconds` | Sum of actor-stage and complete update-publication wall time. |
| `learner_wall_seconds`, `learner_optimization_seconds` | Time inside the persistent learner and its optimization subset. |
| `pipeline_overlap_factor` | `(generation wall + learning wall) / active coordinator wall`. Values above 1 show concurrent stage work; compare end-to-end cadence as well, since contention can lengthen both stages. |
| `average_active_games` | Summed game time divided by generation wall time, a concurrency-occupancy measure for actors. |
| `inference_batch_fill` | Inference positions divided by `batches * --inference-batch`; raw positions, batches, and inference seconds are reported alongside it. |
| `writer_backpressure_seconds` | Actor time blocked behind the completed-game writer queue. |
| `learner_optimization_fraction` | Optimization time divided by learner wall time; a low value points to preparation, evaluation, or synchronization overhead. |
| `replay_cache_hit_ratio` | Prepared-shard hits divided by all cache accesses; raw hit/miss counts distinguish a small sample from persistent churn. |

Tune operational controls from these measurements: actor count and inference
delay affect game occupancy and batch fill; inference slots trade additional
session memory for overlapping backend latency; writer queue depth can absorb
short serialization bursts; and actor/learner overlap can be disabled when
concurrent stage work lowers total throughput. `--inference-batch` is a
learning-identity contract, so changing it requires a new run. A cache miss is expected for a new
shard and after a coordinator restart; repeated misses for stable shards point
to lost learner persistence or replay-window churn. Compare steady-state
updates rather than the first update, which includes compilation and TensorRT
cache creation.

## Search and replay semantics

Opening moves sample root visits as
`P(a) ∝ visits(a)^(1 / temperature)` until `--temperature-plies`; later moves
use argmax. Arenas always use argmax.

Spatial progressive widening draws only the cumulative proposal-budget delta.
Replay v3 stores root visits, exact proposal probability, and raw proposal
multiplicity. Python prepares the self-normalized sparse target and full-legal
denominator once per immutable shard. Replay v1 and v2 remain loadable. See
[`POLICY_REDESIGN.md`](POLICY_REDESIGN.md) for the target derivation and
limitations.

The local baseline measurements that motivated the pipeline are retained under
[`../benchmarks/results/`](../benchmarks/results/): generation, training,
export, and Elo previously formed a roughly 322-second serial cycle, with Elo
alone accounting for about 21% despite being pure telemetry. The new design
removes Elo from candidate cadence and overlaps the two dominant compute
producers; measure the resulting steady-state cadence on the target GPU rather
than comparing the first update, which includes compilation and TensorRT cache
warmup.
