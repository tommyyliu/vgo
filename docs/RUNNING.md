# Running everything

Every command below is run from the repository root unless it says otherwise.
Python commands run from `training/` because that is where the `uv` project
lives.

## Prerequisites

- Rust via `rust-toolchain.toml`; `cargo test --workspace` will fetch what it
  needs.
- `uv` for the Python side. `uv run` creates and syncs `training/.venv` on first
  use.
- Chrome at `C:\Program Files\Google\Chrome\Application\chrome.exe` for the
  browser test suites.
- An NVIDIA GPU for `--training-device cuda` and `--provider tensorrt`. Rust
  inference selects it with `--inference-device-id`; PyTorch accepts ordinary
  device strings such as `cuda`, `cuda:0`, or `cuda:1`. The CPU paths work
  everywhere and are much slower.

## Tests

There are four suites and no single command that runs them all.

```powershell
cargo test --workspace                 # Rust rules engine, search, self-play
.\reference\tests\run-tests.ps1        # engine, game tree, and UI fixtures in Chrome
```

```powershell
cd training
uv run python -m unittest discover -s tests -v
```

`run-tests.ps1` drives headless Chrome and fails the run if any fixture reports
a status other than `pass`, so it is safe in a pipeline.

## The reinforcement-learning loop

[`RL_LOOP.md`](RL_LOOP.md) describes the queue, artifact, and recovery
contracts. A representative run is:

```powershell
cd training
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME `
  --updates 20 --samples-per-shard 1024 `
  --shards-per-update 1 --replay-window 8 --maximum-prefetch-shards 1 `
  --resolution 96 --policy-resolution 32 --coarse-pool 4 `
  --generation-simulations 256 --arena-simulations 256 `
  --maximum-plies 256 `
  --training-epochs 10 --training-batch 64 `
  --warm-learning-rate 5e-4 --value-weight 1.0 `
  --training-device cuda --training-precision bfloat16 --actors 64 `
  --inference-batch 16 --inference-slots 2 `
  --provider tensorrt --inference-device-id 0 `
  --warm-inference
```

Self-play and the persistent learner overlap by default. If same-GPU kernel or
context contention hurts cadence, pass `--no-overlap-actor-learner`. That flag
does not unload the learner's resident allocations; for memory isolation on a
multi-GPU machine, use for example
`--inference-device-id 0 --training-device cuda:1`.

Learner execution defaults to BF16 autocast on supported CUDA hardware. Pass
`--training-precision float32` for older GPUs. TensorRT's default `--fp16`
controls inference separately.

TensorRT inference warmup also defaults on. After exporting a candidate, the
coordinator runs one full configured batch to populate the model-digest-scoped
engine cache while the current actor tail finishes. Pass
`--no-warm-inference` to opt out; CUDA and CPU providers skip this step.

Continue from a published model by passing both halves, and optionally seed the
replay window:

```powershell
  --initial-checkpoint ../artifacts/PREV/updates/update-000019/candidate.pt `
  --initial-onnx ../artifacts/PREV/updates/update-000019/candidate.onnx `
  --initial-replay ../artifacts/PREV/replay/shard-000019/dataset.vgo
```

`--coarse-pool` is the coarse-sampling knob. The pipeline default is `4`; zero
uses legacy candidates. The value is forwarded unchanged to generation, every
arena, and Elo telemetry, and must not exceed `--policy-resolution`. With no
initial model, bootstrap generation uses the naive evaluator and falls back to
legacy candidates because it has no spatial policy grid. Supply an initial
checkpoint/ONNX pair to exercise coarse generation immediately, or let it begin
after the first accepted model.

Current coarse replay is version 3: it stores raw visits, beta, and `u32`
proposal multiplicities. Versions 1 and 2 remain loadable in the same replay
window. The sparse corrected targets and full-legal masks are prepared once
before training rather than recomputed each epoch.

### Resuming

The driver atomically writes `pipeline-state.json`, immutable update
specifications, and publication records. Rerunning the same `--output`
revalidates already-published replay and model digests, then resumes the first
incomplete unit. A file lease excludes a second coordinator, and state is
reloaded after the lease is acquired.

Learning-identity settings cannot change in place: these include replay/search
semantics, shapes, architecture and optimization hyperparameters, training
precision, provider/FP16 mode, inference batch contract, promotion policy,
seeds, and bootstrap artifacts. Use a new output directory for one of those
changes.

Operational controls may change on restart. They include the update target,
prefetch depth, actor/writer counts, inference delay/slots/device, training
device/thread count/compilation, inference warmup, overlap mode, arena actors,
and telemetry counts. Repeat the original command with the new operational
values; any non-default learning settings must still be supplied unchanged.
`--updates` may increase but cannot be set below completed work. Effective
changes are recorded in `pipeline-config-history.json`.

Elo jobs do not block the next learner update. Drain them independently:

```powershell
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME --telemetry-only
```

The drain batches all pending opponents for one candidate into one arena
process, amortizing candidate/TensorRT startup while retaining one atomic result
per comparison. It uses the run-directory lease, so invoke it after the learning
coordinator exits rather than concurrently. `--drain-telemetry` performs the
same drain after learning in one invocation.

### Tuning utilization

Inspect `run.json.utilization` after a steady-state run. It reports measured
generation/update coverage, stage and learner wall times,
`pipeline_overlap_factor`, `average_active_games`, inference batch
fill/positions/batches/time, writer backpressure, the learner optimization
fraction, and prepared-replay cache hits and misses. Use actor count and
inference delay to tune occupancy and fill, vary `--inference-slots` to trade
session memory for overlapped inference latency, use queue depth to absorb short writer
bursts, and compare default overlap with `--no-overlap-actor-learner` when
contention increases total cadence. The full field definitions and caveats are
in [`RL_LOOP.md`](RL_LOOP.md#utilization-feedback-loop).

## Measuring strength

The optional promotion arena asks whether a candidate clears its incumbent.
Queued telemetry compares accepted generations and fits common Bradley–Terry
ratings without blocking publication. For an ad hoc head-to-head, run the arena
yourself:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-arena -- `
  --candidate artifacts/A/updates/update-000019/candidate.onnx `
  --opponent  artifacts/B/updates/update-000019/candidate.onnx `
  --pairs 75 --simulations 256 --max-plies 256 --threads 32 `
  --resolution 96 --policy-resolution 32 --coarse-pool 4 `
  --radius 0.05555555555555555 `
  --maximum-batch 16 --provider tensorrt --device-id 0 `
  --cache-directory artifacts/onnx-cache
```

Omit `--opponent` to play the naive policy. Games are colour-swapped in pairs,
so `--pairs 75` is 150 games. Set `--threads` near `--pairs`: throughput scales
with concurrent games, not threads, because MCTS keeps only one evaluation in
flight per game. Arena verdicts were measured identical from 1 to 48 threads, so
parallel arenas are promotion-grade; see the note in `docs/RL_LOOP.md`.

Run it through `runtime_environment()` if you invoke it from Python, or from a
shell where the TensorRT libraries are already on `PATH`. See the traps below.

### Is search working at all?

`vgo-arena` drives both seats at one `--simulations`, so it cannot tell you
whether search depth is buying anything. `vgo-playout-duel` holds the model
fixed and varies only the playout budget, which isolates the search path from
policy quality:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-playout-duel -- `
  --model artifacts/A/updates/update-000019/candidate.onnx `
  --high 128 --low 16 --pairs 40 --coarse-pool 8 `
  --max-plies 160 --threads 32 --resolution 96 --policy-resolution 32 `
  --radius 0.05555555555555555 --maximum-batch 64 `
  --provider tensorrt --cache-directory artifacts/onnx-cache
```

The deep seat should win comfortably. If it does not, the fault is in search
rather than in the policy, because both seats share one evaluator. Pass
`--coarse-pool 0` to run the same comparison on the legacy candidate path;
running both is how you tell a coarse-sampling regression from a general search
regression. Use the same `--radius` and `--resolution` the model trained at, or
the net sees out-of-distribution rasters and inference returns NaN.

## Benchmarks

```powershell
cargo run --release -p vgo-selfplay --bin vgo-canary -- `
  --pairs 100 --first 1000 --second 10 --max-plies 48 --threads 4 --seed 10001

cargo run --release -p vgo-selfplay --bin vgo-model-smoke -- `
  --resolution 128 --policy-resolution 32
```

```powershell
cd training
uv run python -m vgo_training.benchmark_model `
  --checkpoint ../artifacts/raster-demo/model.pt --batches 1,8,16,32,64
uv run python -m vgo_training.benchmark_precision `
  --dataset ../artifacts/raster-demo/dataset.vgo `
  --checkpoint ../artifacts/raster-demo/model.pt
```

## The reference application

Open `reference/js-reference/voronoi_go.html` directly in a browser; it has no
build step and loads the engine modules from `reference/src/`. Paste a position
or a game record into the VGO-SGF box and press Load.

## Traps

These are all things that fail confusingly rather than obviously.

**TensorRT libraries are not on `PATH`.** They ship inside the `uv` virtual
environment. `rl_loop.runtime_environment()` prepends `tensorrt_libs` and
`torch/lib` for every child process it spawns. Invoking `vgo-arena` or
`vgo-generate-demo` yourself without that environment fails with
`Error loading onnxruntime_providers_tensorrt.dll ... nvinfer_10.dll is
missing`. Import the helper rather than reimplementing the path logic.

**`maximum_batch` is frozen into each ONNX at export time.** A model exported at
16 cannot be driven at 64, and the error is
`configured batch 64 exceeds ONNX maximum 16`. Two models can only meet at the
smaller of their two ceilings, which silently limits both generation throughput
and cross-generation comparison. Re-export with
`uv run python -m vgo_training.export_onnx --checkpoint X --output Y
--maximum-batch 64`.

**The ply-limit flag is spelled differently in each layer.** The driver takes
`--maximum-plies`; the Rust binaries take `--max-plies`.

**A coarse pool larger than the policy grid is rejected.** `--coarse-pool`
counts fine policy cells per coarse region, so use a value from `1` through
`--policy-resolution` when enabling spatial sampling. Zero explicitly selects
the legacy path.

**Truncated games block promotion.** A game that hits the ply limit is excluded
from the arena score, and `--maximum-truncation-rate` rejects the candidate if
too many do. Stronger models play longer games, so a limit that was comfortable
early starts vetoing good candidates later. Raise `--maximum-plies` rather than
the truncation tolerance: a truncated game is an invalid measurement, not a
tolerable one.

**Direct-binary and pipeline defaults are different contracts.** The pipeline
sets both generation and arena search to 256 simulations. Direct Rust binaries
retain their own diagnostic defaults. When reproducing a pipeline generation or
arena, pass `--simulations`, `--resolution`, `--policy-resolution`,
`--maximum-batch`, and `--device-id` explicitly.

**Draws mean the search is too shallow to separate moves.** Voronoi-area scoring
ties only when the areas match to `1e-10`, which in practice means a
mirror-symmetric finish. Self-play at 128 simulations produces none at all;
arenas at 16 produce 12-22%. A nonzero draw rate is a useful signal rather than
a curiosity.

**Replay shards are large.** Replay v3 stores semantic states plus several
policy-supervision arrays. The persistent learner retains prepared tensors for
every shard in `--replay-window`; use each manifest's `dataset_bytes` and the
learner cache report when sizing host memory. `artifacts/` is gitignored for
this reason.

**BF16 training and FP16 inference are independent.** `--training-precision
bfloat16` controls PyTorch autocast and fails early if the selected CUDA device
does not support BF16. `--fp16` controls TensorRT engine precision. Changing
training precision or inference FP16 mode changes the run's learning identity.

**A silently killed training stage is usually the device running out of memory.**
Under a process supervisor the traceback can be lost with the process tree,
leaving no log and no exit code. Rerun the failing stage in the foreground to
see it.
