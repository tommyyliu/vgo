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
- An NVIDIA GPU for `--device cuda` and `--provider tensorrt`. The CPU provider
  works everywhere and is much slower.

## Tests

There are four suites and no single command that runs them all.

```powershell
cargo test --workspace                 # Rust rules engine, search, self-play
.\reference\tests\run-tests.ps1        # engine, game tree, and UI fixtures in Chrome
```

```powershell
cd training
uv run python -m unittest tests.test_dataset tests.test_model `
  tests.test_metrics_batching tests.test_rl_loop
```

The Python suites must be named explicitly. `unittest discover` fails with
`Start directory is not importable` because `training/tests/` has no
`__init__.py`.

`run-tests.ps1` drives headless Chrome and fails the run if any fixture reports
a status other than `pass`, so it is safe in a pipeline.

## The reinforcement-learning loop

`docs/RL_LOOP.md` describes what the five stages do. This is how to drive them.

```powershell
cd training
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME `
  --iterations 4 --samples 3072 --replay-window 4 `
  --resolution 128 --coarse-pool 8 `
  --generation-simulations 128 --arena-simulations 128 `
  --maximum-plies 96 `
  --epochs 120 --training-batch 32 --warm-learning-rate 1e-4 --value-weight 0.1 `
  --device cuda --actors 64 --arena-actors 1 --arena-pairs 40 `
  --maximum-batch 64 --provider tensorrt `
  --promotion-score 0.52 --maximum-truncation-rate 0.02
```

Continue from a published model by passing both halves of it, and add its replay
so the window does not restart empty:

```powershell
  --initial-checkpoint ../artifacts/PREV/iteration-000/model/candidate.pt `
  --initial-onnx ../artifacts/PREV/iteration-000/model/candidate.onnx `
  --initial-replay ../artifacts/PREV/iteration-000/replay/dataset.vgo
```

`--coarse-pool` is the only coarse-sampling knob. Its default `0` uses legacy
candidates; a positive value is forwarded unchanged to generation, every arena,
and Elo telemetry, and must not exceed `--resolution`. With no initial model,
iteration-zero generation uses the naive evaluator and therefore falls back to
legacy candidates because it has no spatial policy grid. Supply an initial
checkpoint/ONNX pair to exercise coarse generation immediately, or let it begin
after the first accepted model.

Current coarse replay is version 3: it stores raw visits, beta, and `u32`
proposal multiplicities. Versions 1 and 2 remain loadable in the same replay
window. The sparse corrected targets and full-legal masks are prepared once
before training rather than recomputed each epoch.

### Resuming

The driver writes `progress.json` after every stage, so rerunning the same
`--output` with the same arguments resumes at the first incomplete stage rather
than regenerating replay. A directory containing `run.json` is final and is not
overwritten, and `run-config.json` refuses a resume whose parameters differ. To
change a parameter, use a new output directory and pass the finished shards
through `--initial-replay`.

## Measuring strength

The in-loop arenas answer two narrow questions: is the candidate better than the
naive policy, and is it better than the model immediately before it. Neither
places generations on a common scale, and the naive baseline saturates. To
compare models directly, run the arena yourself:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-arena -- `
  --candidate artifacts/A/model/candidate.onnx `
  --opponent  artifacts/B/model/candidate.onnx `
  --pairs 75 --simulations 128 --max-plies 96 --threads 4 `
  --resolution 128 --coarse-pool 8 --radius 0.16666666666666666 `
  --maximum-batch 16 --provider tensorrt --cache-directory artifacts/onnx-cache
```

Omit `--opponent` to play the naive policy. Games are colour-swapped in pairs,
so `--pairs 75` is 150 games. Use `--threads 1` for a promotion-grade verdict
and more actors only for aggregate measurements; see the determinism note in
`docs/RL_LOOP.md`.

Run it through `runtime_environment()` if you invoke it from Python, or from a
shell where the TensorRT libraries are already on `PATH`. See the traps below.

## Benchmarks

```powershell
cargo run --release -p vgo-selfplay --bin vgo-canary -- `
  --pairs 100 --first 1000 --second 10 --max-plies 48 --threads 4 --seed 10001

cargo run --release -p vgo-selfplay --bin vgo-model-smoke
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

**A coarse pool larger than the raster is rejected.** `--coarse-pool` counts
fine cells per coarse region, so use a value from `1` through `--resolution`
when enabling spatial sampling. Zero explicitly selects the legacy path.

**Truncated games block promotion.** A game that hits the ply limit is excluded
from the arena score, and `--maximum-truncation-rate` rejects the candidate if
too many do. Stronger models play longer games, so a limit that was comfortable
early starts vetoing good candidates later. Raise `--maximum-plies` rather than
the truncation tolerance: a truncated game is an invalid measurement, not a
tolerable one.

**Arena simulations default to 16 while generation defaults to 64.** Promotion
is decided at a fraction of the search depth the training data was generated
at, which measures a shallower player than the one being built. Measured on one
model pair, the apparent edge fell from 63% at 16 simulations to 58% at 64 and
128. Set `--arena-simulations` to match `--generation-simulations` unless you
are deliberately measuring shallow play.

**Draws mean the search is too shallow to separate moves.** Voronoi-area scoring
ties only when the areas match to `1e-10`, which in practice means a
mirror-symmetric finish. Self-play at 128 simulations produces none at all;
arenas at 16 produce 12-22%. A nonzero draw rate is a useful signal rather than
a curiosity.

**Replay shards are large.** 3072 samples at 128x128x10 float32 is 2.3 GB, so a
four-deep window is roughly 10 GB of host memory. `artifacts/` is gitignored for
this reason.

**A silently killed training stage is usually the device running out of memory.**
Under a process supervisor the traceback can be lost with the process tree,
leaving no log and no exit code. Rerun the failing stage in the foreground to
see it.
