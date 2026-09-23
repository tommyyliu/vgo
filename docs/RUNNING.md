# Running everything

Commands run from the repository root unless they say otherwise. Python runs
from `training/`, where the `uv` project lives.

## Setup

```bash
./scripts/setup.sh           # uv + rustup if absent, venv sync, release build, ORT check
./scripts/setup.sh --check   # verify only
```

The check worth having is the ONNX Runtime one. The Rust binaries `dlopen`
`libonnxruntime` from `ORT_DYLIB_PATH`, and a *failed* load hangs instead of
erroring: the error is built through `ort::api()`, which waits on the lock the
failing load still holds. The process sits at 0% CPU and looks exactly like a
slow TensorRT engine build.

Anything that runs a Rust binary against a model needs that environment:

```bash
source scripts/env/ort.sh    # ORT_DYLIB_PATH and LD_LIBRARY_PATH from the venv
```

The scripts below source it themselves.

What the machine needs: an NVIDIA **driver** (the wheels carry their own CUDA
and TensorRT, including for Blackwell/sm_120), Rust via `rust-toolchain.toml`,
`uv`, and a C compiler for `torch.compile`.

## Tests

```bash
cargo test --release --workspace
cd training && .venv/bin/python -m unittest discover -s tests
```

The geometry research code sits behind a feature:

```bash
cargo test --release -p vgo-core -p vgo-search --features vgo-search/iteration-lab
```

The browser suite for the reference implementation is `reference/tests/`; see
its README.

## The loop

```bash
VGO_OUTPUT=artifacts/my-run VGO_SEED_MODEL=artifacts/sl-w64b16/sl-w64b16.onnx \
  ./scripts/bulk-loop.sh
```

Every knob is an environment variable documented in the script header:
`VGO_WINDOW_SAMPLES`, `VGO_TURNOVER`, `VGO_EPOCHS`, `VGO_SIMULATIONS`,
`VGO_ACTORS`, the `VGO_RESIGN_*` and `VGO_ANCHOR_*` families, and so on. The
script resumes from whatever `models/` holds, so rerunning it after a stop
continues the run.

A run directory holds:

| path | contents |
|---|---|
| `games/gen-N-<sha>/game-M/` | `dataset.vgo`, `manifest.json`, `resign-calibration.jsonl` |
| `models/update-N.{pt,onnx}` | each round's checkpoint and export, with `.json` sidecars |
| `run.log`, `generate.log`, `train.log` | append-only across launches |
| `anchor.jsonl`, `anchor.log` | rating matches |

`run.log` spans every launch. Grep only from the latest
`===== loop start` marker, or old tracebacks read as live.

To stop without losing games in flight:

```bash
./scripts/stop-continuous.sh         # drain: each generator finishes its games
./scripts/stop-continuous.sh --now   # kill immediately
```

`scripts/memory-watchdog.sh` kills training before the box swaps. A window that
does not fit fills swap during loading, and a thrashing machine is worse than a
clean abort.

## Training or exporting by hand

```bash
cd training
../scripts/train-once.py --games-root ../artifacts/my-run/games --window-samples 150000 \
  --output ../artifacts/scratch/model.pt --raster-kind compact-radius \
  --model-width 64 --blocks 16 --context-attention-blocks 1 --norm-groups 8 --epochs 8
.venv/bin/python -m vgo_training.export_onnx --checkpoint ../artifacts/scratch/model.pt \
  --output ../artifacts/scratch/model.onnx --maximum-batch 64 --packed-input
```

`maximum_batch` is frozen into each ONNX at export time. Two models can only meet
in an arena at the smaller of their two ceilings.

## Measuring strength

```bash
scripts/ratings.py artifacts/my-run/anchor.jsonl       # fit the loop's rating graph
scripts/bigboard-curve.sh                               # re-rate checkpoints at radius 1/38
scripts/bigboard-ratings.py artifacts/bigboard-curve/matches.jsonl
```

Ratings from different boards, rulesets or komi are different scales. Check the
komi before trusting any rating: `vgo-arena` defaults to 0.0, where Black wins
most games.

For a single match, call `target/release/vgo-arena` directly (after sourcing
`scripts/env/ort.sh`). Pass `--simulations`, `--resolution`,
`--policy-resolution`, `--max-plies`, `--komi` and `--candidate-raster-kind`
explicitly; the binary's defaults are not the loop's.

## Looking at games

```bash
scripts/selfplay-sgf.py artifacts/my-run/games/gen-000003-.../game-000123 out.sgf
scripts/resign-calibration.py artifacts/my-run/games --games 400
```

The SGF loads in the reference application: open
`reference/js-reference/voronoi_go.html` in a browser and paste it into the
VGO-SGF box.

To play the current model yourself:

```bash
./scripts/play.sh                 # newest model of any run
./scripts/play.sh path/to/model.onnx
```

## Traps

**Match processes by `/proc/PID/comm`, never by command line.** A `pgrep -f`
pattern matches the shell running it, and `pgrep -x` silently matches nothing
for names over 15 characters (`vgo-generate-continuous` is 23).
`stop-continuous.sh` records the details.

**Do not delete `artifacts/onnx-cache` between models.** The engine cache is
per model, but the timing cache beside it is shared and is the difference
between ~10.9 s and 0.3 s per model load. Clear it only after a driver or
TensorRT upgrade.

**Changing `--fp16`, the raster size or `--maximum-batch` rebuilds engines.**
The one-time stall is the cache working, not a regression.

**Replay memory scales with the policy resolution squared times the window.**
Packed states make the window 4.2x smaller than dense ones, but the policy
targets still dominate. Size the window against host RAM with a generator
resident.

**Per-batch rates need intervals.** A few hundred games swing several points by
chance. Three consecutive points are not a trend.

**A saturating activation in front of a squared error stops learning.** This
bit the value head (tanh + MSE) and the ownership head. Check output range and
gradient magnitude at initialization before adding a head.

**Draws mean the search is too shallow to separate moves.** Area scoring ties
only on a mirror-symmetric finish, so a nonzero draw rate is a signal.
