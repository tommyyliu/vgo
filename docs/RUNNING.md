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

### Continuous-run performance

The generator appends `performance.jsonl` inside each generation directory.
Every launch writes a `run` record with its PID, model digest, source revision,
hardware, and resolved configuration. `interval` records arrive every 30 seconds
by default, and a `final` record is written after actors finish. A restart
appends a new run ID. Set `VGO_TELEMETRY_SECONDS` in the bulk loop, or pass
`--telemetry-seconds` directly; zero disables periodic reporting while retaining
startup and final records.

Neural startup records also identify the actual input layout and effective batch
ceiling reported by the backend, plus the GPU description when available. Older
ONNX sidecars incorrectly advertised packed graphs as a single dense input;
the graph metadata and backend contract are authoritative. New exports describe
their actual graph inputs in the `inputs` array and declare `input_layout`.

The records distinguish searched plies (including games in flight) from games
and samples successfully published. They report capped and soft-resigned games,
search and write durations, actor activity, and Linux RSS/peak RSS. Per-board
counters group radii by rounded `1/r`; search buckets further divide each game's
ply cap into opening/middle/late thirds and separate soft-resigned searches.
These stage names describe progress through the configured cap, not settlement.

Neural runs also include interval and cumulative broker metrics: average batch
size/fill, encoding, queue, input packing, native session execution, output
extraction, and dispatch/wait counters. Search duration includes waits for
inference. Parallel wall-time sums overlap, and queue durations are weighted
by positions, so these fields must not be added into CPU-time percentages.
Memory fields are null on systems without Linux `/proc`.

The loop appends timestamped training, export, generator-start, and drain events
to `events.jsonl`. Join their `unix_seconds` and generator PIDs with performance
records to identify training contention and overlapping model generations.
Short `[progress]` summaries also appear in `generate.log`.

For CPU-only checks, use the naive evaluator and avoid CUDA initialization:

```bash
target/release/vgo-generate-continuous --output-root /tmp/vgo-check --label cpu \
  --blocking-sync false --actors 2 --simulations 32 --maximum-games 4 \
  --telemetry-seconds 1
```

This checks generation/accounting; it does not measure neural self-play speed.

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

`--norms-per-block 1` trains a residual trunk with one GroupNorm per convolutional
block; the default remains two. The count is stored in checkpoints and exports.
Checkpoints without this field retain the original two-norm architecture.
Warm starts inherit the checkpoint's architecture. For the bulk loop, set
`VGO_NORMS_PER_BLOCK`; compare fresh training runs before changing a campaign.

### Randomized search-budget experiment

The continuous generator accepts `--full-search-fraction 0.25
--fast-simulations 200` with, for example, `--simulations 1000`. Every game's
first turn is full; later turns choose a seeded full/fast budget. Fast turns
disable root noise and move temperature. Soft resignation can reduce either
budget further. Defaults keep every turn full, preserving the existing recipe.

Only full-selected turns enter replay: **both policy and value training use
these rows with their normal weights**. Fast turns still determine the final
outcome and contribute to resignation calibration. `games.jsonl` contains
per-ply `search_budgets` and `policy_eligible` arrays; join replay rows by ply.
Telemetry distinguishes searched plies from policy-eligible plies and actually
published samples. Budget and eligibility are different after soft resignation.

The bulk-loop equivalents are `VGO_FULL_SEARCH_FRACTION` and
`VGO_FAST_SIMULATIONS`. The initial naive generation stays full-budget. These
are experiment controls, not a recommendation to adopt the mixed recipe.

For a bounded generation slice use `--run-seconds 600`. Its clock starts after
model sessions are ready, requires no profiling feature, and discards incomplete
games at the deadline. Actors finish the search in flight before checking
cancellation; final exit can overrun by one search per actor.
`--profile-seconds` requires `--profile-output`; without it the generator now
rejects the command instead of silently ignoring the deadline. When both limits
are specified, the earlier positive deadline wins.

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

For equal thinking time add `--move-time-ms 200 --simulations 10000`, applying
the same advisory time budget to both seats with a simulation ceiling. Search
completes root evaluation and at least one round; a pending broker round can
overrun the deadline. Reports include each seat's actual average simulations,
average search milliseconds, and longest search. Keep threads and batching
fixed between experiments; queue contention is part of this measurement.

The bounded restart recipes are in
[`scripts/experiments/restart-2026-10-14.py`](../scripts/experiments/restart-2026-10-14.py).
Run its stages sequentially with an otherwise idle CPU/GPU; it records commands,
binary digests, process exit codes and elapsed times alongside raw reports.

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

## Performance history

`scripts/bench.py` times each large CPU path single-threaded on the fixed real
positions in `benchmarks/positions.txt` (16 at 1/38, 16 at 1/18): move
application, board analysis, the legal-set index, the settled plane and packed
raster at 128 and 256, the fine grid, and a 200-simulation search with a fixed
policy standing in for the network. It compares with the last record from the
same machine and appends to `benchmarks/history.jsonl`, which is committed.

```bash
scripts/bench.py              # run, compare, record (needs a clean tree)
scripts/bench.py --no-record  # try a change before committing it
scripts/bench.py --check      # exit 1 on a regression
scripts/bench.py --show       # the history for this machine
```

Run it before merging anything that touches rules, geometry, the raster or
search, and commit the new record with the change. A path is flagged only when
it is more than 10% slower *and* slower by more than the two runs' combined
spread. It refuses to run above load 2, because a running generator inflates
every number. Records from different machines are never compared.

## Arena tree reuse

`vgo-arena --tree-reuse` retains the candidate's tree; use
`--opponent-tree-reuse` independently for the opponent. With the same model on
both seats and `--move-time-ms 200`, this measures the effect of keeping work
at an advisory elapsed-time budget. Root exploration noise must be zero on a
seat using reuse. Existing visits influence selection, and `--simulations`
limits new work per move. Intervening reroot/drop time is charged to the next
move's budget; one completed round remains guaranteed.

`--reuse-maximum-nodes 4096` and `--reuse-maximum-mib 256` bound accepted cache
structure and tracked payload per seat. Payload excludes allocator metadata
and hidden capacity, so consult process RSS too. Over-limit trees are dropped.
Arena JSON includes per-seat hit counts, inherited visits, limit drops, maximum
retained payload and reroot time. Self-play and the browser do not enable reuse.

## Native inference deadlines

All native entry points honor these positive integer environment settings:

| Setting | Default | Scope |
| --- | ---: | --- |
| `VGO_INFERENCE_STARTUP_SECONDS` | 1800 | Each ONNX session's initialization and contract validation |
| `VGO_INFERENCE_TIMEOUT_SECONDS` | 120 | Each dispatched backend batch, independently of other slots |

For a cached model during diagnostics, for example:

```bash
export VGO_INFERENCE_STARTUP_SECONDS=120
export VGO_INFERENCE_TIMEOUT_SECONDS=30
```

Cold TensorRT construction can take minutes; choose its startup allowance
accordingly. Loading logs name the model/backend and report waiting every
30 seconds until readiness. A batch timeout names its slot, sequence, request
IDs and elapsed time; queued callers fail when the broker stops.

Timeouts are fatal to the owning process. A native call cannot be interrupted
safely, so its worker is detached and process exit releases native resources.
Restart in a fresh process. The move server sends an error response and exits;
generation and arena failures return nonzero. Healthy shutdown joins workers.
These deadlines are separate from the generator's post-readiness `--run-seconds`.
Native failures also arm a final three-second watchdog because native exit
handlers can hang after the error is reported. Linux then exits with status 70
without running those handlers; other targets abort. In-flight games can be
abandoned, while published replay remains atomic.

Calibration rows now carry `minimum_ply`; `resign-calibration.py --minimum-ply 20`
filters for the live default floor. The bulk loop passes its configured floor.
Legacy rows describe a counterfactual floor of zero and are excluded from
floor-20 threshold selection until matching new data is available.

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
