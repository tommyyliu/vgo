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

There are three suites and no single command that runs them all. Run the first
two before any change lands; they are fast.

```bash
cargo test --release --workspace          # 21 suites: rules, search, self-play
cd training && .venv/bin/python3 -m unittest discover -s tests   # 88 tests
```

```bash
reference/tests/run-tests.ps1             # engine, game tree, UI fixtures in Chrome
```

The browser suite needs Chrome and PowerShell; it is the only part of the tree
that assumes Windows. The Rust and Python suites are what catch the failures
that actually happen -- a shape or tuple-arity change in the learner shows up
there rather than at runtime, and one such change would otherwise have crashed a
run at its first validation pass.

## The reinforcement-learning loop

[`RL_LOOP.md`](RL_LOOP.md) describes the queue, artifact, and recovery
contracts.

Every run this project has kept is launched from a committed `launch.sh` beside
its output, not from a command typed at a prompt. That file is the record of
what a run *was* -- its settings are learning identity, so a run cannot be
resumed with different ones, and the header comment explains why each value was
chosen. Copy the most recent one and edit it:

```bash
cp artifacts/ddrnet-own/launch.sh artifacts/my-run/launch.sh
$EDITOR artifacts/my-run/launch.sh          # --output, --seed, and the change under test
./artifacts/my-run/launch.sh > artifacts/my-run/logs/run.log 2>&1
```

Run it detached if you want it to survive the shell:

```bash
setsid nohup ./artifacts/my-run/launch.sh > artifacts/my-run/logs/run.log 2>&1 &
```

The current shape of a run, as of `ddrnet-own`:

```bash
.venv/bin/python3 -m vgo_training.rl_loop \
  --output "$root/artifacts/NAME" \
  --updates 60 --samples-per-shard 6144 --shards-per-update 1 --replay-window 6 \
  --resolution 128 --policy-resolution 128 --radius 0.055714285714285716 \
  --raster-kind compact --komi-low=-0.166 --komi-high=0.234 \
  --coarse-pool 16 --generation-simulations 512 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 64 \
  --resign-window 5 --resign-minimum-ply 20 --resign-disable-fraction 0.0 \
  --training-epochs 1 --training-batch 256 \
  --learning-rate 0.001 --warm-learning-rate 0.001 --value-weight 2.0 \
  --schedule wsd --warmup-epochs 1 --compile --restore-optimizer \
  --architecture ddrnet --norm-groups 8 --model-width 96 --blocks 16 \
  --training-device cuda --report-every 1 --validation-fraction 0.1 \
  --actors 64 --arena-actors 64 --leaf-batch 4 \
  --inference-batch 64 --inference-delay-ms 1 --inference-slots 1 \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --seed 8900001 --arena-seed 8905001
```

Note `--komi-low=-0.1` uses the `=` form: clap reads a bare `-0.1` as a flag.

**One epoch, not ten.** A shard is seen once per update and survives
`--replay-window` updates, so each sample is seen six times, by six different
models. That is far from memorisation: measured over 35 updates the training MAE
never fell below 0.109, where ten epochs on fixed shards reached 0.018. Ten
epochs also spent 62% of the cycle on training; one spends about 6%, which
leaves the GPU to self-play. The learning-rate schedule advances per optimizer
step so the full warmup-stable-decay curve still fits inside a single epoch.

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

```bash
  --initial-checkpoint ../artifacts/PREV/updates/update-000019/candidate.pt \
  --initial-onnx ../artifacts/PREV/updates/update-000019/candidate.onnx \
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

```bash
training/.venv/bin/python3 -m vgo_training.rl_loop \
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

```bash
cargo run --release -p vgo-selfplay --bin vgo-arena -- \
  --candidate artifacts/A/updates/update-000034/candidate.onnx \
  --opponent  artifacts/B/updates/update-000034/candidate.onnx \
  --candidate-raster-kind compact \
  --pairs 16 --simulations 512 --max-plies 120 --threads 32 \
  --resolution 128 --policy-resolution 128 --coarse-pool 16 \
  --radius 0.055714285714285716 --komi 0.034 \
  --maximum-batch 64 --provider tensorrt --device-id 0 \
  --cache-directory artifacts/onnx-cache
```

Omit `--opponent` to play the naive policy. Games are colour-swapped in pairs,
so `--pairs 75` is 150 games. Set `--threads` near `--pairs`: throughput scales
with concurrent games, not threads, because MCTS keeps only one evaluation in
flight per game. Arena verdicts were measured identical from 1 to 48 threads, so
parallel arenas are promotion-grade; see the note in `docs/RL_LOOP.md`.

Run it through `runtime_environment()` if you invoke it from Python, or from a
shell where the TensorRT libraries are already on `PATH`. See the traps below.

### Rating a lineage

`scripts/rate-checkpoints.py` plays checkpoints against each other and fits
Bradley-Terry ratings. It runs outside the pipeline -- no state file, no queue
-- so it works on a finished run, an abandoned one, or one that is merely idle.

```bash
./scripts/rate-checkpoints.py artifacts/ddrnet-komi3 --lineage \
  --versions 0,14,28,35,41,47 --pairs 8
```

`--lineage` follows `--initial-checkpoint` back through launch scripts and puts
every ancestor on one version timeline. A warm-started run's version 0 is
already the product of its parent's training, so rating it alone reads the tail
of a history as though it were the whole thing.

Round robin is the default, and matters more than it sounds. Rating everything
against one anchor stops discriminating the moment the field beats it: measured
on one lineage, four checkpoints scored 0.938 to 1.000 against the anchor and
were indistinguishable, and an undefeated record has no finite Elo at all -- the
fit falls back on the prior. Cross-play keeps every rating pinned by games that
were actually close. `--no-round-robin` restores anchor-only, which is `n-1`
matches rather than `n(n-1)/2`.

The naive evaluator joins the field by default. It is the only opponent whose
strength never changes, so it is what makes ratings comparable across runs; an
Elo measured only against siblings says where a model sits in its own field and
nothing more.

Matches are played at the midpoint of the run's own komi range, read from its
config. Rating at komi 0 measures a model on a game it never trained for.

### Where does komi balance the game?

```bash
training/.venv/bin/python3 scripts/komi-balance-fit.py artifacts/ddrnet-komi3 \
  --komi-low=-0.166 --komi-high=0.234 --last 6
```

Fits `P(Black wins) = sigmoid(a + b*komi)` over played-out games and reports the
crossing with a bootstrap interval. **This does not stay put.** On one lineage
the balance point moved from +0.163 to +0.034 over about forty updates as the
model strengthened, and nothing in the loop measures it -- the drift was found
only while investigating something else. Refit every ten updates or so.

`--last N` restricts the fit to recent shards, which is necessary once a run's
play changes character. Resigned games are excluded automatically: under
resignation the mover concedes, so the winner is fixed by ply parity and carries
no information about balance. Pooling them in once produced an apparent 86%
Black where the played-out subset said 62%.

The symptom to watch for between fits is Black's share per shard. Anything
outside 40-60% means the range has drifted, and a one-sided share makes
`value_sign_accuracy` meaningless -- predicting the majority class scores
whatever that class holds. At 18% Black an accuracy of 0.970 was worth +0.15
over that baseline; at 49% Black, 0.911 was worth +0.40.

### Reading a specific position

Two diagnostics answer questions the aggregate metrics cannot.

```bash
# Was the refutation in the tree, or did the value head just miss it?
cargo run --release -p vgo-selfplay --example probe_capture -- \
  position.json 0.371 0.074 artifacts/RUN/updates/update-000030/candidate.onnx 2048

# Does the raster the net reads match the position the engine computes?
cargo run --release -p vgo-selfplay --example raster_fidelity -- game.sgf 128
```

`probe_capture` distinguishes a candidate-generation failure from a value-head
failure, which call for opposite fixes. On the game that prompted it the
refutation *was* proposed -- it drew 1 visit of 2339 and was evaluated at
+1.0000 while capturing 21 stones -- which is what identified the head rather
than the search.

`raster_fidelity` checks the settled channel against its closed-form region and
reports how far stones land from the policy lattice. Snapping pushes placements
off the lattice by design, including moves the engine plays itself, so off-grid
is normal; what matters is that the offset stays under half a cell.

```bash
# Were the resignations right?
cargo run --release -p vgo-selfplay --example review_resignations -- \
  artifacts/RUN/replay/shard-000020 20
```

Writes SGFs to `diagnostics/resignation-review/`, named by whether the conceding
side was ahead or behind on the board.

### Is search working at all?

`vgo-arena` drives both seats at one `--simulations`, so it cannot tell you
whether search depth is buying anything. `vgo-playout-duel` holds the model
fixed and varies only the playout budget, which isolates the search path from
policy quality:

```bash
cargo run --release -p vgo-selfplay --bin vgo-playout-duel -- \
  --model artifacts/A/updates/update-000019/candidate.onnx \
  --high 128 --low 16 --pairs 40 --coarse-pool 8 \
  --max-plies 160 --threads 32 --resolution 96 --policy-resolution 32 \
  --radius 0.05555555555555555 --maximum-batch 64 \
  --provider tensorrt --cache-directory artifacts/onnx-cache
```

The deep seat should win comfortably. If it does not, the fault is in search
rather than in the policy, because both seats share one evaluator. Pass
`--coarse-pool 0` to run the same comparison on the legacy candidate path;
running both is how you tell a coarse-sampling regression from a general search
regression. Use the same `--radius` and `--resolution` the model trained at, or
the net sees out-of-distribution rasters and inference returns NaN.

## Benchmarks

```bash
cargo run --release -p vgo-selfplay --bin vgo-canary -- \
  --pairs 100 --first 1000 --second 10 --max-plies 48 --threads 4 --seed 10001

cargo run --release -p vgo-selfplay --bin vgo-model-smoke -- \
  --resolution 128 --policy-resolution 32
```

```bash
cd training
.venv/bin/python3 -m vgo_training.benchmark_model \
  --checkpoint ../artifacts/raster-demo/model.pt --batches 1,8,16,32,64
training/.venv/bin/python3 -m vgo_training.benchmark_precision \
  --dataset ../artifacts/raster-demo/dataset.vgo \
  --checkpoint ../artifacts/raster-demo/model.pt
```

## The reference application

Open `reference/js-reference/voronoi_go.html` directly in a browser; it has no
build step and loads the engine modules from `reference/src/`. Paste a position
or a game record into the VGO-SGF box and press Load.

## Traps

These are all things that fail confusingly rather than obviously.

**Do not wait on a process with `pgrep -f`.** A monitor or launcher whose own
command line contains the pattern matches itself, and two of them match each
other. That cost a full night: an A/B finished at 02:20, the launcher waiting on
`pgrep -f ownership_effect.py` never saw it exit because two monitors containing
that string were still alive, and the machine sat idle until morning. The same
class of mistake makes `pkill -f rl_loop` kill the shell issuing it. Match on a
PID file, or on something no tool of yours will ever contain.

**Alert thresholds need calibrating against both a healthy and a failing run.**
An absolute train/val gap threshold tightens silently as a model improves --
`gap > 0.06` is nothing at a validation MAE of 1.0 and everything at 0.15, and
it fired on four of six healthy updates. Relative thresholds need the same care:
45% sits *inside* the healthy band, which measured 8-48% with a median of 31%,
where genuine memorisation ran at 90-92%. Put the threshold between the two
regimes, not where it feels strict.

**A shard is ~105 games, so per-shard rates swing several points by chance.**
Three or four consecutive points are not a trend. In one session that produced
three false alarms -- a "doubling" capped rate that reversed, a "monotone" ply
increase that reversed, and a stalling signal that came from a field counting
captured *stones* rather than no-op *moves*. Compute the interval, or pool
shards, before reporting movement.

**Game ids restart in every shard.** Across twenty shards, 2198 games carry only
148 distinct ids. Any analysis that groups by `game_id` alone silently merges
unrelated games; key on `(shard, game)`. The learner's own split avoids this by
salting its hash with the shard digest.

**`load_datasets` expands policy targets to full width.** A record stores 64
touched cells; the loader materialises four tensors of `policy_size` (16385),
which for twenty shards is 30 GB of ~99.6% zeros against 18.8 GB of rasters. Use
`dataset.sparse.expand(rows)` per batch and drop the dense copies if you are
loading many shards outside the pipeline.

**A saturating activation in front of a squared error stops learning.** This has
now bitten three times: `tanh` + MSE on the value head, `tanh` on the ownership
head (which read -1.0000 to -0.9993 at initialisation, gradients 1e-6), and MSE
against bounded +/-1 targets, which keeps pulling correct cells long after their
sign is settled. Before adding any head, check its output range and gradient
magnitude at initialisation rather than after a run disappoints.

**Batch 256 does not fit a 15.5 GiB card at 128x128.** Both a w96/b32 and a
w48/b16 net OOM during backward. Batch 128 peaks at 7.4 and 8.6 GB. The failure
arrives as `CUDA out of memory` deep in `_engine_run_backward`, which reads like
a leak rather than a sizing problem.

**TensorRT libraries are not on `LD_LIBRARY_PATH`.** They ship inside the `uv`
virtual environment, not on the system. `rl_loop.runtime_environment()` prepends
`onnxruntime_trt`, `tensorrt_libs`, `nvidia/cu13/lib`, `nvidia/cudnn/lib`, and
`torch/lib`, and sets `ORT_DYLIB_PATH`, for every child process it spawns.
Anything launched through the loop is already correct. Invoking `vgo-arena` or
`vgo-generate-demo` yourself from a bare shell fails to load
`libonnxruntime_providers_tensorrt.so` because `libnvinfer.so.10` is not
resolvable. Import the helper rather than reimplementing the path logic.

Two ONNX Runtime builds are installed: `onnxruntime_trt` and
`onnxruntime_blackwell`. The helper prefers `onnxruntime_trt` and only falls
back to the other if it is absent, which matters when reading a stack trace and
wondering which one is loaded.

Provider registration is not silent about failure. `OnnxService` calls
`.error_on_failure()` on the TensorRT provider, so a missing library aborts
rather than quietly degrading to the CUDA provider listed after it. A run that
starts is a run that got the provider it asked for.

**Do not delete `artifacts/onnx-cache` between iterations.** It holds two caches
with different lifetimes. The engine cache is keyed on the model digest, so every
trained model rebuilds it -- that is expected. The timing cache is deliberately
hoisted one directory level above it: it records how fast each kernel tactic runs
for a given layer shape on this device, which does not change across RL
iterations because only the weights change. Sharing those measurements is the
difference between roughly 10.9 s and 0.3 s per model load. Clearing the
directory is only correct after a driver or TensorRT upgrade, when the recorded
tactic timings no longer describe the hardware.

Both cache paths are keyed on provider, precision, raster width and height, and
`maximum_batch`; only the engine cache adds the model digest. So changing
`--fp16`, the raster size, or `--maximum-batch` mints a fresh engine *and* a
fresh timing cache, and re-pays the cold build once. That one-time stall is the
cache working, not a regression.

**`maximum_batch` is frozen into each ONNX at export time.** A model exported at
16 cannot be driven at 64, and the error is
`configured batch 64 exceeds ONNX maximum 16`. Two models can only meet at the
smaller of their two ceilings, which silently limits both generation throughput
and cross-generation comparison. Re-export with
`training/.venv/bin/python3 -m vgo_training.export_onnx --checkpoint X --output Y
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
