# Reinforcement-learning loop

The production loop keeps game rules and self-play in Rust and uses Python only
for optimization and ONNX export:

1. `vgo-generate-demo` runs parallel MCTS actors and writes a checksummed,
   versioned replay shard.
2. `vgo_training.train_demo` trains a policy/value network over a replay window.
3. `vgo_training.export_onnx` atomically publishes a self-describing ONNX model.
4. `vgo-arena` plays color-swapped games against the naive policy and the
   incumbent model.
5. `vgo_training.rl_loop` promotes the candidate only when it reaches the
   configured incumbent score without exceeding the arena truncation bound.

The driver writes `progress.json` after every stage. Replay generation uses a
staging directory, while checkpoint, report, and ONNX publication use atomic
file replacement. Rerunning an interrupted output directory resumes at the
first incomplete stage. A directory containing `run.json` is final and is not
overwritten. `run-config.json` prevents a resume with different parameters.

## Run

From `training`:

```powershell
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run `
  --iterations 4 --samples 768 --replay-window 4 `
  --resolution 128 --coarse-pool 8 --generation-simulations 128 `
  --epochs 50 --training-batch 16 --device cuda `
  --actors 16 --arena-actors 1 --arena-pairs 40 `
  --maximum-batch 16 --provider tensorrt `
  --promotion-score 0.52 --maximum-truncation-rate 0.02
```

Generation uses many actors to fill inference batches. Promotion deliberately
defaults to one actor. Concurrent FP16 requests can produce different batch
shapes, and small numerical changes can change an action when logits are nearly
tied. Single-actor arenas issue batch-one requests in a stable order. Use more
arena actors for throughput measurements, not promotion decisions.

The first iteration bootstraps from the deterministic naive evaluator. Later
iterations generate with the incumbent, warm-start training, and retain recent
replay. Pass both `--initial-checkpoint` and `--initial-onnx` to continue from a
published model; `--initial-replay` adds existing shards to the replay window.
`--coarse-pool 8` enables coarse-to-fine policy sampling for ONNX self-play and
arenas; its default of `0` preserves the legacy candidate sequence. The pool is
the number of fine cells per coarse region and cannot exceed `--resolution`.
The loop forwards the same pool to replay generation, baseline and promotion
arenas, and optional Elo matches.

The bootstrap generator has no spatial policy grid, so a run without an initial
model necessarily produces legacy replay in iteration zero even when
`--coarse-pool` is positive. Coarse-to-fine generation begins after a model is
accepted, or immediately when both `--initial-checkpoint` and `--initial-onnx`
are supplied. ONNX candidates in the iteration-zero arena can still use the
coarse path.

Spatial search retains the standard cumulative visit-count widening budget
`min(96, max(4, ceil(2 * sqrt(N + 1))))`. Each widening call draws only the IID
delta needed to reach that budget. Duplicate cells increase their proposal
count rather than being retried, while pass is enumerated deterministically.
Replay v3 stores those `u32` counts after visits and beta; the loader remains
compatible with replay v1 and v2. Training prepares the self-normalized sparse
target and full-legal mask once on CPU, then reuses them across epochs and
metrics. See [`POLICY_REDESIGN.md`](POLICY_REDESIGN.md) for the correction's
mathematical scope.

## Artifacts

Each `iteration-NNN` directory contains:

- `replay/dataset.vgo` and `replay/manifest.json`: immutable training data and
  provenance;
- `replay/images`: RGB overviews and one image per semantic channel;
- `model/candidate.pt` and `candidate.pt.json`: checkpoint and training report;
- `model/candidate.onnx` and `candidate.onnx.json`: deployable model and digest;
- stage logs, `progress.json`, and the final `iteration.json` decision.

`run.json` names the final incumbent. Training loss is diagnostic because the
current split is sample-level. Fresh arena games are the promotion signal.

## Current coarse-to-fine integration result

The retained 2026-07-24 CPU smoke exercised the complete final path from an
existing ONNX incumbent: 4 replay-v3 samples at 128x128, 16 generation
simulations, one training epoch, ONNX export, and a two-game promotion arena.
Every replay row contained the expected eight cumulative proposal draws and 16
visits; all corrected targets normalized, and the full-legal denominators were
substantially larger than the nine explored actions. This is an integration
check, not a playing-strength result. The full audit is
[`../benchmarks/results/2026-07-24-coarse-policy-smoke.json`](../benchmarks/results/2026-07-24-coarse-policy-smoke.json).

## Historical legacy-path result

This 2026-07-22 measurement predates the final coarse-to-fine replay-v3 design;
it remains useful only as a baseline for the surrounding RL-loop machinery. It
trained at 128x128 over 1,920 positions. Held-out policy KL fell from `0.5881`
to `0.4944`, and value MAE fell from `0.3599` to `0.3247`. On 120 fixed-seed,
single-actor games against the same naive policy:

| Model | Wins | Losses | Draws | Score | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial accepted model | 73 | 6 | 41 | 77.9% | 69.7–84.4% |
| Refined model | 115 | 4 | 0 | 96.6% | 91.7–98.7% |

In a direct 80-game single-actor match, the refined model beat the initial
model `44–36` for a 55.0% score. That result is directionally positive, but its
95% interval (`44.1–65.4%`) still includes an even match; the common-baseline
comparison is the stronger evidence from this run.

The commands and raw measurements are retained in
[`benchmarks/results/2026-07-22-rl-loop.json`](../benchmarks/results/2026-07-22-rl-loop.json).
