#!/usr/bin/env bash
# Run the whole RL loop small, to prove a machine works.
#
# Every stage a real run depends on -- generation, training, ONNX export,
# TensorRT warmup, the promotion arena, shard retirement -- with the settings
# turned down until it finishes in minutes instead of hours. Passing means the
# box is ready; failing here costs a few minutes instead of discovering the same
# problem partway through an overnight job.
#
# Deliberately not a unit test: the failures worth catching are environmental
# (a missing dlopen, an unbuildable engine, a GPU too small for the batch) and
# only a real loop exercises them.
#
#   ./scripts/smoke.sh                 # tensorrt, the production path
#   PROVIDER=cuda ./scripts/smoke.sh   # skip the TensorRT engine build
#   KEEP=1 ./scripts/smoke.sh          # leave the output for inspection
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
output="${OUTPUT:-$root/artifacts/smoke-$(date +%Y%m%d-%H%M%S)}"
provider="${PROVIDER:-tensorrt}"

[[ -x training/.venv/bin/python3 ]] || { echo "no venv; run ./scripts/setup.sh first" >&2; exit 1; }

# A fresh clone has no artifacts/ -- it is gitignored -- so the log redirect
# below would fail before the run ever starts. The pipeline creates its own
# output directory but not its parent.
mkdir -p "$(dirname "$output")"

cleanup () {
  if [[ -z "${KEEP:-}" && -d "$output" ]]; then rm -rf "$output"; fi
}
trap cleanup EXIT

echo "smoke run -> $output (provider $provider)"
started=$(date +%s)

# Three updates, not two. Shard 0 always generates with the naive evaluator,
# and under --overlap-actor-learner shard 1 is generated concurrently with
# update 0's training -- so it is still naive. Only by shard 2 is a published
# model feeding generation, and generation-through-ONNX is the whole point:
# it is where a bad ORT_DYLIB_PATH, a missing TensorRT provider, or an
# unbuildable engine actually bites. A two-update smoke passes on a box where
# real generation would hang.
( cd training && exec .venv/bin/python3 -m vgo_training.rl_loop \
  --output "$output" \
  --updates 3 --samples-per-shard 64 --shards-per-update 1 --replay-window 2 \
  --resolution 128 --policy-resolution 128 --radius 0.055714285714285716 \
  --raster-kind compact --komi-low=-0.166 --komi-high=0.234 \
  --coarse-pool 16 --generation-simulations 32 \
  --temperature 1.0 --temperature-plies 10 --maximum-plies 30 \
  --training-epochs 1 --training-batch 32 \
  --learning-rate 0.001 --warm-learning-rate 0.001 --value-weight 2.0 \
  --ownership-weight 0.0 --recency-decay 1.0 \
  --schedule cosine --warmup-epochs 0 --no-compile --restore-optimizer \
  --architecture ddrnet --norm-groups 8 --model-width 64 --blocks 16 \
  --context-attention-blocks 1 --attention-heads 8 \
  --training-device cuda --report-every 1 --validation-fraction 0.2 \
  --actors 8 --arena-actors 8 --leaf-batch 4 \
  --inference-batch 16 --inference-delay-ms 1 --inference-slots 1 \
  --provider "$provider" --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --promotion-arena --promotion-score 0.55 \
  --arena-pairs 2 --arena-simulations 32 \
  --seed 424242 --arena-seed 424243 ) > "$output.log" 2>&1 || {
    echo >&2
    echo "FAILED. Last 30 lines of $output.log:" >&2
    tail -30 "$output.log" >&2
    exit 1
  }

# Assert on the artifacts, not the exit code. The loop can exit 0 having taken
# a shortcut, and each check below corresponds to an environment failure that
# is expensive to discover later.
training/.venv/bin/python3 - "$output" <<'PY' || exit 1
import json, re, sys
from pathlib import Path

run = Path(sys.argv[1])
failures = []

state = json.loads((run / "pipeline-state.json").read_text())
print(f"    updates completed : {state['updates_completed']}")
print(f"    shards generated  : {state['next_shard']}")
print(f"    models accepted   : {len([m for m in state['models'] if m.get('accepted')])}"
      f"  rejected: {len(state.get('rejected_models') or [])}")

published = sorted((run / "updates").glob("update-*/publication.json"))
if not published:
    failures.append("no update was published")
else:
    report = json.loads(published[-1].read_text())
    metrics = (report.get("training") or {}).get("final_validation") or {}
    print(f"    final value_mae   : {metrics.get('value_mae', float('nan')):.4f}")

# The one that matters: did generation ever run through the exported model?
# Everything else can pass on a box where the Rust side cannot dlopen ORT.
runtimes = {}
for command in sorted((run / "logs").glob("generate-*.command.json")):
    text = " ".join(json.loads(command.read_text())["command"])
    match = re.search(r"--runtime (\w+)", text)
    runtimes[command.name.split(".")[0]] = match.group(1) if match else "?"
print(f"    generation runtime: {', '.join(f'{k.split('-')[1]}={v}' for k, v in runtimes.items())}")
if "onnx" not in runtimes.values():
    failures.append("no shard generated through ONNX -- the inference path is untested")

if not list((run / "logs").glob("warmup-*.command.json")):
    failures.append("no TensorRT warmup ran")
if not list((run / "logs").glob("promotion-*.command.json")):
    failures.append("no promotion arena ran")

for failure in failures:
    print(f"    FAIL {failure}", file=sys.stderr)
sys.exit(1 if failures else 0)
PY

echo "    total             : $(( $(date +%s) - started ))s"
echo
echo "PASS -- generation, training, export, warmup and arena all ran."
[[ -n "${KEEP:-}" ]] && echo "kept: $output (log at $output.log)"
exit 0
