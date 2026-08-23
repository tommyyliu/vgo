#!/usr/bin/env bash
# Does more model capacity buy Elo, now that we train with Adam?
#
# Two results make this worth rerunning rather than trusting the existing shape:
#
#   - The search targets are far stronger than the network that generates them
#     (3200 sims beat 1600 by +147 Elo), while the policy head sits at
#     policy_kl ~0.70 with no overfitting even at twelve epochs. That is
#     underfitting, and capacity is the remaining suspect after the optimizer
#     sweep came back a null.
#   - `LearnerConfig`'s own comment records that "the architecture sweep that
#     chose w64 ran entirely under Muon". Adam then beat Muon by 77 Elo
#     (2.2 sigma) in runs/optimizer-ab.sh, so the width that won under Muon is
#     not necessarily the width that wins under Adam.
#
# Arms, all from scratch on the same 40 official-rules shards, Adam, wsd, 12
# epochs, seed 1 -- identical to the winning arm of the optimizer sweep except
# for width:
#
#   w64    8.2M params   the incumbent (already trained and rated at +38)
#   w96    ~18M
#   w128   ~33M
#
# Each arm is also benchmarked for inference throughput. That number decides
# whether a stronger model is usable: generation is the loop's bottleneck, so a
# model worth +100 Elo at half the speed can still be a net loss once it halves
# the playouts per game or the games per shard.
#
# VRAM is 16 GB. If w128 fails to fit at batch 256, lower VGO_BATCH rather than
# dropping the arm -- batch size changes optimization, so note it if you do.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-$root/artifacts-official/capacity-ab}"
stage="$root/artifacts-official/official-v2/window-ab/shards"
python="$root/training/.venv/bin/python"
batch="${VGO_BATCH:-256}"
shards=("$stage"/shard-*/dataset.vgo)
echo "training on ${#shards[@]} shards at batch $batch"

run_arm () {
  local name=$1 width=$2
  echo "=== arm $name: width $width ==="
  mkdir -p "$work/$name"
  "$python" "$root/scripts/train-once.py" "${shards[@]}" \
    --output "$work/$name/candidate.pt" \
    --optimizer adam \
    --raster-kind compact-dead-zone \
    --epochs 12 --seed 1 --batch-size "$batch" \
    --model-width "$width" --blocks 16 --context-attention-blocks 1 \
    --architecture ddrnet --value-weight 2.0 --learning-rate 0.001 \
    --schedule wsd --warmup-epochs 0 \
    2>&1 | tee "$work/$name/train.log" | grep -E "^epoch=|^train/val"
  (cd "$root/training" && "$python" -m vgo_training.export_onnx \
     --checkpoint "$work/$name/candidate.pt" \
     --output "$work/$name/final.onnx" --maximum-batch 32) | tail -1
  echo "--- inference throughput, $name ---"
  (cd "$root/training" && "$python" -m vgo_training.benchmark_model \
     --checkpoint "$work/$name/candidate.pt" --batches 32 --slots 2 \
     --iterations 300 2>&1 | tail -5) | tee "$work/$name/bench.log"
  echo "=== arm $name done ==="
}

run_arm w96  96
run_arm w128 128

# w64 is the optimizer sweep's Adam arm, already trained on these exact shards
# with these exact settings. It is re-rated here rather than retrained, both to
# save an hour and to give its +38 an independent second measurement.
source "$root/scripts/env/ort.sh"
cd "$root"
./target/release/vgo-arena \
  --candidate "$root/artifacts/raster-ab/compact-dead-zone/candidate.onnx" \
  --candidate-raster-kind compact-dead-zone \
  --opponent "$root/artifacts-official/optimizer-ab/adam/final.onnx" \
  --opponent "$work/w96/final.onnx" \
  --opponent "$work/w128/final.onnx" \
  --pairs 100 --simulations 256 --coarse-pool 16 --leaf-batch 4 --threads 8 \
  --max-plies 70 --resolution 128 --policy-resolution 128 \
  --radius 0.05555555555555555 --ruleset official --komi 0.104 \
  --seed 82000101
echo "CAPACITY AB DONE"
