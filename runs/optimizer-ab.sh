#!/usr/bin/env bash
# Which optimizer trains the strongest model from a fixed set of shards?
#
# The loop's search targets are much stronger than the network that generates
# them -- 3200 simulations beat 1600 by +147 Elo -- while the policy head sits
# at policy_kl ~0.70 and shows no overfitting even at twelve epochs. A head that
# cannot reach its target while not overfitting is underfitting, which points at
# optimization or capacity rather than at data.
#
# The training curves point the same way. Warm-started bulk training held flat
# for eight epochs at lr 1e-3 -- 0.727, 0.729, 0.732, 0.727, 0.722, 0.721,
# 0.721, 0.723 -- and only improved once the schedule decayed. That is a model
# bouncing around a minimum it cannot settle into.
#
# ## The design
#
#   adam        Adam,   wsd 12 epochs      the current production optimizer
#   muon        Muon on the trunk + Adam on heads, wsd 12 epochs
#   ranger21    Ranger21, wsd 12 epochs    AdamW + lookahead + gradient
#                                          centralization + AGC + norm loss
#   adam-long   Adam,   wsd 24 epochs      the SCHEDULE control
#
# Arms 1-3 share one schedule so the comparison is about the optimizer. Arm 4
# changes only the schedule length, so if a longer decay is what the models
# actually want, this separates that from any optimizer effect -- Ranger21 in
# particular ships its own warmdown, and it is switched off here for exactly
# that reason.
#
# From scratch, not warm-started: every checkpoint on disk was trained by Adam
# and sits in a basin that would flatter it.
#
# ## What this cannot tell you
#
# One seed per arm, and a 200-game arena carries about +/-48 Elo. Differences
# under roughly 70 Elo will not be resolvable, so read this as a screen for a
# large effect, not as a ranking. Ranger21 is also a bundle of five techniques,
# so a win by it does not say which one did the work.
#
# There is also a standing result that Muon leads early in the RL loop and Adam
# overtakes by update 24, and that short offline A/Bs reproduce the misleading
# half. Whatever wins here is a supervised-training answer; treat its transfer
# to the loop as unproven.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-$root/artifacts-official/optimizer-ab}"
stage="$root/artifacts-official/official-v2/window-ab/shards"
python="$root/training/.venv/bin/python"

if [ ! -d "$stage" ]; then
  echo "staged shards missing: $stage" >&2; exit 1
fi
shards=("$stage"/shard-*/dataset.vgo)
echo "training on ${#shards[@]} shards"

run_arm () {
  local name=$1 opt=$2 epochs=$3
  echo "=== arm $name: --optimizer $opt, $epochs epochs ==="
  mkdir -p "$work/$name"
  "$python" "$root/scripts/train-once.py" "${shards[@]}" \
    --output "$work/$name/candidate.pt" \
    --optimizer "$opt" \
    --raster-kind compact-dead-zone \
    --epochs "$epochs" --seed 1 \
    --model-width 64 --blocks 16 --context-attention-blocks 1 \
    --architecture ddrnet --value-weight 2.0 --learning-rate 0.001 \
    --schedule wsd --warmup-epochs 0 \
    2>&1 | tee "$work/$name/train.log" | grep -E "^epoch=|^train/val"
  (cd "$root/training" && "$python" -m vgo_training.export_onnx \
     --checkpoint "$work/$name/candidate.pt" \
     --output "$work/$name/final.onnx" --maximum-batch 32) | tail -1
  echo "=== arm $name done ==="
}

run_arm adam      adam     12
run_arm muon      muon     12
run_arm ranger21  ranger21 12
run_arm adam-long adam     24

# Rated against the supervised seed, which is the reference every other Elo
# number in this investigation uses.
source "$root/scripts/env/ort.sh"
cd "$root"
./target/release/vgo-arena \
  --candidate "$root/artifacts/raster-ab/compact-dead-zone/candidate.onnx" \
  --candidate-raster-kind compact-dead-zone \
  --opponent "$work/adam/final.onnx" \
  --opponent "$work/muon/final.onnx" \
  --opponent "$work/ranger21/final.onnx" \
  --opponent "$work/adam-long/final.onnx" \
  --pairs 100 --simulations 256 --coarse-pool 16 --leaf-batch 4 --threads 8 \
  --max-plies 70 --resolution 128 --policy-resolution 128 \
  --radius 0.05555555555555555 --ruleset official --komi 0.104 \
  --seed 81000101
echo "OPTIMIZER AB DONE"
