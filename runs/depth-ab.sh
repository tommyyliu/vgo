#!/usr/bin/env bash
# Does a much larger network learn this data better?
#
# The RL loop has plateaued at w64/b16, and "make the model bigger" has been the
# standing suggestion with no evidence behind it. Every capacity test so far was
# run inside the loop, where a change has to survive generation, search and a
# moving opponent before it shows up -- which is exactly where a real effect can
# hide. This asks the narrower question on fixed data: given the same 40 shards,
# does 60M parameters fit them better than 8.2M, and does that turn into Elo?
#
# Round one took w128/b32 against w64/b16 and got a clean, useless answer: +70
# Elo at equal simulations, +9 at equal time. 7.3x the parameters bought real
# strength and then handed all of it back paying for itself.
#
# The exchange rate says why, and says which rungs to try instead. One doubling
# of simulations is worth ~61 Elo here (the small model went 256->512 and gained
# 70 - 9), so a model costing k times as much forgoes 61*log2(k):
#
#     w64/b32   1.20x TensorRT  ->  forgoes 16 Elo
#     w96/b16   1.17x           ->  forgoes 14
#     w96/b32   1.83x           ->  forgoes 53
#    w128/b32   2.22x           ->  forgoes 70   <- gained 70, netted 0
#
# The last line reproduces the measured +9, which is why the first two are worth
# a run: they need +14 to +16 Elo, against the +70 that 7.3x the parameters
# bought. Cost is from TensorRT at generation's batch of 32, not from PyTorch,
# which reads these as 1.53x and 1.90x -- the small model is launch-latency-bound
# there, so capacity is far cheaper than a training-time benchmark suggests.
#
# Both arms share seed, data, schedule, loss weights and batch size. Batch is
# 128 rather than the usual 256 because w128/b32 does not fit in 16 GB at 256
# (measured: 13.6 GB peak at 128), and a comparison where only one arm changed
# batch size would confound capacity with optimisation.
#
# 24 epochs, not the seed run's 12: that run reported best_epoch 12 of 12, so it
# was still improving when it stopped. Under-training penalises the larger model
# more, which would bias the answer toward the incumbent.
#
# Validation loss does not decide this. The noise run looked good on every
# training metric and lost 58 Elo on the board. The verdict is the arena at the
# end.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${VGO_OUTPUT:-$root/artifacts/depth-ab}"
epochs="${VGO_EPOCHS:-24}"
batch="${VGO_BATCH:-128}"
seed="${VGO_SEED:-90210}"

# vgo-wide's shards only. vgo-noise's 22 were generated under root noise, whose
# targets are measurably flatter (+0.13 nats of entropy), and this question is
# about capacity rather than about that.
shards=()
for shard in "$root"/artifacts/vgo-wide/replay/shard-*/dataset.vgo; do
  [ -e "$shard" ] && shards+=("$shard")
done
echo "training on ${#shards[@]} shards"

python="$root/training/.venv/bin/python"
mkdir -p "$output"

train() {
  local name="$1" width="$2" blocks="$3"
  local arm="$output/$name"
  if [ -e "$arm/candidate.onnx" ]; then
    echo "=== $name: already done, skipping ==="
    return
  fi
  mkdir -p "$arm"
  echo "=== $name: width $width, blocks $blocks ==="
  "$python" "$root/scripts/train-once.py" "${shards[@]}" \
    --output "$arm/candidate.pt" \
    --raster-kind compact-pass \
    --architecture ddrnet \
    --model-width "$width" --blocks "$blocks" \
    --context-attention-blocks 1 --attention-heads 8 \
    --norm-groups 8 \
    --epochs "$epochs" --batch-size "$batch" \
    --learning-rate 0.001 --schedule wsd --warmup-epochs 2 \
    --value-weight 2.0 --ownership-weight 0.0 \
    --validation-fraction 0.1 --full-adam --compile \
    --seed "$seed" --report-every 1 \
    2>&1 | tee "$arm/train.log"
  ( cd "$root/training" && "$python" -m vgo_training.export_onnx \
      --checkpoint "$arm/candidate.pt" --output "$arm/candidate.onnx" \
      --maximum-batch 64 ) 2>&1 | tail -3
}

train small 64 16
train large 128 32
train deep 64 32
train wide 96 16
echo "arms trained; rate them with scripts/depth-ab-arena.sh"
