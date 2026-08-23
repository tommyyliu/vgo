#!/usr/bin/env bash
# Does a replay window's *generational spread* matter, or only its size?
#
# official-v3 declined 88 Elo (2.5 sigma) between updates 13 and 27 against its
# own starting model, and the decline tracked what the window held:
#
#     update 13   18 seeded official-v2 shards + 14 v3-generated   ->  +21 vs w32
#     update 27    4 seeded official-v2 shards + 28 v3-generated   ->  -67 vs w32
#
# The seeded shards spanned 40 different models across the whole of official-v2.
# v3's own shards come from 28 models that are nearly identical. Both windows
# held 32 shards, so if spread is what matters, size was never the whole story --
# and that would also explain why the offline replay, which always drew on v2's
# forty generations, measured a bigger gain than the live loop realises.
#
# Both arms: same starting checkpoint, same shard COUNT, same update count, same
# everything else. Only the generational spread of the window differs.
#
#   spread   16 shards taken every other one across official-v2 0..30
#            -> 16 distinct models spanning 30 generations
#   narrow   16 consecutive shards from official-v2 24..39
#            -> 16 distinct models spanning 16 generations
#
# ## The honest limitation
#
# This is a 2x contrast in spread, not the 40-vs-1 the live loop suggests, because
# every shard on disk comes from a *different* model -- nothing here was generated
# by one model repeatedly. A null result therefore does not clear the hypothesis;
# only a positive one is informative. Testing it properly means generating several
# shards from a single frozen model, which costs a night of GPU.
#
# Needs ~10 GB for a 16-shard window. Do not run it beside a live loop; the
# learner holds its window resident for the whole run and the box has 60 GB.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-$root/artifacts-official/official-v2/window-diversity}"
stage="$root/artifacts-official/official-v2/window-ab/shards"
python="$root/training/.venv/bin/python"

if [ ! -d "$stage" ]; then
  echo "staged shards missing: $stage" >&2
  echo "re-stage them the way runs/official-v3.sh does, or point at another copy" >&2
  exit 1
fi

spread=""; for i in $(seq 0 2 30); do spread="$spread $(printf '%d' "$i")"; done
narrow=""; for i in $(seq 24 39); do narrow="$narrow $(printf '%d' "$i")"; done

run_arm () {
  local name=$1; shift
  local shards=("$@")
  echo "=== arm $name: ${#shards[@]} shards -> ${shards[0]}..${shards[-1]} ==="
  local paths=()
  for i in "${shards[@]}"; do paths+=("$(printf '%s/shard-%06d/dataset.vgo' "$stage" "$i")"); done
  "$python" "$root/scripts/train-once.py" "${paths[@]}" \
    --output "$work/$name/candidate.pt" \
    --initial-checkpoint "$root/artifacts-official/official-v2/updates/update-000031/candidate.pt" \
    --raster-kind compact-dead-zone \
    --epochs 8 --seed 1 --model-width 64 --blocks 16 \
    --context-attention-blocks 1 --architecture ddrnet \
    --value-weight 2.0 --learning-rate 0.001 --full-adam \
    --schedule cosine --warmup-epochs 0
  (cd "$root/training" && "$python" -m vgo_training.export_onnx \
     --checkpoint "$work/$name/candidate.pt" \
     --output "$work/$name/final.onnx" --maximum-batch 32) | tail -1
}

mkdir -p "$work"
run_arm spread $spread
run_arm narrow $narrow

source "$root/scripts/env/ort.sh"
cd "$root"
./target/release/vgo-arena \
  --candidate "$work/spread/final.onnx" \
  --candidate-raster-kind compact-dead-zone \
  --opponent "$work/narrow/final.onnx" \
  --opponent "$root/artifacts/raster-ab/compact-dead-zone/candidate.onnx" \
  --pairs 100 --simulations 256 --coarse-pool 16 --leaf-batch 4 --threads 8 \
  --max-plies 70 --resolution 128 --policy-resolution 128 \
  --radius 0.05555555555555555 --ruleset official --komi 0.104 \
  --seed 80000101
echo "DIVERSITY AB DONE"
