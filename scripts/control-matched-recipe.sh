#!/usr/bin/env bash
# Control: is the bulk loop's self-play data any good?
#
# The first attempt at this run exposed why the loop looked flat. Generation
# labels restart at `gen-000000` every run, so `window_from_games` -- which took
# recency from directory order -- put the bulk run's output *before* the
# continuous run's. Updates 26, 27 and 28 each trained on the identical 759
# stale games from continuous gens 5-12 while ~2,000 fresh ones sat unread, and
# every symptom followed: the sample count was byte-identical across rounds
# (150,297), validation was flat to four digits (1.6702/1.6723/1.6713), and the
# models measured -66 Elo pooled against sl-w64b16 because they were trained on
# 150k of that anchor's own 200k for twice the epochs.
#
# `window_from_games` now orders by game number, which the generator assigns
# globally. So this is the first training run ever to see the loop's own games.
#
# It holds the recipe at the anchor's values except for window size, and swaps
# the data, which is the comparison originally intended.
#
# The window is 150k, not the anchor's 200k. A 200k window peaks around 46 GB of
# 60 and needs the box to itself: launched against a live desktop it filled all
# 7 GB of swap while still loading, at 2% GPU, and had to be killed. 150k costs
# ~25 GB and fits. That leaves data volume as a difference from the anchor, so a
# NEGATIVE result here keeps volume as a confound and wants the exact 200k rerun
# on an idle machine; a POSITIVE result settles it either way. Everything
# else already matched and is repeated verbatim: batch 64, lr 5e-4, wsd, warmup
# 0, Adam (--full-adam; the LearnerConfig default is Muon), bfloat16, --compile,
# value-weight 2.0, validation-fraction 0.1.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/control-matched-recipe"
python="$root/training/.venv/bin/python"
source "$root/scripts/env/ort.sh"
mkdir -p "$out/sgf"

echo "===== control start $(date '+%F %T') ====="

echo "[control] training: ${VGO_WINDOW:-150000} samples, 4 epochs"
( cd "$root/training" && "$python" "$root/scripts/train-once.py" \
    --games-root "$root/artifacts/vgo-continuous/games" --window-samples "${VGO_WINDOW:-150000}" \
    --output "$out/control.pt" \
    --raster-kind compact-radius --architecture ddrnet \
    --model-width 64 --blocks 16 --context-attention-blocks 1 \
    --attention-heads 8 --norm-groups 8 --precision bfloat16 \
    --epochs 4 --batch-size 64 --learning-rate 0.0005 \
    --value-weight 2.0 --ownership-weight 0.0 --validation-fraction 0.1 \
    --schedule wsd --warmup-epochs 0 --full-adam --compile \
    --seed 91000 --report-every 1 \
) || { echo "[control] training failed"; exit 1; }

echo "[control] exporting"
( cd "$root/training" && "$python" -m vgo_training.export_onnx \
    --checkpoint "$out/control.pt" --output "$out/control.onnx" \
    --maximum-batch 64 --packed-input \
) || { echo "[control] export failed"; exit 1; }

# Identical to the anchor match that produced -89, except for the seed, so the
# two results are directly comparable. SGFs kept this time: whether plies 50-70
# are real moves or both sides filling settled space is a question the JSON
# cannot answer and a human reading one game can.
echo "[control] arena vs sl-w64b16, 24 pairs at 800 sims"
"$root/target/release/vgo-arena" \
  --candidate "$out/control.onnx" \
  --opponent "$root/artifacts/sl-w64b16/sl-w64b16.onnx" \
  --candidate-raster-kind compact-radius \
  --radius 0.05555555555555555 --komi 0.09 \
  --policy-resolution 128 --resolution 256 \
  --simulations 800 --pairs 24 --max-plies 70 --threads 8 --maximum-batch 32 \
  --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
  --sgf-directory "$out/sgf" --seed 500129 \
  > "$out/arena.json"
status=$?
echo "[control] arena exited $status"
cat "$out/arena.json"
echo "===== control done $(date '+%F %T') ====="
