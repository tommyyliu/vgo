#!/usr/bin/env bash
# The RL loop without batch boundaries.
#
# One generator runs continuously, writing each finished game as its own
# one-game shard; the trainer takes the most recent N samples whenever enough
# new ones have arrived. Nothing waits for a batch to fill, so no actor is ever
# idle because the shard it is feeding is nearly full.
#
# ## Why this exists
#
# The shard pipeline spends a fixed cost per shard finishing the slowest game in
# flight. Measured on the multi-radius run: 57 minutes reaching the sample
# target with 32 actors busy, then 65 minutes with *two* busy and thirty idle --
# half the wall clock at 6% utilisation, on a shard that had already collected
# 1.66x the samples it asked for. Every available workaround only moved the cost
# around: `--no-drain-tail` discards ~32 part-games instead, a second generator
# hides the tail behind another shard, a larger shard amortises it.
#
# ## Model handoff without a hot swap
#
# Generations are directories. Training exports a new model, this starts a
# generator writing into a fresh `gen-NNNNNN-<sha>` directory, and touches the
# previous generator's stop file. That one finishes the games in hand and exits
# while the new one is already producing, so the changeover costs nothing and
# neither process ever swaps an `Arc<dyn Evaluator>` on the per-leaf hot path.
#
# The overlap is deliberate: for one game's duration two generators run, which
# is far cheaper than the thirty idle cores it replaces.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${VGO_OUTPUT:-$root/artifacts/vgo-continuous}"
games="$output/games"
models="$output/models"
window="${VGO_WINDOW_SAMPLES:-40000}"
step="${VGO_STEP_SAMPLES:-4000}"
updates="${VGO_UPDATES:-200}"
actors="${VGO_ACTORS:-32}"
simulations="${VGO_SIMULATIONS:-1600}"
seed_model="${VGO_SEED_MODEL:-}"

python="$root/training/.venv/bin/python"
source "$root/scripts/env/ort.sh"
mkdir -p "$games" "$models"

# Samples currently on disk, from manifests rather than by loading anything.
count_samples () {
  "$python" - "$games" <<'PY'
import json, sys
from pathlib import Path
total = 0
for generation in sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir()):
    for game in sorted(p for p in generation.iterdir() if p.is_dir()):
        manifest = game / "manifest.json"
        if not manifest.is_file() or not (game / "dataset.vgo").is_file():
            continue
        try:
            total += int(json.loads(manifest.read_text())["samples"])
        except (ValueError, KeyError, OSError):
            continue
print(total)
PY
}

start_generator () {
  local label="$1" model="$2" first_game="$3"
  local stop_file="$games/$label.stop"
  rm -f "$stop_file"
  local model_flag=()
  [ -n "$model" ] && model_flag=(--model "$model")
  setsid nohup "$root/target/release/vgo-generate-continuous" \
    --output-root "$games" --label "$label" \
    --stop-file "$stop_file" --first-game "$first_game" \
    --actors "$actors" --simulations "$simulations" \
    --resolution 256 --policy-resolution 128 --raster-kind compact-radius \
    --board-mix 50:38 --board-mix 25:18 --board-mix 25:18-38 \
    --ply-sample-rate 0.25 --max-plies 70 --radius 0.05555555555555555 \
    --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
    --komi-low 0.017 --komi-high 0.137 \
    --temperature 1.0 --temperature-plies 30 \
    --leaf-batch 4 --maximum-batch 32 --delay-ms 1 --inference-slots 2 \
    --provider tensorrt --fp16 true \
    --cache-directory "$root/artifacts/onnx-cache" \
    --seed $((70000 + first_game)) \
    "${model_flag[@]}" \
    >> "$output/generate.log" 2>&1 &
  echo $!
}

model="$seed_model"

# Continue the update numbering from whatever is already on disk. Restarting
# from zero rewrites `update-000000`, which on a restart is usually the model
# just passed as `--seed-model` -- the loop would destroy its own starting point
# and only notice later, when the history it wanted to compare against was gone.
first_update=$(ls "$models"/update-*.pt 2>/dev/null \
  | sed 's/.*update-0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
first_update=$(( ${first_update:-(-1)} + 1 ))
[ "$first_update" -gt 0 ] && echo "[loop] continuing from update $first_update"

generation=0
# Game indices never restart: a reused index would replay a seed, and two games
# with the same seed are the same game.
next_game=$(( 1000000 + first_update * 1000000 ))
label="gen-$(printf '%06d' "$generation")-seed"
generator=$(start_generator "$label" "$model" "$next_game")
echo "[loop] generation $generation started (pid $generator, model ${model:-none})"

for ((update = first_update; update < first_update + updates; update++)); do
  target=$(( $(count_samples) + step ))
  echo "[loop] update $update: waiting for $target samples"
  while [ "$(count_samples)" -lt "$target" ]; do
    if ! kill -0 "$generator" 2>/dev/null; then
      echo "[loop] generator exited unexpectedly; see $output/generate.log" >&2
      exit 1
    fi
    sleep 30
  done

  checkpoint="$models/update-$(printf '%06d' "$update").pt"
  echo "[loop] update $update: training on the most recent $window samples"
  ( cd "$root/training" && "$python" "$root/scripts/train-once.py" \
      --games-root "$games" --window-samples "$window" \
      --output "$checkpoint" \
      ${model:+--initial-checkpoint "${model%.onnx}.pt"} \
      --raster-kind compact-radius --architecture ddrnet \
      --model-width 64 --blocks 16 --context-attention-blocks 1 \
      --attention-heads 8 --norm-groups 8 \
      --epochs 1 --batch-size 64 --learning-rate 0.0005 \
      --value-weight 2.0 --ownership-weight 0.0 --validation-fraction 0.1 \
      --schedule wsd --warmup-epochs 0 --full-adam --compile \
      --seed $((90000 + update)) --report-every 1 \
  ) >> "$output/train.log" 2>&1 || { echo "[loop] training failed" >&2; exit 1; }

  onnx="${checkpoint%.pt}.onnx"
  ( cd "$root/training" && "$python" -m vgo_training.export_onnx \
      --checkpoint "$checkpoint" --output "$onnx" --maximum-batch 64 \
  ) >> "$output/train.log" 2>&1 || { echo "[loop] export failed" >&2; exit 1; }

  # Start the successor before stopping the incumbent, so generation never
  # pauses. `next_game` advances past anything the old process could still
  # claim.
  generation=$((generation + 1))
  next_game=$((next_game + 1000000))
  sha=$("$python" -c "
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:8])" "$onnx")
  previous_label="$label"
  previous="$generator"
  label="gen-$(printf '%06d' "$generation")-$sha"
  generator=$(start_generator "$label" "$onnx" "$next_game")
  touch "$games/$previous_label.stop"
  model="$onnx"
  echo "[loop] update $update done: generation $generation started (pid $generator), $previous_label draining"
done

touch "$games/$label.stop"
echo "[loop] finished $updates updates"
