#!/usr/bin/env bash
# Hash whole self-play and arena games, run through the real binaries on CPU.
#
# `vgo-verify` hashes the engine. This hashes what sits above it: generation,
# resignation, replay writing, the broker and arena play, none of which it sees.
# An untrained model is enough, because the digest compares builds and says
# nothing about strength. Output is independent of actor count, so a broker or
# scheduling change that alters a game shows up as a mismatch.
#
# Inference floats can move with the onnxruntime version or the CPU, so a
# mismatch after either changes needs that explanation before refreshing golden.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source "$root/scripts/env/ort.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
model="$root/correctness/fixture/model.onnx"
bin="$root/target/release"

generate () {
  local ruleset="$1" actors="$2" maximum_plies="$3"; shift 3
  local output="$work/generate-$ruleset-$actors-$maximum_plies-${#evaluator[@]}"
  "$bin/vgo-generate-continuous" --output-root "$output" --label digest \
    "${evaluator[@]}" --provider cpu --fp16 false --cache-directory "$work/cache" \
    --maximum-games 8 --actors "$actors" --inference-slots 1 --maximum-batch 8 --delay-ms 1 \
    --simulations 24 --fast-simulations 8 --full-search-fraction 0.5 --max-plies "$maximum_plies" \
    "$@" --ruleset "$ruleset" \
    --coarse-pool 16 --widening-coefficient 6.0 --maximum-candidates 321 \
    --temperature 1.0 --temperature-plies 10 --root-exploration-noise 0.1 \
    --resign-threshold "$RESIGN" --resign-window 3 --resign-minimum-ply 4 \
    --resign-soft-simulations 8 --resign-disable-fraction 0.25 \
    --resolution 64 --policy-resolution 64 --telemetry-seconds 0 --seed 7 \
    >"$output.log" 2>&1 || { cat "$output.log" >&2; return 1; }
  # The first eight only: a stopping generator may finish extra games in flight.
  for index in 0 1 2 3 4 5 6 7; do
    local game="$output/digest/game-00000000$index"
    cat "$game/dataset.vgo" "$game/games.jsonl" "$game/manifest.json" \
      "$game/resign-calibration.jsonl"
  done | sha256sum | cut -d' ' -f1
}

arena () {
  local threads="$1"; shift
  local output="$work/arena-$threads-$*"
  output="${output// /_}"
  mkdir -p "$output"
  "$bin/vgo-arena" --candidate "$model" "$@" --candidate-raster-kind compact-radius \
    --provider cpu --fp16 false --cache-directory "$work/cache" --sgf-directory "$output" \
    --resolution 64 --policy-resolution 64 --radius 0.05555555555555555 --komi 0.09 \
    --simulations 24 --pairs 2 --max-plies "$arena_plies" --threads "$threads" --maximum-batch 8 \
    --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 --seed 7 \
    >"$output.json" 2>"$output.log" || { cat "$output.log" >&2; return 1; }
  [ "$(find "$output" -name '*.sgf' | wc -l)" -eq 4 ] || { echo "arena wrote no SGFs" >&2; return 1; }
  # Hashed per game and sorted: file numbers follow finishing order, which is timing.
  find "$output" -name '*.sgf' -exec sha256sum {} + | cut -d' ' -f1 | LC_ALL=C sort \
    | sha256sum | cut -d' ' -f1
}

RESIGN="${RESIGN:-0.3}"
# `echo "$(...)"` would hide a failed run behind echo's exit status.
report () {
  local name="$1" value
  shift
  value="$("$@")"
  echo "$name $value"
}
# Model games reach the cap; naive ones end by passing, which covers the other exit.
evaluator=()
report generate-naive generate vgo 1 40 --board-mix 1:18
evaluator=(--model "$model")
report generate-mixed generate vgo 4 24 --board-mix 1:18 --board-mix 1:18-30
report generate-vgo generate vgo 1 40 --board-mix 1:18
report generate-official generate official 1 40 --board-mix 1:18
arena_plies=24
report arena-vs-model arena 2 --opponent "$model"
report arena-vs-naive arena 1
report arena-reuse arena 2 --opponent "$model" --tree-reuse
# Early enough that games hit the cap and are adjudicated instead of ending.
arena_plies=6
report arena-capped arena 1 --opponent "$model"
