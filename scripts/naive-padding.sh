#!/usr/bin/env bash
# Naive-evaluator games, to pad the window while the model is no better than one.
#
# The model generator is GPU-bound -- 60% GPU against 3.6 of 32 cores -- so the
# machine has idle CPU that model generation cannot use. `NaiveEvaluator` never
# touches the GPU, so these run in that gap for free.
#
# Worth doing only while the premise holds. Measured at update 0, the trained
# model scored 20-20 against naive at 18 units: exactly even, so naive games are
# exactly as good as model games as training data, and they carry real game
# outcomes, which is what the starved value head learns from.
#
# Stop them once the model clearly beats naive. Past that point these are not
# free data, they are a drag on the window -- and the test that says so is the
# same arena run that established the 20-20.
#
# Labels sort alongside the model generations rather than after them, so the
# sample window treats these as contemporary. As model generations advance past
# this index the games age out of the window on their own, which is the right
# default: padding that stops mattering when it stops helping.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
games="${VGO_GAMES:-$root/artifacts/vgo-continuous/games}"
actors="${VGO_NAIVE_ACTORS:-12}"
label="${VGO_NAIVE_LABEL:-gen-000000-naiveA}"
simulations="${VGO_NAIVE_SIMULATIONS:-1600}"
first_game="${VGO_NAIVE_FIRST_GAME:-9000000}"

source "$root/scripts/env/ort.sh"
mkdir -p "$games"
exec "$root/target/release/vgo-generate-continuous" \
  --output-root "$games" --label "$label" \
  --stop-file "$games/$label.stop" --first-game "$first_game" \
  --actors "$actors" --simulations "$simulations" \
  --resolution 256 --policy-resolution 128 --raster-kind compact-radius \
  --board-mix 50:38 --board-mix 25:18 --board-mix 25:18-38 \
  --ply-sample-rate 0.25 --max-plies 70 --radius 0.05555555555555555 \
  --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
  --komi-low 0.017 --komi-high 0.137 \
  --temperature 1.0 --temperature-plies 30 --leaf-batch 4 \
  --seed $((80000 + first_game % 1000))
