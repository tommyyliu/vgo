#!/usr/bin/env bash
# Measure the ddrnet-attn-komi overnight segment (updates 62-144) against the
# strongest checkpoints that existed before it.
#
#   ./runs/night-vs-best.sh                       # -> artifacts/night-vs-best
#   VGO_PAIRS=4 ./runs/night-vs-best.sh           # fewer games
#   VGO_BINARY=... ./runs/night-vs-best.sh        # replay under other rules
#
# A complete round-robin, not a sampled curve. The field is ten players, so
# every new checkpoint plays every old one directly and the answer does not
# have to be read off a pooled scale.
#
# Why this is not run through dense-curve.py. That tool bands on prior ratings
# to avoid lopsided pairings across a field of eighty; at ten players banding
# has nothing to choose between, and its stride mechanism (`run:stride:low-high`,
# keeping versions where `version % stride == 0`) cannot express a hand-picked
# field that has to include update 58 specifically. Everything else here --
# 800 simulations, 105 maximum plies, komi 0.034, coarse-pool 16, leaf-batch 4,
# 128x128 -- is copied from the invocation dense-curve.py builds, so the games
# are the same games.
#
# THE RULES BOUNDARY, which is the reason this recipe exists as its own script.
#
# crates/vgo-core/src/game.rs was edited on 2026-08-14 at 22:12 to fix a
# no-op-placement bug: a placement that captured exactly one enemy stone left
# the stone count unchanged and was therefore scored as a pass, so two even
# trades in a row ended a live game and scored a position neither player had
# passed on. vgo-generate-demo was rebuilt at 22:13 and vgo-arena at 22:15, and
# the coordinator resumed at 22:36:24 on shard 84. So within this one run:
#
#   updates <= 83   self-play generated under the old, buggy rule
#   updates >= 84   self-play generated under the fixed rule
#
# Every rating in ratings.json, and every records.jsonl under artifacts/, was
# played by a vgo-tournament built 2026-08-13 15:27 -- before the fix. Those
# numbers therefore rate the models at a game that the newest checkpoints were
# not trained on.
#
# This tournament uses a rebuilt binary and plays the fixed game, because that
# is the game the project plays from here. The pre-fix binary is kept beside the
# results as vgo-tournament-prefix-rules so the same field can be replayed under
# the old rule as a control; VGO_BINARY selects it.
#
# The confound this leaves, stated rather than hidden: the four reference
# checkpoints (komi 54/58/61, fresh-attn 59) and komi 70 all trained under the
# buggy rule, while komi 85-144 trained under the fixed one. A win for the newer
# checkpoints here is partly "trained more" and partly "trained on this game".
# The control run separates them; this one alone does not.
#
# Why the records file is not named records.jsonl. build-dense-curve.py pools
# every `artifacts/*/records.jsonl` it can find into one Bradley-Terry fit, with
# only the search budget checked. These games are at the same 800 simulations
# but under different rules, so that glob would silently mix two games into one
# scale. The name keeps them out until somebody decides otherwise.
#
# The field.
#
#   Overnight, unrated:  70, 85, 100, 115, 130, 144
#     70 sits below the rules boundary and 85 just above it, so the pair also
#     brackets the change. 144 is the run's final model.
#   Previously strongest, already rated:
#     komi 58   1314 Elo, the best checkpoint anywhere before last night
#     komi 61   1286, the last model before the overnight segment
#     komi 54   1273
#     fresh-attn 59  1176, the best of the run this one continues, and the
#     checkpoint komi was seeded from -- it reads "previously" in the other
#     sense, the strongest thing from before this run existed.
#
# Sizing. 45 pairings x 6 colour-swapped pairs = 540 games, 108 per checkpoint,
# which is the game count that took the extension curve to about +/-45 Elo.
# dense-curve.py measured 0.165 games/s at 800 simulations on an idle box, so
# roughly 55 minutes. Concurrency is 80 rather than the curve's 100 because a
# vgo-serve-move instance was holding 740 MB of the card when this was written.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)"
# An output directory is optional and must come first; anything else is passed
# through to vgo-tournament. A leading dash is never treated as a path, so
# `--dry-run` cannot become a directory name -- the same guard extension-curve.sh
# carries, for the same reason.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  out="$1"
  shift
fi
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
mkdir -p "$out/logs"

# ONNX Runtime is loaded at runtime by path, not linked, so the binary needs
# ORT_DYLIB_PATH and LD_LIBRARY_PATH pointing into the venv's wheels. Without
# them the `ort` crate falls back to fetching a runtime and the process simply
# hangs -- one thread, 3.6 MB resident, no error and no games, which reads as a
# slow TensorRT engine build rather than as a failure. dense-curve.py gets this
# from pipeline.runtime_environment(); ask that same function rather than
# restating its search order here, because it encodes which wheel layout won
# (lib vs lib64, custom build vs prebuilt) and that is not guessable.
eval "$("$root/training/.venv/bin/python3" - "$root" <<'PY'
import shlex, sys
sys.path.insert(0, sys.argv[1] + "/training")
from vgo_training.pipeline import runtime_environment
environment = runtime_environment()
for key in ("LD_LIBRARY_PATH", "ORT_DYLIB_PATH"):
    if key in environment:
        print(f"export {key}={shlex.quote(environment[key])}")
PY
)"
[ -n "${ORT_DYLIB_PATH:-}" ] || { echo "no ONNX Runtime found in the venv" >&2; exit 1; }

binary="${VGO_BINARY:-$root/target/release/vgo-tournament}"
records="$out/records-fixed-rules.jsonl"
if [ "$binary" != "$root/target/release/vgo-tournament" ]; then
  records="$out/records-$(basename "$binary").jsonl"
fi

komi="$root/artifacts/ddrnet-attn-komi/updates"
attn="$root/artifacts/ddrnet-fresh-attn/updates"
models=(
  "$komi/update-000054/candidate.onnx"
  "$komi/update-000058/candidate.onnx"
  "$komi/update-000061/candidate.onnx"
  "$komi/update-000070/candidate.onnx"
  "$komi/update-000085/candidate.onnx"
  "$komi/update-000100/candidate.onnx"
  "$komi/update-000115/candidate.onnx"
  "$komi/update-000130/candidate.onnx"
  "$komi/update-000144/candidate.onnx"
  "$attn/update-000059/candidate.onnx"
)
arguments=()
for model in "${models[@]}"; do
  [ -f "$model" ] || { echo "missing checkpoint: $model" >&2; exit 1; }
  arguments+=(--model "$model")
done

# Appending to an existing records file would double-count whatever it already
# holds: vgo-tournament plays the whole round-robin every time and has no
# resume, so a second run is a second copy of the same pairings.
if [ -s "$records" ]; then
  echo "$records already has results; move it aside to replay the field" >&2
  exit 1
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

{
  echo "=== $(date -Is) starting, output $out"
  echo "=== binary $binary ($(date -Is -r "$binary"))"
  echo "=== ${#models[@]} models, $(( ${#models[@]} * (${#models[@]} - 1) / 2 )) pairings"
} | tee -a "$out/logs/run.log"

exec "$binary" \
  "${arguments[@]}" \
  --pairs "${VGO_PAIRS:-6}" \
  --concurrency "${VGO_CONCURRENCY:-80}" \
  --simulations 800 \
  --maximum-plies 105 \
  --coarse-pool 16 --leaf-batch 4 \
  --maximum-batch 64 --delay-ms 1 \
  --resolution 128 --policy-resolution 128 \
  --radius 0.055714285714285716 --komi 0.034 \
  --provider tensorrt --fp16 \
  --cache-directory "$root/artifacts/onnx-cache" \
  --seed 8150001 \
  "$@" > "$records"
