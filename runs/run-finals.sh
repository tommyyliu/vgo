#!/usr/bin/env bash
# Where each run actually ended: a round-robin between the final model of every
# major run, and nothing else.
#
#   ./runs/run-finals.sh                     # -> artifacts/run-finals
#   VGO_PAIRS=6 ./runs/run-finals.sh         # cheaper
#
# The field is each run's last *accepted* model, read from `models` in its
# pipeline-state.json rather than from the highest-numbered updates/ directory.
# Those differ: ddrnet-attn-komi's last directory is update 144, a candidate
# that lost its promotion arena 6-10 and was discarded, while the model the run
# actually ended on is update 140. build-dense-curve rates every exported
# candidate.onnx and so does not make this distinction -- which is why
# ratings.json's "best checkpoint" for that run (komi 58, 1314 Elo) is also a
# rejected candidate.
#
#   ddrnet-attn-komi   140    dynamic komi, continued from fresh-attn 59
#   ddrnet-fresh-attn   59    Adam, 5.1k samples/shard
#   ddrnet-fresh-muon   53    Muon, 5.1k samples/shard
#   shard-sweep-15000   39    Muon, 15.7k samples/shard
#   shard-sweep-10000   28    Muon, 10.7k samples/shard
#   shard-sweep-5000     9    Muon, 5.8k samples/shard
#
# komi 61 is the seventh player and the only non-final one. It is the last model
# accepted before the overnight segment, so including it is what makes this
# tournament answer "did last night's 79 updates buy anything" as well as "where
# did each run end". It costs six extra pairings.
#
# Sizing: 21 pairings x 10 colour-swapped pairs = 420 games, 120 per model.
# night-vs-best measured 13.2 games/min on this box, so about 32 minutes.
#
# Search settings are identical to night-vs-best.sh -- 800 simulations, 105
# plies, komi 0.034, coarse-pool 16, leaf-batch 4, 128x128 -- so the two fields
# are comparable, with komi 61 appearing in both.
#
# Rules: this plays the fixed game (the game.rs even-trade fix). Every rating in
# ratings.json predates it. See runs/night-vs-best.sh for the full note.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  out="$1"
  shift
fi
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
mkdir -p "$out/logs"

# ONNX Runtime is loaded at runtime by path, not linked. Without ORT_DYLIB_PATH
# the `ort` crate falls back to fetching a runtime and the process hangs with
# one thread, no error and no games -- see the note in night-vs-best.sh.
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

arguments=()
for entry in \
  "ddrnet-attn-komi:000140" \
  "ddrnet-attn-komi:000061" \
  "ddrnet-fresh-attn:000059" \
  "ddrnet-fresh-muon:000053" \
  "shard-sweep-15000:000039" \
  "shard-sweep-10000:000028" \
  "shard-sweep-5000:000009"
do
  model="$root/artifacts/${entry%%:*}/updates/update-${entry##*:}/candidate.onnx"
  [ -f "$model" ] || { echo "missing checkpoint: $model" >&2; exit 1; }
  arguments+=(--model "$model")
done

# vgo-tournament has no resume and replays the whole round-robin, so appending
# to an existing file would double-count every pairing already in it.
if [ -s "$records" ]; then
  echo "$records already has results; move it aside to replay the field" >&2
  exit 1
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

{
  echo "=== $(date -Is) starting, output $out"
  echo "=== binary $binary ($(date -Is -r "$binary"))"
  echo "=== 7 models, 21 pairings"
} | tee -a "$out/logs/run.log"

exec "$binary" \
  "${arguments[@]}" \
  --pairs "${VGO_PAIRS:-10}" \
  --concurrency "${VGO_CONCURRENCY:-80}" \
  --simulations 800 \
  --maximum-plies 105 \
  --coarse-pool 16 --leaf-batch 4 \
  --maximum-batch 64 --delay-ms 1 \
  --resolution 128 --policy-resolution 128 \
  --radius 0.055714285714285716 --komi 0.034 \
  --provider tensorrt --fp16 \
  --cache-directory "$root/artifacts/onnx-cache" \
  --seed 8150003 \
  "$@" > "$records"
