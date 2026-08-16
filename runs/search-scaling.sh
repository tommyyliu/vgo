#!/usr/bin/env bash
# What is search worth? One model against itself at two budgets.
#
#   ./runs/search-scaling.sh                    # -> artifacts/search-scaling
#   VGO_PAIRS=32 ./runs/search-scaling.sh       # tighter
#   VGO_MODEL=... ./runs/search-scaling.sh      # a different checkpoint
#
# Phase 1 of docs/CLIENT_BOT.md. A browser bot cannot afford 800 simulations a
# move; this measures what it gives up by running fewer. Because both seats are
# the *same network*, the score isolates search from strength: whatever the model
# is worth, the difference between the seats is what the extra simulations buy on
# top of its own priors.
#
# --opponent-simulations exists for exactly this and says so in its help text.
# Note the warning there: this is not a rating. Two seats at different budgets
# are two different players, so these records must never be pooled into the Elo
# scale -- which is why they are written to records-scaling.jsonl rather than
# records.jsonl, out of build-dense-curve.py's glob.
#
# The reference seat is 800 simulations, matching every tournament in
# artifacts/ and ratings.json, so a point here reads directly as "N simulations
# costs X Elo against the engine we have been measuring all along".
#
# Komi 0.104, the measured balance point (see runs/deep-vs-komi.sh). At the old
# 0.034 a game goes about 80-20 to Black from the draw, which would spend most
# of these games on a foregone conclusion.
#
# Expected shape, worth knowing before reading the output: the policy head is
# this project's ceiling -- the search prior sits ~0.6 bits from uniform -- so
# cutting simulations leans on the weakest component and the curve should fall
# faster than the usual "every doubling is worth a fixed Elo" rule of thumb. If
# it does *not*, that is the more interesting result: it would mean search is
# contributing little even at 800, and the client bot is nearly free.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  out="$1"
  shift
fi
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
if [ -f "$out/pipeline-config.json" ]; then
  echo "refusing to write into $out: it holds a training run" >&2
  exit 1
fi
mkdir -p "$out/logs"

# ONNX Runtime is loaded by path, not linked; without ORT_DYLIB_PATH the `ort`
# crate falls back to fetching a runtime and hangs with one thread and no error.
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

# The newest accepted model of the live run, unless told otherwise. Accepted,
# not highest-numbered: those differ, and a rejected candidate is not what the
# run would hand you.
model="${VGO_MODEL:-}"
if [ -z "$model" ]; then
  version="$("$root/training/.venv/bin/python3" - "$root" <<'PY'
import json, sys
state = json.load(open(sys.argv[1] + "/artifacts/ddrnet-deep-komi/pipeline-state.json"))
accepted = [m["version"] for m in state["models"]
            if isinstance(m, dict) and m.get("version", -1) >= 0]
print(max(accepted))
PY
)"
  printf -v padded "%06d" "$version"
  model="$root/artifacts/ddrnet-deep-komi/updates/update-$padded/candidate.onnx"
fi
[ -f "$model" ] || { echo "missing checkpoint: $model" >&2; exit 1; }

records="$out/records-scaling.jsonl"
if [ -s "$records" ]; then
  echo "$records already has results; move it aside to replay" >&2
  exit 1
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"
exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) search scaling for $model"
echo "=== reference seat 800 simulations, komi 0.104"

# One invocation per point: --simulations and --opponent-simulations are single
# values, so a budget sweep cannot be batched into one arena the way opponents
# can. Each pays a ~21s model load, which is why the list is short.
#
# Descending so the cheap, most decision-relevant points (can a 100-simulation
# browser bot play?) land first if this is interrupted.
for budget in 400 200 100 50 1600; do
  echo
  echo "=== $budget simulations vs 800"
  "$root/target/release/vgo-arena" \
    --candidate "$model" \
    --opponent "$model" \
    --candidate-raster-kind compact \
    --pairs "${VGO_PAIRS:-24}" \
    --simulations "$budget" \
    --opponent-simulations 800 \
    --coarse-pool 16 \
    --max-plies 105 \
    --threads "${VGO_THREADS:-32}" \
    --leaf-batch 4 \
    --resolution 128 --policy-resolution 128 \
    --radius 0.055714285714285716 \
    --komi 0.104 \
    --maximum-batch 32 --delay-ms 1 \
    --provider tensorrt --device-id 0 --fp16 true \
    --cache-directory "$root/artifacts/onnx-cache" \
    --seed "$((9200000 + budget))" \
    >> "$records"
done

echo
echo "=== done; read with scripts/read-search-scaling.py"
