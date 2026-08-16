#!/usr/bin/env bash
# Spot-check ddrnet-deep-search against the model it was seeded from.
#
#   ./runs/deep-vs-komi.sh                     # latest accepted vs komi 140
#   VGO_PAIRS=80 ./runs/deep-vs-komi.sh        # tighter
#   VGO_UPDATE=9 ./runs/deep-vs-komi.sh        # pin a specific update
#   VGO_REFERENCE=000061 ./runs/deep-vs-komi.sh  # a different opponent
#
# The reference is settable because komi 140 alone cannot answer the question.
# deep-search trains on komi 140's self-play, so a model that merely learned to
# counter komi 140 specifically, without getting generally stronger, would beat
# it convincingly -- which is the one result this pairing cannot distinguish
# from real progress. komi 61 is the discriminator: it contributed nothing to
# the training data, and its gap to komi 140 is already measured at +18 +/- 51
# (artifacts/run-finals). If the gain is general, u9's margin over komi 61
# should be about its margin over komi 140 plus that 18.
#
# Sized sequentially, deliberately. A wide gap needs few games to establish: at
# a true 75% the standard error over 32 games is 7.7 points, so the null is
# already four standard errors away. Only a close result needs the expensive
# match, and then you spend on the question that is actually open.
#
# Re-runnable as the run grows: the challenger is whichever update is newest in
# `models` (pipeline-state.json), and each invocation writes its own
# records-uNNN.jsonl, so results accumulate instead of colliding. Pinning
# --update lets an old point be replayed with more games later.
#
# The challenger is the newest *accepted* model, not the highest-numbered
# updates/ directory. Those differ constantly -- 7 of the first 12 updates here
# were rejected -- and a rejected candidate is not what the run would hand you.
#
# The reference is ddrnet-attn-komi 140, which is the checkpoint this run was
# seeded from, so the match reads directly as "what has the deeper search bought
# since the fork". komi 140's position against the wider field is already known
# from artifacts/run-finals: +18 +/- 51 Elo over komi 61, at the top of a
# seven-run ladder spanning 799 Elo.
#
# WHAT THIS CAN AND CANNOT SHOW
#
# 96 games resolve about +/-70 Elo on the gap, and 240 about +/-45. The parent
# run managed +1.7 Elo/update over the stretch where it was still improving, so
# a 12-update challenger is expected to be ~20 Elo ahead -- well inside the
# noise of any affordable match. Early runs of this are therefore a check that
# nothing has gone backwards, not a measurement of progress. The measurement
# wants ~40 updates of separation.
#
# SEARCH BUDGET AND KOMI
#
# 800 simulations, matching artifacts/run-finals and every rating in
# ratings.json. Note the asymmetry this leaves: the challenger trains at 3200
# and is tested at 800. If deeper training buys strength that only appears at
# depth, this understates it. Testing at 3200 would answer a different and
# equally fair question -- at four times the cost per game, and off the
# established scale.
#
# Komi moved from 0.034 to 0.104 on 2026-08-16, so results from here are NOT
# comparable with the two matches already in this directory, nor with
# run-finals, nor with ratings.json -- all of those are 0.034. The reason for
# moving: 0.034 was the balanced value long ago; the 50% crossing is now
# +0.104. At 0.034 a game goes about 80-20 to Black from the draw.
#
# An intermediate value of 0.08 was written here on 2026-08-15 and never played.
# It came from a logistic fit over ddrnet-deep-search data, whose komi range had
# sigma 0.10 -- wide enough that the tail buckets sit at 96% and 5%, and
# saturated tails pull the fitted crossing. Refitting under the narrowed range
# converged to +0.1035 over shards 20-25 of ddrnet-deep-komi, agreeing with the
# komi run's own shards 130-144 (+0.1042) from a completely different fit. Two
# narrow-range estimates agreeing against one wide-range outlier is why 0.104 is
# the value and 0.08 was the mistake.
#
# Colours swap within a pairing, so playing off-balance biases nothing -- it
# just spends most of the information in each game on a foregone conclusion.
#
# The records carry a "komi" field from this date so the two cannot be pooled by
# accident; build-dense-curve.py filters on it and treats a record without the
# field as 0.034, which is what it was.
#
# CONTENDS WITH THE RUN. The generator is on the same card, so this slows the
# run it is measuring, roughly in proportion to how long it takes.
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

challenger="${VGO_UPDATE:-}"
if [ -z "$challenger" ]; then
  challenger="$("$root/training/.venv/bin/python3" - "$root" <<'PY'
import json, sys
state = json.load(open(sys.argv[1] + "/artifacts/ddrnet-deep-search/pipeline-state.json"))
accepted = [m["version"] for m in state["models"]
            if isinstance(m, dict) and m.get("version", -1) >= 0]
if not accepted:
    raise SystemExit("no accepted model yet")
print(max(accepted))
PY
)"
fi
printf -v padded "%06d" "$challenger"

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

reference="${VGO_REFERENCE:-000140}"
new="$root/artifacts/ddrnet-deep-search/updates/update-$padded/candidate.onnx"
old="$root/artifacts/ddrnet-attn-komi/updates/update-$reference/candidate.onnx"
for model in "$new" "$old"; do
  [ -f "$model" ] || { echo "missing checkpoint: $model" >&2; exit 1; }
done

# Keyed by both sides: the same challenger is deliberately played against more
# than one reference, so challenger alone would collide.
records="$out/records-u$padded-vs-komi$reference.jsonl"
if [ -s "$records" ]; then
  echo "$records already has results; move it aside to replay this point" >&2
  exit 1
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

{
  echo "=== $(date -Is) deep-search update $challenger vs ddrnet-attn-komi $((10#$reference))"
  echo "=== $(( ${VGO_PAIRS:-48} * 2 )) games at 800 simulations"
} | tee -a "$out/logs/run.log"

exec "${VGO_BINARY:-$root/target/release/vgo-tournament}" \
  --model "$new" \
  --model "$old" \
  --pairs "${VGO_PAIRS:-48}" \
  --concurrency "${VGO_CONCURRENCY:-64}" \
  --simulations 800 \
  --maximum-plies 105 \
  --coarse-pool 16 --leaf-batch 4 \
  --maximum-batch 64 --delay-ms 1 \
  --resolution 128 --policy-resolution 128 \
  --radius 0.055714285714285716 --komi 0.104 \
  --provider tensorrt --fp16 \
  --cache-directory "$root/artifacts/onnx-cache" \
  --seed "$((8160000 + 1000 * (10#$challenger) + 10#$reference))" \
  "$@" > "$records"
