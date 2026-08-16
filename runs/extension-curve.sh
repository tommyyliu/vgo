#!/usr/bin/env bash
# Rate the ddrnet-attn-komi extension densely against the tail of the run it
# continues, so the continuation's shape is measured rather than inferred.
#
#   ./runs/extension-curve.sh                      # -> artifacts/extension-curve
#   ./runs/extension-curve.sh --dry-run            # schedule and cost, plays nothing
#   VGO_ROUNDS=6 ./runs/extension-curve.sh         # more games per checkpoint
#   ./runs/extension-curve.sh artifacts/my-curve   # somewhere else
#
# An optional output directory comes first; every other argument is forwarded
# to dense-curve.py.
#
# Why this exists. In the general dense curve the extension carried 28 games a
# checkpoint against the parent's 140-308, so its points landed at +/-100 Elo
# against the parent tail's +/-32 -- wide enough that "did the continuation
# improve" could not be answered either way. Every point here is a checkpoint of
# the extension or one of the six parent checkpoints it has to beat, so the
# games go where the question is.
#
# Why the parent enters at :3:45- rather than in full. Its early checkpoints are
# 1500 Elo below this field and every game against one is decided before it is
# played, which is the pairing that carries the least information. Updates 45-59
# are the ones the extension has to beat, and they are already densely rated
# from the general curve, so they also tie these new games into the existing
# scale for free -- this tournament needs no anchor of its own.
#
# Why naive is dropped from the seed ratings. Pooled, it is banded like any
# other player, and with the weakest checkpoint here around 1100 Elo it would
# sit alone at the bottom of the band and take games it loses 8-0. Removing it
# from the matchmaking input is what makes `--naive-rounds 0` mean no naive at
# all; leaving it in the file would pool it instead. The absolute scale still
# comes from the parent's existing games against it.
#
# Why the seed ratings are generated once and then frozen. dense-curve.py builds
# its whole schedule at startup from this file and resumes by round *index*, so
# a file that changed between runs would renumber the schedule and the indices
# in rounds-done.json would no longer mean the same rounds. Regenerating it as
# the general curve improves would silently corrupt the resume.
#
# --komi 0.034 is pinned rather than left to the default, which changed to 0.08
# on 2026-08-15 when the balance point was re-measured. Every round already in
# this curve was played at 0.034, and dense-curve.py resumes by round index, so
# taking the new default would silently fit two different games as one scale.
# A curve keeps the komi it started with; a new curve gets the new default.
#
# Raising VGO_ROUNDS on resume is safe, and is the way to tighten the error bars
# without starting over: banded_rounds draws rounds in order from one seeded
# RNG, so a larger --rounds-per-checkpoint extends the schedule and leaves every
# already-played round identical. Lowering it would strand played rounds past
# the end of the new schedule, so don't.
#
# Sizing, at 19 rounds of 112 games: each checkpoint gets 4 rounds x 28 games =
# 112, which takes the extension from +/-100 Elo to roughly +/-45. About 3.6h on
# an idle card, longer while anything else is on the GPU.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)"
# An output directory is optional and must come first; anything else is passed
# through to dense-curve.py. Taking $1 unconditionally as the output silently
# swallowed `--dry-run` into the directory name and started a real 3.6h
# tournament, so a leading dash is never treated as a path.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  out="$1"
  shift
fi
# Pin a relative argument to the invoking shell's directory, as the training
# recipes do: every path below is derived from $out and the resume reads
# rounds-done.json out of it, so resolving it twice would start a second curve.
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
mkdir -p "$out/logs"

seed="$out/ratings-seed.json"
if [ ! -f "$seed" ]; then
  echo "=== building matchmaking seed from every tournament played so far"
  "$root/training/.venv/bin/python3" "$root/scripts/build-dense-curve.py" \
    --output "$out/logs/seed-curve.html" --ratings-json "$seed.all"
  # Drop naive; see the note above. Done as a separate step so the fit itself
  # stays the ordinary one and this file is reproducible from it.
  "$root/training/.venv/bin/python3" - "$seed.all" "$seed" <<'PY'
import json, sys
source, target = sys.argv[1], sys.argv[2]
ratings = json.load(open(source))
kept = {k: v for k, v in ratings.items() if not k.startswith("naive/")}
json.dump(kept, open(target, "w"), indent=2)
print(f"seed ratings: {len(kept)} checkpoints, naive dropped")
PY
  rm -f "$seed.all"
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out"

cd "$root"
exec training/.venv/bin/python3 scripts/dense-curve.py \
  artifacts/ddrnet-attn-komi:2 \
  artifacts/ddrnet-fresh-attn:3:45- \
  --ratings "$seed" \
  --rounds-per-checkpoint "${VGO_ROUNDS:-4}" \
  --field 8 --pairs 2 --band 10 --spanning-every 4 --naive-rounds 0 \
  --simulations 800 --maximum-plies 105 \
  --komi 0.034 \
  --concurrency "${VGO_CONCURRENCY:-100}" \
  --parallel-rounds "${VGO_PARALLEL_ROUNDS:-1}" \
  --output "$out" "$@"
