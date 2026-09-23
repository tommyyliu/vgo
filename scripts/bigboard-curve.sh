#!/usr/bin/env bash
# Rate a handful of vgo-continuous checkpoints on the board they were trained on.
#
# ## Why this exists
#
# The loop's own rating arena plays at `--radius 0.0555` with a flat
# `--max-plies 70` -- the mini board. Generation draws from `50:38 / 25:18 /
# 25:18-38`, so three quarters of the training games are bigger than anything
# ever measured, and the run's whole rating history describes a board it mostly
# does not play on. This asks the same question on the 1/38 board.
#
# Generation scales its ply cap by `(reference / radius)^2`; the arena takes a
# flat number, so it is told: 70 at 1/18 is 312 at 1/38. Komi follows the same
# rule generation uses, `KOMI_AREA_COEFFICIENT * radius^2`, which is 0.104 at
# 1/18 and 0.023335 here. Both seats always get the same simulation count -- an
# unequal budget measures what search is worth, not who is stronger.
#
# Most of these games end at the cap and are adjudicated by area: 172 of 451
# size-38 games in the run's own data did. That is the expected ending here, not
# a truncation artifact -- the bots effectively never resign, and by 312 plies
# the board is full and the position settled. The rate is reported for context.
#
# ## The shape, not the number
#
# A ladder of adjacent checkpoints plus one end-to-end link. Adjacent pairs are
# close in strength, so each link is the slope of the curve there; the long link
# measures the total and catches non-transitivity between the two. Ratings come
# from fitting the whole graph at once (`scripts/bigboard-ratings.py`), so the
# links constrain each other rather than being read one at a time.
#
# Games are long here -- a full board is ~330 stones against the mini board's
# ~28 -- so a skill difference has far more moves to compound over than it does
# at 1/18. That is what buys a usable signal from four games a link.
#
# ## Cost and incremental output
#
# One process per candidate, each playing its opponents in sequence and emitting
# a JSON record per opponent. A process that dies loses only the link in hand:
# everything already written stays, and re-running skips what `matches.jsonl`
# already holds. A single invocation covering every link would be unrecoverable.
#
# `vgo-arena` intermittently aborts with `corrupted double-linked list` during
# teardown, after the JSON is written. The exit code is recorded rather than
# acted on; a link whose record landed is kept.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="${VGO_MODELS:-$root/artifacts/vgo-continuous/models}"
output="${VGO_OUTPUT:-$root/artifacts/bigboard-curve}"
matches="$output/matches.jsonl"
log="$output/curve.log"

# Size 38: the board that is half of generation on its own, and the top of the
# mix's range. 1/18 is the mini board the loop has always rated on.
radius="${VGO_RADIUS:-0.02631578947368421}"
komi="${VGO_KOMI:-0.023335}"
max_plies="${VGO_MAX_PLIES:-312}"
# A quarter of the 800 every 1/18 rating was measured at. A 1/38 game is ~4.5x
# the plies, so 800 here costs what nothing else in this project costs, and the
# nerf is identical on both seats.
#
# It is not neutral, but it cuts the right way: less search leans harder on the
# policy head, and search is what lets a weaker net catch up to a stronger one.
# Checkpoint differences therefore read *wider* at 200 than at 800, which is
# what a shape-of-the-curve pass wants. The absolute numbers are not comparable
# with the 1/18 series -- one simulation doubling is worth ~61 Elo on its own.
simulations="${VGO_SIMULATIONS:-200}"
# The savings go back into games rather than into search. Four games a link
# leaves a 95% interval about +/-0.35 wide; six is still thin read alone, which
# is why the links are fitted as a graph rather than one at a time.
pairs="${VGO_PAIRS:-3}"
# Concurrent games. Both games of a pair and both opponents' worth run together,
# so a link of `pairs` pairs finishes in about the wall time of its longest game
# as long as this is at least `2 * pairs`.
threads="${VGO_THREADS:-8}"
seed_base="${VGO_SEED_BASE:-600000}"

# Each entry is `candidate:opponent,opponent`. One process per candidate loads
# that model once and plays each opponent in turn -- ~21s of load against ~1s a
# pair, so grouping is most of the fixed cost of the whole sweep.
#
# The links: 0-12, 12-24, 24-36, 36-45 walk the run, and 45-0 closes it.
links=(
  "12:0,24"
  "36:24,45"
  "45:0"
)

# A directory per candidate. `vgo-arena` names an SGF
# `<parent>-gameNNN-cand{B,W}-{capped,natural}.sgf`, which carries neither model,
# so every link in a sweep writes the same names and the later ones silently
# overwrite the earlier: the first pass here kept 14 files from 30 games. The
# games are the evidence behind a rating, so they get somewhere unique to live.
mkdir -p "$output" "$output/sgf"
source "$root/scripts/env/ort.sh"

model_path () { printf '%s/update-%06d.onnx' "$models" "$1"; }

# A link already in `matches.jsonl` is not replayed. The record names both
# models, so the file is its own progress marker -- no separate state to fall
# out of step with the results.
already_done () {
  local candidate="$1" opponent="$2"
  [ -f "$matches" ] || return 1
  "$root/training/.venv/bin/python" - "$matches" "$candidate" "$opponent" <<'PY'
import json, re, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text()
want = (sys.argv[2], sys.argv[3])
for blob in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw):
    record = json.loads(blob)
    pair = (record.get("candidate_model", ""), record.get("opponent_model", ""))
    if tuple(Path(p).name for p in pair) == want:
        sys.exit(0)
sys.exit(1)
PY
}

echo "===== big-board curve $(date '+%F_%T') : radius $radius, komi $komi, cap $max_plies, $simulations sims, $pairs pairs =====" >> "$log"

for link in "${links[@]}"; do
  candidate="${link%%:*}"
  opponents="${link#*:}"
  candidate_path="$(model_path "$candidate")"
  [ -f "$candidate_path" ] || { echo "[curve] missing $candidate_path" >> "$log"; continue; }

  opponent_flags=()
  pending=()
  IFS=',' read -ra wanted <<< "$opponents"
  for opponent in "${wanted[@]}"; do
    opponent_path="$(model_path "$opponent")"
    if [ ! -f "$opponent_path" ]; then
      echo "[curve] missing $opponent_path" >> "$log"
      continue
    fi
    if already_done "$(basename "$candidate_path")" "$(basename "$opponent_path")"; then
      echo "[curve] have update-$(printf '%06d' "$candidate") vs update-$(printf '%06d' "$opponent"), skipping" >> "$log"
      continue
    fi
    opponent_flags+=(--opponent "$opponent_path")
    pending+=("$opponent")
  done
  [ "${#opponent_flags[@]}" -eq 0 ] && continue

  echo "[curve] update-$(printf '%06d' "$candidate") vs ${pending[*]} ($pairs pairs each) started $(date '+%T')" >> "$log"
  started=$(date +%s)
  "$root/target/release/vgo-arena" \
    --candidate "$candidate_path" "${opponent_flags[@]}" \
    --candidate-raster-kind compact-radius \
    --radius "$radius" --komi "$komi" --max-plies "$max_plies" \
    --policy-resolution 128 --resolution 256 \
    --simulations "$simulations" --pairs "$pairs" \
    --threads "$threads" --maximum-batch 32 \
    --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
    --cache-directory "$root/artifacts/onnx-cache" \
    --sgf-directory "$output/sgf/update-$(printf '%06d' "$candidate")" \
    --seed $(( seed_base + candidate )) \
    >> "$matches" 2>> "$log"
  status=$?
  echo "[curve] update-$(printf '%06d' "$candidate") exited $status after $(( $(date +%s) - started ))s" >> "$log"
done

echo "[curve] finished $(date '+%F_%T')" >> "$log"
