#!/usr/bin/env bash
# Swap a running generator's model in place, through the real binary on CPU.
#
# Checks what the loop relies on: a request is answered with `.done`, games go
# on landing in the new label's directory, a game that spans the swap lists both
# models, a bad request is refused with `.failed` while generation carries on,
# and the stop file still ends the process cleanly.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source "$root/scripts/env/ort.sh"
work="$(mktemp -d)"
pid=""
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; rm -rf "$work"' EXIT
model="$root/correctness/fixture/model.onnx"
fail () { echo "generator-swap: $*" >&2; [ -f "$work/log" ] && tail -20 "$work/log" >&2; exit 1; }

# Wait up to $1 seconds for a condition.
wait_for () {
  local seconds="$1"; shift
  for ((tick = 0; tick < seconds * 4; tick++)); do
    "$@" && return 0
    sleep 0.25
  done
  return 1
}
games_in () { alive; [ "$(find "$work/out/$1" -maxdepth 1 -name 'game-*' ! -name '*.staging' 2>/dev/null | wc -l)" -ge "$2" ]; }

cat > "$work/tail.json" <<'JSON'
{"stages": [0.1], "boards": [{"up_to_units": null, "thresholds": [0.3]}]}
JSON
"$root/target/release/vgo-generate-continuous" --output-root "$work/out" --label first \
  --stop-file "$work/stop" --swap-file "$work/swap.json" \
  --model "$model" --provider cpu --fp16 false --cache-directory "$work/cache" \
  --actors 4 --inference-slots 1 --maximum-batch 8 --delay-ms 1 \
  --simulations 16 --fast-simulations 8 --max-plies 40 --board-mix 1:18 \
  --tail-schedule "$work/tail.json" --tail-simulations 8 --tail-window 3 --tail-exit-value 0.2 \
  --resolution 64 --policy-resolution 64 --telemetry-seconds 1 --seed 11 \
  > "$work/summary.json" 2> "$work/log" &
pid=$!
alive () { kill -0 "$pid" 2>/dev/null || fail "the generator exited"; }

wait_for 120 games_in first 4 || fail "no games before the swap"
printf '{"label": "second", "model": "%s", "tail_schedule": "%s"}' "$model" "$work/tail.json" \
  > "$work/swap.staging" && mv "$work/swap.staging" "$work/swap.json"
wait_for 60 test -f "$work/swap.json.done" || fail "the swap was not answered"
[ "$(cat "$work/swap.json.done")" = second ] || fail "answered for $(cat "$work/swap.json.done")"
[ ! -e "$work/swap.json" ] || fail "the request was left in place"

# Games in flight at the swap finish in the new directory with both models listed.
spanned () { grep -qs '"behavior_models"' "$work"/out/second/game-*/manifest.json; }
wait_for 120 spanned || fail "no game lists the models that played it"
wait_for 120 games_in second 6 || fail "games stopped after the swap"
swap_reported () { grep -qs '"kind":"swap"' "$work/out/second/performance.jsonl"; }
wait_for 30 swap_reported || fail "no swap record in telemetry"

# A request for the label already running is refused, and generation goes on.
printf '{"label": "second", "model": "%s"}' "$model" > "$work/swap.staging" \
  && mv "$work/swap.staging" "$work/swap.json"
wait_for 30 test -f "$work/swap.json.failed" || fail "a bad request was not refused"
kill -0 "$pid" 2>/dev/null || fail "a refused request stopped the generator"
before=$(find "$work/out/second" -maxdepth 1 -name 'game-*' | wc -l)
more () { [ "$(find "$work/out/second" -maxdepth 1 -name 'game-*' | wc -l)" -gt "$before" ]; }
wait_for 120 more || fail "no games after a refused request"

touch "$work/stop"
wait_for 120 bash -c "! kill -0 $pid 2>/dev/null" || fail "the stop file did not end the run"
wait "$pid" || fail "the generator exited with an error"
pid=""
grep -q '"label": "second"' "$work/summary.json" || fail "the summary does not name the new label"
echo "generator swap ok"
