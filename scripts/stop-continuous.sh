#!/usr/bin/env bash
# Stop the continuous run without losing finished work.
#
# Games are written only when they finish -- `write_game` takes a complete
# `GameSamples`, there is no partial flush and no resume -- so killing a
# generator discards every game in flight. With 32 actors and 38-unit boards
# that can be 32 games, the oldest of them hours in.
#
# So the default drains: each generation is told to stop taking new games and
# finishes the ones it holds. `--now` skips that when the machine is needed
# back immediately and the in-flight games are worth less than the wait.
#
# ## Finding the processes
#
# By executable, matched as a *substring*, never by command line:
#
#   * A cmdline match finds this script's own shell, because the pattern being
#     searched for is in the command running the search. That has killed the
#     shell mid-cleanup here, leaving half the job done and reporting success.
#   * `pgrep -x` silently matches nothing when the binary name is over 15
#     characters, and `vgo-generate-continuous` is 23. A kill loop over it is a
#     no-op that looks like a clean shutdown.
#   * Equality against the absolute path breaks after a rebuild: a running
#     process's `exe` link then reads `<path> (deleted)`, so an exact match
#     skips exactly the stale processes a restart is trying to clear.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${VGO_OUTPUT:-$root/artifacts/vgo-continuous}"
games="$output/games"
immediate=0
timeout_minutes="${VGO_STOP_TIMEOUT_MINUTES:-180}"

for argument in "$@"; do
  case "$argument" in
    --now) immediate=1 ;;
    --help|-h)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      echo
      echo "Usage: stop-continuous.sh [--now]"
      echo "  --now   kill generators immediately, discarding games in flight"
      exit 0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

generator_pids () {
  local p pid
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    case "$(readlink "/proc/$pid/exe" 2>/dev/null)" in
      *vgo-generate-continuous*) echo "$pid" ;;
    esac
  done
}

label_of () {
  tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null | grep -A1 '^--label$' | tail -1
}

# The loop first: it watches its generator and would start a replacement.
# `$$` and every other bash are excluded explicitly -- see the header.
loop_pids=()
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  [ "$(readlink -f "/proc/$pid/exe" 2>/dev/null)" = "/usr/bin/bash" ] || continue
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "continuous-loop" && loop_pids+=("$pid")
done
if [ "${#loop_pids[@]}" -gt 0 ]; then
  echo "[stop] loop: ${loop_pids[*]}"
  kill "${loop_pids[@]}" 2>/dev/null
  sleep 2
else
  echo "[stop] no loop running"
fi

mapfile -t pids < <(generator_pids)
if [ "${#pids[@]}" -eq 0 ]; then
  echo "[stop] no generators running"
else
  for pid in "${pids[@]}"; do
    label="$(label_of "$pid")"
    echo "[stop] generator $pid ($label)"
    [ -n "$label" ] && touch "$games/$label.stop"
  done
fi

before=$(find "$games" -maxdepth 2 -name 'game-*' -type d 2>/dev/null | wc -l)

if [ "$immediate" -eq 1 ]; then
  echo "[stop] --now: discarding games in flight"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null; done
  sleep 8
  for pid in $(generator_pids); do kill -9 "$pid" 2>/dev/null; done
else
  echo "[stop] draining; finished games keep landing (Ctrl-C is safe, it just stops the wait)"
  deadline=$(( $(date +%s) + timeout_minutes * 60 ))
  while [ -n "$(generator_pids)" ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[stop] still draining after ${timeout_minutes}m; killing the rest" >&2
      for pid in $(generator_pids); do kill -9 "$pid" 2>/dev/null; done
      break
    fi
    sleep 30
  done
fi
sleep 2

after=$(find "$games" -maxdepth 2 -name 'game-*' -type d 2>/dev/null | wc -l)
staging=$(find "$games" -maxdepth 2 -name '*.staging' -type d 2>/dev/null | wc -l)
echo
echo "[stop] generators left: $(generator_pids | wc -l)"
echo "[stop] games: $before -> $after (+$(( after - before )) landed while stopping)"
echo "[stop] staging leftovers: $staging"
echo "[stop] models: $(ls "$output"/models/*.onnx 2>/dev/null | wc -l)"
