#!/usr/bin/env bash
# Kill the training process before the box starts swapping, not after.
#
# A 200k window filled all 7 GB of swap while still loading and had to be killed
# by hand; at that point the machine is unusable and the run is not recoverable
# anyway. Thrashing is worse than a clean abort, so abort first.
set -uo pipefail
floor="${1:-4}"          # GB of MemAvailable below which we abort
log="${2:-/dev/stderr}"
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  swap=$(awk '/SwapFree/{f=$2} /SwapTotal/{t=$2} END{print (t>0)? int((t-f)/1048576) : 0}' /proc/meminfo)
  pid=""
  for p in /proc/[0-9]*; do
    q=${p#/proc/}
    c=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null) || continue
    case "$c" in *train-once*) pid=$q; break;; esac
  done
  [ -z "$pid" ] && exit 0            # training finished or never started
  if [ "$avail" -lt "$floor" ] || [ "$swap" -gt 2 ]; then
    echo "[watchdog] aborting: ${avail}GB available, ${swap}GB swap in use" | tee -a "$log"
    kill -9 "$pid" 2>/dev/null
    exit 1
  fi
  sleep 15
done
