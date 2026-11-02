#!/usr/bin/env bash
# Kill the training process before the box starts swapping, not after.
#
# A 200k window filled all 7 GB of swap while still loading and had to be killed
# by hand; at that point the machine is unusable and the run is not recoverable
# anyway. Thrashing is worse than a clean abort, so abort first.
set -uo pipefail
floor="${1:-4}"          # GB of MemAvailable below which we abort
log="${2:-/dev/stderr}"
swap_used () {
  awk '/SwapFree/{f=$2} /SwapTotal/{t=$2} END{print (t>0)? int((t-f)/1048576) : 0}' /proc/meminfo
}
# Growth, not the total: a machine that's been up a while holds idle pages in swap,
# and reading those as thrashing killed a run four seconds into its first update.
swap_at_start=$(swap_used)
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  swap=$(( $(swap_used) - swap_at_start ))
  pid=""
  for p in /proc/[0-9]*; do
    q=${p#/proc/}
    c=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null) || continue
    # The script's path, on a Python command line. Anything looser matches a shell
    # whose command happens to mention it, and this kills what it finds.
    case "$c" in *python*loop/train-once.py*) pid=$q; break;; esac
  done
  [ -z "$pid" ] && exit 0            # training finished or never started
  if [ "$avail" -lt "$floor" ] || [ "$swap" -gt 2 ]; then
    echo "[watchdog] aborting: ${avail}GB available, swap up ${swap}GB" | tee -a "$log"
    kill -9 "$pid" 2>/dev/null
    exit 1
  fi
  sleep 15
done
