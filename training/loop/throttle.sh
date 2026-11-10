#!/usr/bin/env bash
# Run a loop's heavy work at a duty cycle during set hours, to keep the machine
# quieter. Outside those hours, and whenever this script exits, everything runs
# freely.
#
#   training/loop/throttle.sh OUTPUT LOOP_PID [FIRST_HOUR LAST_HOUR [DUTY_PERCENT [PERIOD_SECONDS]]]
#   training/loop/throttle.sh OUTPUT LOOP_PID 8 22 50 4   # the defaults
#   training/loop/throttle.sh --release OUTPUT            # resume anything left paused
#
# It pauses (SIGSTOP) and resumes (SIGCONT) the generator, training, export and
# rating processes whose command lines name OUTPUT, and their children. The GPU
# averages a short period out: a 4 s period at 50% reads as half the power to
# the fans, not as bursts. Hours are local and inclusive of FIRST_HOUR, exclusive
# of LAST_HOUR. It exits when LOOP_PID does.
#
# Kill it with TERM, never KILL: TERM resumes everything first. After a KILL,
# run it with --release.
set -uo pipefail

targets () {
  local output="$1" p c
  local -a found=()
  for p in /proc/[0-9]*; do
    c=$( { tr '\0' ' ' < "$p/cmdline"; } 2>/dev/null) || continue
    # Each program by the argument that ties it to this run, not by OUTPUT
    # anywhere on the line: another run seeded from this one's models names
    # them too, and a shell or editor may mention any of it.
    case "$c" in
      */vgo-generate-continuous\ *"--output-root $output/games "*|\
      */vgo-arena\ *"--candidate $output/"*|\
      *python*loop/train-once.py\ *"--games-root $output/games "*|\
      *python*-m\ vgo_training.export_onnx\ *"--checkpoint $output/"*)
        found+=("${p#/proc/}") ;;
    esac
  done
  # Their children too: training's data workers do the CPU half of its work.
  local -a all=("${found[@]}")
  local i=0
  while [ "$i" -lt "${#all[@]}" ]; do
    for c in $( { cat "/proc/${all[$i]}/task/"*/children; } 2>/dev/null); do all+=("$c"); done
    i=$((i + 1))
  done
  echo "${all[@]}"
}

signal () {
  local name="$1"; shift
  [ "$#" -gt 0 ] && kill "-$name" "$@" 2>/dev/null
  return 0
}

if [ "${1:-}" = --release ]; then
  # shellcheck disable=SC2046
  signal CONT $(targets "$2")
  exit 0
fi

output="$(cd "$1" && pwd)"
loop="$2"
first="${3:-8}" last="${4:-22}" duty="${5:-50}" period="${6:-4}"
on=$(awk -v p="$period" -v d="$duty" 'BEGIN { printf "%.2f", p * d / 100 }')
off=$(awk -v p="$period" -v d="$duty" 'BEGIN { printf "%.2f", p * (100 - d) / 100 }')
paused=""
release () { signal CONT $paused; paused=""; }
trap 'release; exit 0' TERM INT HUP
trap 'release' EXIT
echo "[throttle] $output: ${duty}% duty, ${period}s period, ${first}:00-${last}:00 (pid $$)"
while kill -0 "$loop" 2>/dev/null; do
  hour=$((10#$(date +%H)))
  if [ "$hour" -ge "$first" ] && [ "$hour" -lt "$last" ]; then
    paused=$(targets "$output")
    # shellcheck disable=SC2086
    signal STOP $paused
    sleep "$off"
    release
    sleep "$on"
  else
    sleep 60
  fi
done
echo "[throttle] loop $loop has exited"
