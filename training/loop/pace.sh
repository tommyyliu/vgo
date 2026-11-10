#!/usr/bin/env bash
# Set a loop's generation duty by the hour, to keep the machine quieter in the day.
#
#   training/loop/pace.sh OUTPUT [FIRST_HOUR LAST_HOUR [DUTY_PERCENT]]
#   training/loop/pace.sh artifacts/vgo-128 8 22 50     # the defaults
#
# Writes DUTY_PERCENT to OUTPUT/pace from FIRST_HOUR (inclusive) to LAST_HOUR
# (exclusive), local time, and 100 otherwise. The generator reads that file every
# second and rests its inference lanes to match, so the GPU and the CPU behind it
# run at a steady fraction instead of starting and stopping. Training runs at full
# speed: it happens less often when generation is slower, since updates wait on
# samples. Stateless, so cron can run it every few minutes:
#
#   */5 * * * * /path/to/vgo/training/loop/pace.sh /path/to/vgo/artifacts/vgo-128
#
# To run at full speed by hand, remove the cron line and write 100 to the file.
set -euo pipefail
output="$1"
first="${2:-8}" last="${3:-22}" duty="${4:-50}"
hour=$((10#$(date +%H)))
if [ "$hour" -ge "$first" ] && [ "$hour" -lt "$last" ]; then wanted="$duty"; else wanted=100; fi
[ "$(cat "$output/pace" 2>/dev/null)" = "$wanted" ] && exit 0
echo "$wanted" > "$output/pace.staging"
mv "$output/pace.staging" "$output/pace"
