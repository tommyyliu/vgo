#!/usr/bin/env bash
# Compress replay shards that have aged out of the training window.
#
# The learner reads only the last --replay-window shards by sequence
# (pipeline.py `_active_replay`: eligible[-replay_window:]), so anything older
# is never read again by this run. It is still worth keeping: a shard records
# which model generated it, so the set is a reproducible history and a seed for
# future runs. Compressing rather than deleting keeps that for ~3% of the space.
#
# The .vgo format stores five policy-shaped arrays of `policy_resolution^2 + 1`
# float32 slots per sample, of which ~24 are nonzero -- a dense array holding a
# handful of values. That is why zstd -3 reaches ~37x here and why compression
# is the cheap fix; the real fix is a sparse encoding in the format itself.
#
# Safety: this refuses to touch any shard inside the window, re-reading the live
# state file at run time rather than trusting a precomputed list, and verifies
# each archive round-trips before removing the original.
set -euo pipefail

run=${1:?usage: compress-aged-shards.sh <run-directory> [--keep-uncompressed N] [--apply]}
keep_extra=2
apply=0
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-uncompressed) keep_extra=$2; shift 2 ;;
    --apply) apply=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

state="$run/pipeline-state.json"
config="$run/pipeline-config.json"
[[ -f $state && -f $config ]] || { echo "not a pipeline run directory: $run" >&2; exit 1; }

window=$(python3 -c "import json;print(json.load(open('$config'))['replay_window'])")
highest=$(python3 -c "
import json
s=json.load(open('$state'))
print(max((int(r['sequence']) for r in s['replay']), default=-1))
")

# Never touch the active window, and hold back an extra margin so a shard that
# is about to enter the window (or an in-flight prefetch) is never a candidate.
cutoff=$(( highest - window - keep_extra + 1 ))
echo "highest shard=$highest  window=$window  extra margin=$keep_extra"
if (( cutoff <= 0 )); then
  echo "nothing has aged out yet (cutoff=$cutoff)"; exit 0
fi
echo "compressing shards with sequence < $cutoff"

total_before=0; total_after=0; count=0
for dir in "$run"/replay/shard-*/; do
  name=$(basename "$dir")
  [[ $name =~ ^shard-([0-9]+)$ ]] || continue
  seq=$((10#${BASH_REMATCH[1]}))
  (( seq < cutoff )) || continue
  src="$dir/dataset.vgo"
  [[ -f $src ]] || continue

  # A shard the run still has open is never a candidate, whatever the arithmetic says.
  if lsof -- "$src" >/dev/null 2>&1; then
    echo "  $name: SKIP (open file handle)"; continue
  fi

  before=$(stat -c%s "$src")
  if (( apply == 0 )); then
    printf "  %s: would compress %.2f GB\n" "$name" "$(echo "scale=4;$before/1000000000"|bc)"
    total_before=$((total_before+before)); count=$((count+1)); continue
  fi

  zstd -3 -T4 -q -f "$src" -o "$src.zst"
  # Verify the archive decompresses to the exact original bytes before deleting.
  if ! zstd -t -q "$src.zst" 2>/dev/null; then
    echo "  $name: FAILED integrity check, keeping original"; rm -f "$src.zst"; continue
  fi
  expected=$(python3 -c "
import json
s=json.load(open('$state'))
print(next((r['dataset_sha256'] for r in s['replay'] if int(r['sequence'])==$seq),''))
")
  actual=$(zstd -dc "$src.zst" | sha256sum | cut -d' ' -f1)
  if [[ -n $expected && $expected != "$actual" ]]; then
    echo "  $name: FAILED checksum ($expected != $actual), keeping original"; rm -f "$src.zst"; continue
  fi
  after=$(stat -c%s "$src.zst")
  rm -f "$src"
  printf "  %s: %.2f GB -> %.3f GB (%.1fx)\n" "$name" \
    "$(echo "scale=4;$before/1000000000"|bc)" "$(echo "scale=4;$after/1000000000"|bc)" \
    "$(echo "scale=2;$before/$after"|bc)"
  total_before=$((total_before+before)); total_after=$((total_after+after)); count=$((count+1))
done

if (( count == 0 )); then echo "no aged-out shards to compress"; exit 0; fi
if (( apply == 0 )); then
  printf "\ndry run: %d shard(s), %.1f GB. Re-run with --apply.\n" "$count" \
    "$(echo "scale=2;$total_before/1000000000"|bc)"
else
  printf "\ncompressed %d shard(s): %.1f GB -> %.2f GB\n" "$count" \
    "$(echo "scale=2;$total_before/1000000000"|bc)" "$(echo "scale=3;$total_after/1000000000"|bc)"
  echo "restore one with: zstd -d <shard>/dataset.vgo.zst"
fi
