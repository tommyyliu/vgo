#!/usr/bin/env bash
# Does raising the actor count buy generation throughput?
#
# RAM forced 24 actors while the replay window cost 25 GB. Packing the policy
# targets cut that by ~14 GB, so the question is whether generation was starved
# all along. GPU *power* is the honest signal here: utilization.gpu only reports
# whether a kernel is resident, and a GPU drawing 130 W of 300 W is idling
# between batches however busy that counter looks.
#
#   scripts/actor-scaling.sh <actors>
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/env/ort.sh"
cd "$root"
actors=$1
out=/tmp/actor-probe-$actors
rm -rf "$out"
start=$(date +%s)
"$root/target/release/vgo-generate-demo" --samples 500 --resolution 128 --policy-resolution 128 \
  --simulations 6400 --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
  --root-exploration-noise 0.0 --temperature 1.0 --temperature-plies 30 --max-plies 70 \
  --radius 0.05555555555555555 --ruleset vgo --raster-kind compact-pass \
  --model-raster-kind compact-pass --actors "$actors" --leaf-batch 4 \
  --writer-queue-games 2 --drain-tail true --maximum-batch 32 --delay-ms 1 \
  --inference-slots 2 --provider tensorrt --device-id 0 --fp16 true --runtime onnx \
  --cache-directory "$root/artifacts/onnx-cache" \
  --model "$root/artifacts/raster-ab/compact-pass/candidate.onnx" \
  --seed 5 --output "$out" > "/tmp/actor-probe-$actors.log" 2>&1 &
gen=$!
power=0; n=0; peak=0
while kill -0 $gen 2>/dev/null; do
  w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  r=$(ps -o rss= -p "$(pgrep -x vgo-generate-de | head -1)" 2>/dev/null || echo 0)
  case "$w" in ''|*[!0-9.]*) ;; *) power=$(python3 -c "print($power+$w)"); n=$((n+1));; esac
  [ "${r:-0}" -gt "$peak" ] && peak=$r
  sleep 3
done
wait $gen || true
end=$(date +%s)
[ "$n" -eq 0 ] && n=1
if [ ! -f "$out/manifest.json" ]; then
  echo "actors=$actors  GENERATION FAILED"; tail -3 "/tmp/actor-probe-$actors.log"; exit 1
fi
python3 - "$out" "$actors" "$((end-start))" "$power" "$n" "$peak" <<'PY'
import json, sys
out, actors, wall, power, n, peak = sys.argv[1:]
m = json.load(open(f"{out}/manifest.json"))
wall, power, n, peak = int(wall), float(power), int(n), int(peak)
print(f"actors={actors:>3}  wall={wall:4d}s  samples={m['samples']:4d}  games={m['completed_games']:3d}  "
      f"power={power/n:5.0f}W  rss={peak/1048576:5.1f}GB  samples/s={m['samples']/max(wall,1):5.2f}")
PY
