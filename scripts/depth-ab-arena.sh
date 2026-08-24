#!/usr/bin/env bash
# Rate the two depth-ab arms against each other.
#
# Lives in scripts/ rather than runs/: everything in runs/ launches the trainer,
# a tournament, or a one-off supervised run, and `runs/` is tested for it. This
# drives the arena directly, so it is a measurement tool, not a recipe.
#
# Both were trained on the same shards with the same seed and schedule, so this
# is capacity and nothing else. Equal simulations on both seats: the large model
# costs ~3.7x more to train and is slower per evaluation, but this asks whether
# the weights are better, not whether they are worth their compute. If the
# answer is yes, the follow-up question -- equal *time* rather than equal
# simulations -- is the one that decides whether it belongs in the loop.
#
# Chunked so a partial run still says something; a job silent until the end is
# unrecoverable and reads as hung.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
source scripts/env/ort.sh

out="${VGO_OUTPUT:-$root/artifacts/depth-ab}"
pairs="${VGO_PAIRS:-30}"
sims="${VGO_SIMS:-256}"
seed="${VGO_SEED:-73001}"
chunks="${VGO_CHUNKS:-4}"

for arm in small large; do
  [ -e "$out/$arm/candidate.onnx" ] || { echo "missing $arm/candidate.onnx" >&2; exit 1; }
done

for chunk in $(seq 1 "$chunks"); do
  file="$out/h2h-$chunk.json"
  [ -s "$file" ] && { echo "skip chunk $chunk"; continue; }
  cargo run --release -q -p vgo-selfplay --bin vgo-arena -- \
    --candidate "$out/large/candidate.onnx" --candidate-raster-kind compact-pass \
    --opponent  "$out/small/candidate.onnx" --opponent-raster-kind compact-pass \
    --pairs "$pairs" --simulations "$sims" --coarse-pool 16 \
    --widening-coefficient 4.0 --maximum-candidates 321 \
    --ruleset vgo --max-plies 70 --threads 24 --leaf-batch 4 \
    --resolution 128 --policy-resolution 128 --radius 0.05555555555555555 \
    --komi 0.104 --seed "$((seed + chunk))" --maximum-batch 32 --delay-ms 1 \
    --provider tensorrt --device-id 0 --fp16 true \
    --cache-directory "$root/artifacts/onnx-cache" > "$file" 2>"$out/h2h-$chunk.err"
  if [ -s "$file" ]; then
    python3 -c "
import json;d=json.load(open('$file'))
print(f\"  chunk $chunk: large {d['candidate_wins']}-{d['candidate_losses']} of {d['completed']} = {d['candidate_score']:.1%}\")"
  else
    echo "  chunk $chunk FAILED"; tail -3 "$out/h2h-$chunk.err"
  fi
done

python3 - "$out" <<'PY'
import glob, json, math, sys
def binom_cdf(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def cp(k,n,a=0.05):
    a/=2; x,y=0.0,1.0
    for _ in range(100):
        m=(x+y)/2
        if 1-binom_cdf(k-1,n,m)<a: x=m
        else: y=m
    lo=(x+y)/2
    x,y=0.0,1.0
    for _ in range(100):
        m=(x+y)/2
        if binom_cdf(k,n,m)>a: x=m
        else: y=m
    return lo,(x+y)/2
def elo(s):
    s=min(max(s,1e-9),1-1e-9); return -400*math.log10(1/s-1)
w=g=0
for f in sorted(glob.glob(f"{sys.argv[1]}/h2h-*.json")):
    d=json.load(open(f)); w+=d['candidate_wins']+0.5*d['draws']; g+=d['completed']
if not g: sys.exit("no games")
lo,hi=cp(int(round(w)),g)
print(f"\nlarge (w128/b32) vs small (w64/b16): {w:.0f}/{g} = {w/g:.1%}")
print(f"  Elo {elo(w/g):+.0f}   95% CI [{elo(lo):+.0f}, {elo(hi):+.0f}]")
PY
