#!/usr/bin/env bash
# The shard-size sweep: identical Muon runs differing only in how much fresh
# self-play goes into each update.
#
#   ./runs/shard-sweep.sh 10000                      # -> artifacts/shard-sweep-10000
#   VGO_UPDATES=40 ./runs/shard-sweep.sh 10000       # continue an existing one
#   ./runs/shard-sweep.sh 15000 artifacts/elsewhere
#
# The argument is the *nominal* shard size, which is not what gets passed to
# --samples-per-shard. Generation cannot stop below its drain floor: the games
# already in flight yield actors x mean_plies samples on their own, about 3,520
# at 64 actors. So a request of 6,480 produces a ~10,000-sample shard, and
# asking for 10,000 would produce ~13,500. The sweep is named for what it
# actually produces; the subtraction here is what makes that true.
#
# Why no --full-adam: these are the Muon arm. Muon is the default, so the flag
# is simply absent -- but note that ddrnet-attn.sh must spell out --full-adam
# for the opposite reason. Do not "fix" the asymmetry.
#
# Everything else is byte-identical to ddrnet-attn.sh, which is the point: the
# sweep is meant to isolate shard size and nothing else. Seeds are derived from
# the nominal size so each arm is independent but reproducible.
# Inference batch is a serving control rather than an experimental variable;
# batch 32 was 9.3% faster than 64 in a paired run on this exact model shape.
#
# Measured so far, at fixed total data, bigger shards are *worse*: -190 +/- 53
# Elo (3.6 sigma) against the small-shard arm. This script exists to push the
# arms further and find out whether that gap closes.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <nominal-shard-size> [output-dir]" >&2
  exit 2
fi
size="$1"

# The floor generation cannot go below; see the header. Kept as a named
# constant because the shard a run actually produced is size, not size-floor,
# and every comparison between arms depends on that being right.
#
# It is actors x mean_plies, so it moves with VGO_ACTORS. At the default 64
# the 15000 arm's shards came out at 15,710 +/- 50 against a request of 11,480,
# putting the real floor near 4,230 -- mean plies is closer to 66 than the 55
# this constant assumed. Do not correct it on a running arm: the number that
# has to stay fixed is the shard size, and changing either this or --actors
# mid-run changes it. Raising actors to 80 would add ~1,100 samples a shard,
# a 7% drift in the one variable the sweep is measuring.
floor=3520
requested=$(( size - floor ))
if [ "$requested" -lt 1 ]; then
  echo "$0: nominal size $size is at or below the drain floor $floor" >&2
  exit 2
fi

# Promotion gating. Both arms ran gated at 0.55 through their first twenty-odd
# updates; VGO_GATE=off continues one ungated, which promotes every candidate.
# That is allowed on a resume -- see the note beside OPERATIONAL_CONFIG_FIELDS
# in pipeline.py for why, and for the measurement saying an 8-pair arena cannot
# tell the two checkpoints apart. Keep the default on so this script still
# reproduces the gated updates it was used for.
gate=(--promotion-arena --promotion-score 0.55)
if [ "${VGO_GATE:-on}" = off ]; then
  gate=()
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${2:-$root/artifacts/shard-sweep-$size}"
mkdir -p "$out/logs"

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out, shard ~$size (requested $requested)"
if [ ${#gate[@]} -eq 0 ]; then
  echo "=== promotion gate OFF: every candidate becomes the generator"
else
  echo "=== promotion gate ON at 0.55 over 8 pairs"
fi

cd "$root/training"
exec .venv/bin/python3 -m vgo_training.rl_loop \
  --output "$out" \
  --updates "${VGO_UPDATES:-10}" \
  --samples-per-shard "$requested" --shards-per-update 1 --replay-window 6 \
  --resolution 128 --policy-resolution 128 --radius 0.055714285714285716 \
  --raster-kind compact --komi-low=-0.166 --komi-high=0.234 \
  --coarse-pool 16 --generation-simulations 1600 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 800 \
  --resign-window 5 --resign-minimum-ply 20 --resign-disable-fraction 0.0 \
  --training-epochs 1 --training-batch 256 \
  --learning-rate 0.001 --warm-learning-rate 0.001 --value-weight 2.0 \
  --ownership-weight 0.0 --recency-decay 1.0 --drain-tail \
  --concurrent-generators 1 \
  --schedule cosine --warmup-epochs 0 --compile --restore-optimizer \
  --architecture ddrnet --norm-groups 8 --model-width 64 --blocks 16 \
  --context-attention-blocks 1 --attention-heads 8 \
  --training-device cuda --training-threads "${VGO_TRAINING_THREADS:-4}" \
  --report-every 1 --validation-fraction 0.1 \
  --actors "${VGO_ACTORS:-64}" --arena-actors "${VGO_ARENA_ACTORS:-${VGO_ACTORS:-64}}" \
  --leaf-batch 4 \
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  "${gate[@]}" \
  --arena-pairs 8 --arena-simulations 256 \
  --seed $(( 31000000 + size )) --arena-seed $(( 31500000 + size ))
