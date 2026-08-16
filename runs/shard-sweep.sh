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
# On the strength result, corrected 2026-08-14. This header used to claim that
# at fixed total data bigger shards are worse by -190 +/- 53 Elo (3.6 sigma).
# Treat that as withdrawn:
#
#   * No rating data for the 5k or 10k arms survives anywhere in the repo.
#     ratings.json and both dense-curve records.jsonl files contain only
#     ddrnet-fresh-attn, shard-sweep-15000 and naive, and none of the sweep runs
#     has completed telemetry. The figure cannot be reproduced from what is here.
#   * The only cross-run pairing with real games (1,456 of them) is
#     ddrnet-fresh-attn against shard-sweep-15000, which differs in optimizer
#     (Adam vs Muon) *and* gating (0.55 vs ungated) as well as shard size. At
#     matched total data that pairing runs -30 to -222 Elo across updates 6-30
#     and passes through -192 at update 24, which is close enough to -190 to
#     suspect the original number came from it.
#   * Read off the curves, the two Muon arms that actually isolate shard size --
#     5.1k and 10.7k -- overlap substantially, with 5.1k perhaps slightly ahead.
#
# So the honest state is: no measured shard-size effect on the Muon arm, and the
# Adam arm untested. Do not quote a number here without a fit behind it.
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
# It is actors x mean_plies, so it moves with VGO_ACTORS. Measured 2026-08-14
# from the manifest `samples` of every shard on disk, at the default 64 actors:
#
#   arm                request   produced        implied floor
#   ddrnet-fresh-attn    1,600   5,455 +/- 291   3,855
#   shard-sweep-10000    6,480  10,248 +/- 354   3,768
#   shard-sweep-15000   11,480  15,016 +/- 353   3,536
#
# 3520 is therefore about right, and this comment's previous correction -- that
# the 15000 arm produced 15,710 +/- 50, putting the floor near 4,230 at ~66 mean
# plies -- does not reproduce. Measured mean plies is 52-56 across all three.
#
# Do not correct the constant on a running arm regardless: the number that has
# to stay fixed is the shard size, and changing either this or --actors mid-run
# changes it. Raising actors to 80 would add ~1,100 samples a shard, a 7% drift
# in the one variable the sweep is measuring.
floor=3520
requested=$(( size - floor ))
if [ "$requested" -lt 1 ]; then
  echo "$0: nominal size $size is at or below the drain floor $floor" >&2
  exit 2
fi

# Promotion gating. Both arms ran gated at 0.55 through their first twenty-odd
# updates. The gate was removed from the pipeline on 2026-08-16 -- it rejected
# roughly a quarter of genuine improvements and stalled the generator on a stale
# checkpoint for a shard each time it fired -- so resuming either arm now
# publishes every candidate. VGO_GATE is gone with it; the updates already in
# those runs were produced gated and their pipeline-state records which.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${2:-$root/artifacts/shard-sweep-$size}"
# Pin a relative argument to the invoking shell's directory; --output would
# otherwise resolve it against `cd "$root/training"` below and start a second
# run there. training/artifacts/shard-sweep-15000 is what that looks like.
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
mkdir -p "$out/logs"

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out, shard ~$size (requested $requested)"
echo "=== ungated: every candidate becomes the generator"

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
