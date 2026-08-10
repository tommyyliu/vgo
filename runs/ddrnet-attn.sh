#!/usr/bin/env bash
# The strongest configuration measured so far: DDRNet w64/b16, one attention
# block per context stage, plain Adam, 1600 simulations, cold start.
#
#   ./runs/ddrnet-attn.sh                      # -> artifacts/ddrnet-attn
#   ./runs/ddrnet-attn.sh artifacts/my-run     # somewhere else
#   VGO_UPDATES=100 ./runs/ddrnet-attn.sh      # longer
#   VGO_ACTORS=48 ./runs/ddrnet-attn.sh        # smaller box
#
# Why --full-adam is spelled out. The run this is ported from
# (artifacts/ddrnet-fresh-attn) predates the flag: its pipeline-config.json has
# no full_adam key at all, because when it ran, Adam was the only option.
# The default is now Muon-on-trunk, so replaying that script verbatim today
# would quietly train a different optimizer than the one that produced these
# results. This is the single most important line in the file.
#
# Why these values, from the measurements that chose them:
#   w64/b16       w96 memorises -- train value MAE 0.018 against 0.247
#                 validation -- while w64 generalises better and is half the size
#   1 attention   one transformer block per context stage generalised better on
#                 value and generated ~8% faster; it did not win an arena
#                 outright (32 games, 17-15), so this is a bet on the cheaper
#                 model, not a demonstrated strength gain
#   Adam          over 40 updates Adam ended ~112 Elo ahead of Muon. Muon learns
#                 value much faster early (it reaches at update 8 what Adam
#                 needs 24 for) but then has no headroom left, and its value
#                 head overfits its window at 2.6x Adam's rate
#   1600 sims     cheaper per shard than 2400, and the search-time candidate
#                 panic seen at 2400 has not reproduced below it
#
# Only settings in the pipeline's OPERATIONAL_CONFIG_FIELDS are exposed as
# environment variables. Everything else is part of the run's identity: the
# coordinator refuses to resume a run whose identity config changed, so
# parameterising one of those would silently make the run unresumable.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)}"
mkdir -p "$out/logs"

# Copy the recipe in beside the run. scripts/rate-checkpoints.py reads
# <run>/launch.sh to follow --initial-checkpoint lineage, and it is also the
# only record of what a run actually was once the flags scroll away.
install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

# Without this the coordinator's narrative -- per-stage timings, the retire
# messages, the resignation calibration table printed after every publication --
# exists only on a terminal nobody is watching. Appended, so it survives resume.
exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out"

cd "$root/training"
exec .venv/bin/python3 -m vgo_training.rl_loop \
  --output "$out" \
  --updates "${VGO_UPDATES:-60}" \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 6 \
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
  --full-adam \
  --training-device cuda --training-threads "${VGO_TRAINING_THREADS:-4}" \
  --report-every 1 --validation-fraction 0.1 \
  --actors "${VGO_ACTORS:-64}" --arena-actors "${VGO_ARENA_ACTORS:-${VGO_ACTORS:-64}}" \
  --leaf-batch 4 \
  --inference-batch 64 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --promotion-arena --promotion-score 0.55 \
  --arena-pairs 8 --arena-simulations 256 \
  --seed 21000001 --arena-seed 21005001
