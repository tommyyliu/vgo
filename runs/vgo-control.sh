#!/usr/bin/env bash
# The control for artifacts-official/official-v2, changing exactly one thing.
#
#   ./runs/vgo-control.sh                       # -> artifacts-control/vgo-control
#   VGO_UPDATES=60 ./runs/vgo-control.sh
#
# ## What this is for
#
# official-v2 got significantly worse: update 36 lost 69-131 to the seed it was
# trained from over 200 games, Elo -111 with a 95% interval of [-162, -61]. That
# run changed four things at once against ddrnet-deep-komi, the last run that
# worked:
#
#     ruleset       vgo     -> official
#     raster_kind   compact -> compact-dead-zone
#     radius        39/700  -> 1/18
#     replay window seeded  -> cold
#
# So "did the official rules cause it?" is not answerable from that run. This
# one is identical to official-v2 in every respect but the ruleset.
#
#   - If this also regresses, the rules are exonerated and the cause is among
#     the raster, the radius, or the cold start. That rules out a lot.
#   - If this improves, the cause is specific to the official rules, and the
#     leading candidate becomes what happens to refused self-capture moves.
#
# It is deliberately *not* seeded and deliberately keeps the same radius and
# raster as official-v2, even though both are suspects and seeding is known to
# help. Fixing them here would confound the comparison, which is the mistake
# that made official-v2 uninterpretable.
#
# Telemetry is the one exception: 16 pairs every 5 updates rather than 5 every
# update. That changes what can be *seen*, not what is learned -- official-v2's
# 10-game matches could not resolve anything under ~150 Elo, which is how a
# regression ran for 37 updates while its ratings looked like noise around zero.
#
# ## Reading the result
#
# Not from telemetry. Run a 200-game arena against the same seed
# (artifacts/raster-ab/compact-dead-zone/candidate.onnx) at komi 0.104, radius
# 1/18, threads 1 -- the setup in the official-v2 measurement -- and compare the
# interval against [-162, -61].
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts-control/vgo-control}"
case "$output" in
  /*) ;;
  *) output="$PWD/$output" ;;
esac

seed_checkpoint="${VGO_SEED_CHECKPOINT:-$root/artifacts/raster-ab/compact-dead-zone/candidate.pt}"
seed_onnx="${VGO_SEED_ONNX:-$root/artifacts/raster-ab/compact-dead-zone/candidate.onnx}"
updates="${VGO_UPDATES:-40}"

# Shards to fill the replay window with before the first update, space
# separated. Without them the window ramps 1, 2, 3... and the earliest updates
# train on a fraction of a normal one: measured on official-v2, the four updates
# with a partial window rated -178, -77, -125 and -147 against the seed while
# every later update scattered around zero.
#
# Copy the shards in rather than pointing at another run's replay directory.
# Retirement compresses a shard and deletes the uncompressed original, so
# `--initial-replay` aimed at a live run rewrites that run's data as
# housekeeping.
#
# They must be *official-rules* shards. Seeding from a vgo-rules run would put
# games in the window whose policy targets include self-captures, which are
# illegal here -- teaching the net to propose moves the rules refuse. That is
# also why there was nothing to seed the first run with, and why the second one
# can be seeded from the first.
seed_replay="${VGO_SEED_REPLAY:-}"

for required in "$seed_checkpoint" "$seed_onnx"; do
  if [ ! -e "$required" ]; then
    echo "missing seed: $required" >&2
    echo "run ./runs/raster-ab.sh first; it produces both" >&2
    exit 1
  fi
done

if [ -e "$output/launch.sh" ] && [ ! -e "$output/.vgo-control" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.vgo-control"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

python="$root/training/.venv/bin/python"

cd "$root/training"
exec "$python" -m vgo_training.rl_loop \
  --output "$output" \
  --updates "$updates" \
  --initial-checkpoint "$seed_checkpoint" \
  --initial-onnx "$seed_onnx" \
  ${seed_replay:+--initial-replay $seed_replay} \
  --ruleset vgo \
  --raster-kind compact-dead-zone \
  --architecture ddrnet --norm-groups 8 --model-width 64 --blocks 16 \
  --context-attention-blocks 1 --attention-heads 8 \
  --resolution 128 --policy-resolution 128 \
  --radius 0.05555555555555555 \
  --coarse-pool 16 --generation-simulations 3200 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 1200 \
  --resign-window 5 --resign-minimum-ply 20 --resign-disable-fraction 0.0 \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 6 \
  --komi-low 0.017 --komi-high 0.137 \
  --dynamic-komi --komi-target-black-win-rate 0.5 \
  --komi-recenter-minimum-games 256 --komi-recenter-maximum-step 0.025 \
  --training-epochs 1 --training-batch 256 \
  --learning-rate 0.001 --warm-learning-rate 0.001 --value-weight 2.0 \
  --ownership-weight 0.0 --recency-decay 1.0 --drain-tail \
  --concurrent-generators 1 \
  --schedule cosine --warmup-epochs 0 --compile --restore-optimizer \
  --full-adam \
  --training-device cuda --training-threads "${VGO_TRAINING_THREADS:-4}" \
  --report-every 1 --validation-fraction 0.1 \
  --actors "${VGO_ACTORS:-64}" --arena-actors "${VGO_ARENA_ACTORS:-${VGO_ACTORS:-64}}" \
  --leaf-batch 4 \
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --arena-komi 0.104 --telemetry-pairs 16 --telemetry-every 5 \
  --seed 30100001 --arena-seed 30105001 \
  2>&1 | tee -a "$output/run.log"
