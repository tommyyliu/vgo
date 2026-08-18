#!/usr/bin/env bash
# Reinforcement learning under voronoigo.com's rules, seeded from the supervised
# compact-dead-zone model.
#
#   ./runs/official-finetune.sh                          # -> artifacts-official/official-v1
#   VGO_UPDATES=60 ./runs/official-finetune.sh           # longer
#   ./runs/official-finetune.sh artifacts-official/foo   # somewhere else
#
# Run it from the project root. A relative output directory is pinned to the
# invoking shell below, and the seed paths are derived from it; those paths are
# part of run identity, so invoking from elsewhere makes the run unresumable.
#
# Do not name an output directory after an existing run. This installs its own
# launch record there before anything else reads it, and doing that to a live
# run destroys the only copy of how that run was started -- artifacts roots are
# gitignored.
#
# ## Why a separate artifacts root
#
# `scripts/build-dense-curve.py` globs `artifacts/*/records.jsonl` and fits one
# Bradley-Terry scale over everything it finds. A run under different rules
# sitting inside `artifacts/` would be pooled into an Elo scale with models it
# has never played and, under these rules, could not play. `artifacts-official/`
# keeps that from happening by construction rather than by remembering.
#
# ## What is different about these rules
#
# Two things, both about capture, and docs/OFFICIAL_RULES.md has the derivations.
#
#   - A group lives while a stone could still be placed *touching* its territory,
#     rather than while a future stone could take area from it. That is strictly
#     the more aggressive rule: everything the official rules keep alive, ours
#     keeps alive, and ours keeps some alive that theirs captures.
#   - A move that would take only the mover's own stones is illegal. With no
#     self-capture there is no no-op placement, so the rule counting one as a
#     pass never fires here.
#
# The raster carries the matching capture field: `compact-dead-zone` is
# `compact-pass` with `dead_zone` in slot 3 instead of `settled`, so the network
# reads the predicate it is actually judged by.
#
# ## Why it is seeded rather than started cold
#
# A six-channel input cannot be warm-started into from a five-channel model, so
# the supervised A/B exists to produce one. `--initial-checkpoint` and
# `--initial-onnx` must be given together: on its own `--initial-replay` seeds
# only the training window and generation still starts from noise.
#
# ## The radius changes here
#
# `--radius` is 1/18 exactly, which is what the real game uses: an 18-unit board
# with stone radius 1. Every run so far used 39/700, 0.286% larger, which is not
# a game constant at all -- it is the reference client's radius slider sitting at
# its default of 39 pixels on a 700-pixel board, copied into the recipes from
# there. A run aimed at the official rules should be played on the official
# board, and this is the cheapest moment to stop carrying the artefact.
#
# It does mean the seed saw a slightly different board than this run plays on.
# That is a smaller discrepancy than the rules change it is already absorbing.
#
# ## One epoch per update, not ten
#
# Every working recipe in runs/ trains for a single epoch per update, and the
# first attempt at this run used ten -- carried over from the supervised A/B,
# where ten is right because the data is fixed and more passes are free.
#
# In the loop each update sees about 560 games, and ten passes over that with an
# 8.2M-parameter model memorises them. Measured on artifacts-official/official-v1
# at update 22: training value_mae 0.026 against validation 0.252, and validation
# policy_kl climbing 0.798 -> 0.888 -> 0.910 -> 0.952 across updates 0, 6, 12 and
# 22 while training loss fell throughout. Every update was rated below the seed
# it started from, ending 510 Elo down after 24 of them.
#
# The signature to watch for is that divergence, not the training curve, which
# looked healthy the whole way down.
#
# The seed learned from games played under *our* rules. It has seen the official
# capture field but never a game decided by it, so early updates are as much
# about unlearning as learning, and the first few arena results are not a
# verdict on anything.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts-official/official-v2}"
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

if [ -e "$output/launch.sh" ] && [ ! -e "$output/.official-finetune" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.official-finetune"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

python="$root/training/.venv/bin/python"

cd "$root/training"
exec "$python" -m vgo_training.rl_loop \
  --output "$output" \
  --updates "$updates" \
  --initial-checkpoint "$seed_checkpoint" \
  --initial-onnx "$seed_onnx" \
  ${seed_replay:+--initial-replay $seed_replay} \
  --ruleset official \
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
  --arena-komi 0.104 --telemetry-pairs 5 \
  --seed 30100001 --arena-seed 30105001 \
  2>&1 | tee -a "$output/run.log"
