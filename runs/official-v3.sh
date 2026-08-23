#!/usr/bin/env bash
# Reinforcement learning under voronoigo.com's rules, restarted from everything
# official-v2 produced: its best model and all forty of its shards, with a
# window wide enough to hold them.
#
#   ./runs/official-v3.sh                       # -> artifacts-official/official-v3
#   VGO_UPDATES=60 ./runs/official-v3.sh        # longer
#
# Run it from the project root. Relative output directories are pinned to the
# invoking shell below and the seed paths derive from it; those paths are part
# of run identity, so invoking from elsewhere makes the run unresumable.
#
# ## Why this run exists
#
# official-v2 lost ~110 Elo to the supervised model it was seeded from and never
# recovered. Three experiments located the cause, none of which needed new games:
#
#   - `vgo-control` reproduced the same loss under *our* rules, so the ruleset
#     was not it.
#   - Bulk-training the seed on official-v2's own forty shards tied the seed
#     (-7 Elo, 200 games), so the data was not it either.
#   - Replaying official-v2's updates offline at different window widths
#     (scripts/replay-window-ab.py) moved the result from -104 to +23 Elo on
#     identical data and identical starting weights and Adam moments.
#
# The window was the whole leak. Measured, all against the same seed:
#
#     width  6, 8 updates    -104      width 20, 20 updates    -38
#     width  6, 20 updates    -85      width 32, 8 updates     +23
#
# width 32 - width 6 is +126 Elo at 4.1 sigma; width 32 - width 20 is +61 at
# 2.0 sigma. Wider kept winning as far as the data could test.
#
# ## Why the window is 32 and not 40, and not 6
#
# 32 is the width the offline replay actually measured at +23 Elo against the
# supervised seed, +126 over width 6 (4.1 sigma) and +61 over width 20 (2.0).
#
# The first attempt at this run used 40 -- every seeded shard at once -- and
# thrashed the box to a standstill before update 0. Measured while it died:
#
#     generation (vgo-generate-demo, 64 actors)   21 GB
#     learner, 40-shard window                    22.6 GB and still climbing
#     desktop applications                        ~6 GB
#                                                 ---------------------------
#                                                 55+ GB of 60, swap exhausted
#
# The 28.7 GB figure for a 40-shard window was measured with generation *not*
# running, and that mislead cost three launches. The learner starts once per run
# and holds its ReplayCache for the whole of it, so its ~24 GB is resident even
# while it is idle -- that part does not go away. What can be removed is the
# overlap, so generation's 15 GB is not also resident during training.
#
# The second attempt used window 32 with 64 actors and was killed during
# update 0 with 56 of 60 GB used:
#
#     learner, 32-shard window   ~24 GB   resident for the entire run
#     generation, 64 actors      ~20 GB   resident whenever a shard is building
#     desktop applications        ~5 GB
#     zram swap holds compressed pages in RAM, so 60 GB is not all available
#
# --actors 48 buys back about 5 GB at roughly a quarter less generation
# throughput. That is the right side to cut: window width is the finding this
# run exists to exploit (+126 Elo at 4.1 sigma over width 6), and actors are
# only throughput. Lower VGO_ACTORS further if anything else is running.
#
# Cutting actors alone was still not enough -- a third attempt died in the same
# place. The kills all landed in one spot: update 0's *training*, while the next
# shard was generating. So the fix is --no-overlap-actor-learner, which puts
# generation, training and telemetry under one GPU lease and stops them
# coexisting:
#
#     during generation   learner idle holding its cache 24 + generation 15
#                         + desktop 7 + zram 3.5  =  ~50 GB   (observed stable)
#     during training     learner 24 + load transients + desktop 7 + zram 3.5
#                                                 =  ~35 GB   (was ~57, killed)
#
# It costs about 10% wall clock -- training is 2-3 minutes against generation's
# 25 -- and removes the only window where the two peaks add.
#
# The peak arrives at update 0, not update 39, because --initial-replay fills
# the window before the first update. That is worth knowing: an over-subscribed
# window fails in the first few minutes rather than after a night of work.
#
# Do not raise --policy-resolution without shrinking this. Replay RAM scales
# with the window, and the policy-shaped arrays scale with the square of that
# resolution.
#
# ## Why recency weighting is off
#
# `--recency-decay 0.93` was tried in the same offline replay and came in 35 Elo
# *below* uniform. There is no evidence for down-weighting old shards here, and
# two signals against it: the arm whose final window reached furthest back
# (shards 8..39) beat the one that only saw 20..39.
#
# ## What is seeded
#
# The checkpoint is the w32 replay's final model -- the strongest official-rules
# model measured, +23 Elo against the supervised seed. It carries optimizer
# state from a window-32 regime, which is the regime this run continues.
#
# The shards are *copies*. `--initial-replay` adopts the directories it is given
# and retirement compresses them in place, so pointing this at official-v2's
# replay would rewrite that run's data as housekeeping.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts-official/official-v3}"
case "$output" in
  /*) ;;
  *) output="$PWD/$output" ;;
esac

source_run="${VGO_SOURCE_RUN:-$root/artifacts-official/official-v2}"
seed_checkpoint="${VGO_SEED_CHECKPOINT:-$source_run/window-ab/w32/update-000039/candidate.pt}"
seed_onnx="${VGO_SEED_ONNX:-$source_run/window-ab/w32/final.onnx}"
updates="${VGO_UPDATES:-40}"
window="${VGO_WINDOW:-32}"

for required in "$seed_checkpoint" "$seed_onnx"; do
  if [ ! -e "$required" ]; then
    echo "missing seed: $required" >&2
    exit 1
  fi
done

if [ -e "$output/launch.sh" ] && [ ! -e "$output/.official-v3" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.official-v3"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

# Copy the source run's shards in, decompressing the retired ones. Skipped when
# already present so the run stays resumable.
seed_replay_root="$output/seed-replay"
mkdir -p "$seed_replay_root"
for shard in "$source_run"/replay/shard-*; do
  name=$(basename "$shard")
  destination="$seed_replay_root/$name"
  if [ ! -e "$destination/dataset.vgo" ]; then
    mkdir -p "$destination"
    cp "$shard/manifest.json" "$destination/manifest.json"
    [ -e "$shard/games.jsonl" ] && cp "$shard/games.jsonl" "$destination/games.jsonl"
    if [ -e "$shard/dataset.vgo" ]; then
      cp "$shard/dataset.vgo" "$destination/dataset.vgo"
    else
      zstd -dq "$shard/dataset.vgo.zst" -o "$destination/dataset.vgo"
    fi
  fi
done

# Enumerate the staging directory rather than the source glob, so shards placed
# here by hand are seeded too -- a run killed part way leaves finished shards
# worth keeping, and they are the freshest data available.
seed_replay=()
for shard in "$seed_replay_root"/shard-*; do
  [ -e "$shard/dataset.vgo" ] || continue
  seed_replay+=("$shard")
done
echo "seeded ${#seed_replay[@]} shards into $seed_replay_root"

if [ "${#seed_replay[@]}" -gt "$window" ]; then
  echo "note: ${#seed_replay[@]} shards seeded but window is $window;" \
       "the oldest will retire immediately" >&2
fi

python="$root/training/.venv/bin/python"

cd "$root/training"
exec "$python" -m vgo_training.rl_loop \
  --output "$output" \
  --updates "$updates" \
  --initial-checkpoint "$seed_checkpoint" \
  --initial-onnx "$seed_onnx" \
  --initial-replay "${seed_replay[@]}" \
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
  --samples-per-shard 1600 --shards-per-update 1 --replay-window "$window" \
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
  --actors "${VGO_ACTORS:-48}" --arena-actors "${VGO_ARENA_ACTORS:-${VGO_ACTORS:-48}}" \
  --leaf-batch 4 \
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --no-overlap-actor-learner --retire-shards \
  --arena-komi 0.104 --telemetry-pairs 5 \
  --seed 30200001 --arena-seed 30205001 \
  2>&1 | tee -a "$output/run.log"
