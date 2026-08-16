#!/usr/bin/env bash
# ddrnet-deep-search.sh continued from update 12, with the komi range narrowed
# from sigma 0.10 to sigma 0.03.
#
#   ./runs/ddrnet-deep-komi.sh                     # -> artifacts/ddrnet-deep-komi
#   VGO_UPDATES=60 ./runs/ddrnet-deep-komi.sh      # shorter
#   ./runs/ddrnet-deep-komi.sh artifacts/my-run    # somewhere else
#
# Run it from the project root. A relative output directory is pinned to the
# invoking shell below, and --initial-replay paths are derived from it; those
# paths are part of run identity.
#
# WHY
#
# A third of self-play games were being decided by the komi draw before a stone
# was placed. Measured over ddrnet-deep-search's 1231 games:
#
#   P(Black wins) = sigmoid(2.33 - 24.8 * komi)
#
# That is steep. The band where Black takes between 25% and 75% is 0.0885 wide,
# and the range being sampled was 0.40 -- about five times the width of the
# question. The result: 36% of games decided (>90% or <10% from the draw alone),
# 64% lopsided beyond 25/75, and only 36% genuinely contested. Deeper search
# cannot help a game that was never in doubt, so this is the largest remaining
# source of targets that carry no signal.
#
# WHY sigma 0.03 SPECIFICALLY
#
# komi is drawn from a normal truncated to [low, high] with sigma = width/4
# (generate_demo.rs:496 -- note the stale doc comment directly above it still
# says "uniformly", which it has not been for some time). So sigma 0.03 means a
# width of 0.12, spanning +/-2 sigma. Projected against the fitted curve, that
# gives 86% contested and 0% decided, against 36%/36% today. Tighter still --
# 0.10 wide -- would reach 92%, but the returns are flattening and the tails are
# what teach the komi channel the *relationship* rather than a constant. The
# ends are not wasted, only their current abundance.
#
# WHAT NARROWING COSTS
#
# The model stops seeing komi far from balance, so it gets worse at playing
# handicap-like positions. That is a deliberate trade for self-play strength and
# it is reversible: a later run can widen the range again, seeded from whatever
# this produces.
#
# It also removes 0.034 from the training distribution, which is what the
# tournaments used to be played at. That is handled by moving the tournaments:
# see the komi note in runs/deep-vs-komi.sh. Records now carry a "komi" field so
# the two scales cannot be pooled by accident.
#
# WHY THE CONFIGURED RANGE IS CENTRED AT 0.077 AND WHY IT BARELY MATTERS
#
# With replay seeded, _effective_komi_range (pipeline.py:1273) walks back to the
# newest manifest and takes *its* centre, so the configured range supplies only
# the width. The seeded shards carry centre +0.083, and the controller closes
# the rest at its 0.025/shard cap. The value below is written at the measured
# balance point anyway so the file says what was intended.
#
# The balance point is moving, which is worth knowing: fitted on the komi run's
# shards 130-144 it was +0.104, on all of deep-search +0.092, and on
# deep-search's last six shards +0.077 (95% CI [+0.067, +0.086]). Those last two
# do not overlap the first, so this is drift as the models improve rather than
# noise -- which is exactly why --dynamic-komi stays on. A narrower range does
# not weaken the controller's own fit: across +/-2 sigma the win rate still runs
# from about 82% to 18%, which is ample leverage for the 50% crossing.
#
# NO PROMOTION GATE
#
# The gate was removed from the pipeline entirely on 2026-08-16, so every
# candidate is now the next incumbent. It was a measurement with no power that
# cost real work when it fired: at --arena-pairs 8, a candidate exactly as
# strong as the incumbent promoted 40% of the time and one that truly scored 0.6
# was rejected 28% of the time, and because the incumbent is both the training
# parent and the generator, each rejection made the next update retrain from the
# same checkpoint and generate another shard from it. ddrnet-attn-komi rejected
# 60 of 83 overnight, advancing its lineage 23 times in 83 updates.
#
# Strength is measured after the fact instead. --telemetry-every 5 rates every
# fifth checkpoint against --telemetry-opponents 2 earlier ones at 16 pairs,
# which is 64 games per rated point and decides nothing. One caveat on what that
# actually samples: _queue_telemetry draws its opponents from *every* earlier
# accepted model by a seeded hash, not from the previous five, so the comparison
# graph gets long baselines as well as short ones. That is better for a curve
# and worse for "did the last five updates help"; read it with a
# Bradley-Terry fit over the whole run rather than pairwise.
#
# WHAT IS UNCHANGED
#
# Generation stays at 3200 simulations and soft resign at 1200, so this isolates
# the komi change against ddrnet-deep-search, which isolated the search change
# against ddrnet-attn-komi. That run bought +172 +/- Elo over its seed in twelve
# updates (artifacts/deep-vs-komi/RESULT.md), which is why the search budget is
# not being touched again in the same breath.
#
# Everything below --resolution is otherwise byte-identical to
# ddrnet-deep-search.sh apart from the komi range and the seeds. The three open
# questions recorded there -- promotion arena size, the 70-ply cap, and now the
# komi width -- are down to two.
#
# SIZING
#
# Unchanged from ddrnet-deep-search: about 20 minutes an update, so roughly 45
# updates in a 15-hour window. Narrowing komi does not change the cost per game;
# it changes what those games are worth.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)}"
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac

# Refuse to write into a run this recipe did not create. A recipe derives its
# output directory from its own filename, so naming one after an existing run
# points it at that run -- and the coordinator's identity check only fires after
# this script has copied in seed replay and overwritten launch.sh. That
# destroyed artifacts/ddrnet-deep's launch.sh on 2026-08-15, unrecoverably,
# since artifacts/ is gitignored.
if [ -f "$out/pipeline-config.json" ] && [ ! -d "$out/seed-replay" ]; then
  echo "refusing to write into $out: it holds another run's pipeline-config.json" >&2
  echo "pick a different output directory, or rename this recipe" >&2
  exit 1
fi

mkdir -p "$out/logs"

parent="$root/artifacts/ddrnet-deep-search"
parent_update="$parent/updates/update-000012"
# --initial-checkpoint below spells this path out as "$root/..." rather than
# reusing $parent_update: scripts/rate-checkpoints.py recovers the parent run by
# regex over this file (rate-checkpoints.py:98) and that pattern only understands
# a literal `root`; through a variable the lineage reads as "no parent".

# Update 12 is the parent's newest *accepted* model. Its updates/ directory goes
# to 13, and 13 was rejected -- taking the highest-numbered directory would seed
# from a candidate the run itself discarded.
seed="$out/seed-replay"
if [ ! -d "$seed" ]; then
  echo "=== seeding replay from $parent"
  mkdir -p "$seed"
  for shard in 000008 000009 000010 000011 000012 000013; do
    cp -r "$parent/replay/shard-$shard" "$seed/shard-$shard"
  done
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out"
echo "=== seeded from $parent_update, komi sigma 0.03 (width 0.12), 3200/1200"

cd "$root/training"
exec .venv/bin/python3 -m vgo_training.rl_loop \
  --output "$out" \
  --updates "${VGO_UPDATES:-240}" \
  --initial-checkpoint "$root/artifacts/ddrnet-deep-search/updates/update-000012/candidate.pt" \
  --initial-onnx "$root/artifacts/ddrnet-deep-search/updates/update-000012/candidate.onnx" \
  --initial-replay \
    "$seed/shard-000008" "$seed/shard-000009" "$seed/shard-000010" \
    "$seed/shard-000011" "$seed/shard-000012" "$seed/shard-000013" \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 6 \
  --resolution 128 --policy-resolution 128 --radius 0.055714285714285716 \
  --raster-kind compact --komi-low=0.017 --komi-high=0.137 \
  --dynamic-komi --komi-target-black-win-rate 0.5 \
  --komi-recenter-minimum-games 256 --komi-recenter-maximum-step 0.025 \
  --coarse-pool 16 --generation-simulations 3200 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 1200 \
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
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --telemetry-every 5 --telemetry-opponents 2 --telemetry-pairs 16 \
  --arena-pairs 8 --arena-simulations 256 \
  --seed 41100001 --arena-seed 41105001
