#!/usr/bin/env bash
# Multi-radius RL: train one network to play every board voronoigo.com offers.
#
# Every model so far plays exactly one board size, and it is the smallest one.
# `voronoigo.com` serves 18 "mini", 26 "midi" and 38 "standard" units across --
# radii of 1/18, 1/26 and 1/38 here -- and its matchmaking defaults to standard.
# Every run in this repository trained at 1/18. The bot we shared plays the
# beginner board and has never seen the one people actually play on.
#
# ## What changes
#
# `--raster-kind compact-radius`, seven planes: `compact-pass` plus the radius.
# The plane is not optional. An empty board renders identically at every radius
# under six planes -- no stones, so no ridge and no settled region, and the two
# scalars say nothing about scale -- so a net opening on an unseen board is
# guessing which game it is in. Seven channels cannot load six-channel weights,
# so this is a cold start; nothing before it can seed it.
#
# `--board-mix 50:38 25:18 25:18-38`. Half the games on the board people play,
# a quarter on the one we have always trained, a quarter spread between them.
# Ranges sample uniformly in *units*: uniform radius puts density as 1/units^2
# and would pile the wide band into its smallest boards.
#
# Nothing above 38 units, because nothing above 38 units exists. voronoigo.com
# serves 18, 26 and 38; a band running to 50 spent the highest per-game cost in
# the mix on boards no one plays. It also set the tail: the ply cap scales as
# 1/r^2, so 50 units allowed 541 plies against 38's 312, and one unfinished
# 541-ply game holds a whole shard open.
#
# Nothing below 18 units. Small boards are a different game -- living is hard,
# single sequences decide everything, and Go's fair komi across 5x5 to 9x9 runs
# 25, 4, 9, 7 points, non-monotonic, because of it. Those patterns do not
# transfer, and the komi law below is fitted above that regime.
#
# ## Komi, which cannot be one number any more
#
# Komi compensates about one stone's worth of area and a board holds ~1/r^2
# stones, so komi as a fraction of the board goes as r^2. Three things fix the
# coefficient at 33.7 and agree with each other: our own measurement of 0.104 at
# r = 1/18; Go's 9x9 komi of 8.6% of the board against our 10.4% at nearly the
# same stone capacity; and Go's 9x9-to-19x19 exponent of 1.89. It predicts 2.3%
# at standard, where Go's 19x19 sits at 2.1%.
#
# It is a prior, not a law. `fit_komi_power_law` re-estimates the coefficient
# *and* the exponent from real games once enough have accumulated.
#
# ## Resolution, decoupled
#
# `--resolution 256 --policy-resolution 128`. They cost different things: model
# compute tracks the raster (256 is 3.95x, and policy adds 2-5% on top), while
# `FineGrid` caches logits and legality per policy cell at about one node per
# simulation, so generation *memory* tracks policy alone -- 21 GB at 128, 84 GB
# at 256, against 60 GB of RAM.
#
# Raster 256 gives the standard board 3.37 trunk cells per stone and 0.84 in the
# context branch, matching what the mini board gets today at 128 (3.56 and
# 0.89). Holding policy at 128 keeps memory where it is. At policy 128 legal
# placements on the standard board still sit ~6.7 cells apart, so the grid can
# name every distinct move.
#
# ## Batch 64, because 256 does not fit
#
# Activation memory goes as the raster area, so 256 square needs four times what
# 128 did and the old batch of 256 wants ~31 GB on a 15.5 GB card. Measured at
# this raster: 4.1 GB at batch 32, 7.9 at 64, 11.8 at 96, OOM at 128. Batch 64
# lands where batch 256 sat at the old raster.
#
# The learning rate halves with it. Four times the steps at four times smaller
# batches would move four times as far at the same rate; linear scaling says
# quarter it, the usual Adam heuristic says halve it, and this takes the
# heuristic. That is a choice, not a measurement -- if the first updates diverge,
# this is the first thing to look at.
#
# ## Ply sampling, because the value head would starve
#
# A standard game runs past three hundred plies. Shards are sized in positions,
# so 1600 of them would be about five games -- and the value head learns from
# game-level labels only. `--ply-sample-rate 0.25` restores the game count at
# the same shard size, and decorrelates the window besides: consecutive plies
# are nearly the same position carrying nearly the same gradient.
#
# The ply cap scales with the board for the same reason it exists: 70 plies is
# about the mini board's capacity, and left there a standard game is cut off a
# fifth of the way in with a truncation artifact for a label.
#
# ## Two generators, because one leaves the machine idle
#
# A shard ends when its slowest game does. Measured here: 57 minutes to reach
# the sample target with 32 actors busy, then 65 minutes of tail with *two*
# actors busy and thirty cores idle -- half the wall clock at 6% utilisation.
#
# `--concurrent-generators 2` starts the next shard while the current one
# drains, which is what the field was added for: measured 4.16 -> 5.49 samples/s
# when the tail was 24% of wall time. Ours was 50%, so the headroom is larger.
# Actors halve to 16 each so the machine still runs 32 in total.
#
# `--maximum-prefetch-shards 2` with it, or the first flag does nothing. The
# scheduler gates on `prefetched + in_flight < maximum_prefetch_shards`, so a
# limit of 1 caps in-flight shards at one however many generator slots are
# allowed -- and the symptom is not an error, it is a second generator that
# never starts and a load average that quietly sits at half the actor count.
#
# This is a workaround and not a fix. The tail exists because shards are batch
# boundaries and a batch ends with its slowest member; continuous generation --
# actors writing games as they finish, training on the last N games -- removes
# it rather than overlapping it. That is the next thing to build, against a
# baseline this produces.
#
# ## What it costs, measured rather than estimated
#
# 32 actors on the real mix produced 60 samples from 4 games in 329 s, which is
# 2.4 hours a shard at 6400 simulations -- six days for sixty updates. Measured
# after launching twice on an estimate, which is the wrong order.
#
# `--generation-simulations 1600` rather than 6400 buys that back fourfold, to
# roughly 35 minutes a shard. Three things say it is the right knob rather than
# a concession: search width past the current setting was measured at +64 Elo
# and simulations are the same currency; the value head is starved of *games*,
# which this quadruples per hour; and on a cold start the prior is random, so
# deep search on it is the least valuable search there is. Raise it once the
# policy is worth searching.
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts/vgo-multiradius}"
case "$output" in
  /*) ;;
  *) output="$PWD/$output" ;;
esac

# No seed. Seven channels cannot load six-channel weights and every shard on
# disk is six-channel, so there is nothing to warm-start from: this begins at
# random initialisation. Expect the first ten updates to look like a broken run
# for reasons that have nothing to do with multi-radius.
updates="${VGO_UPDATES:-60}"

if [ -e "$output/launch.sh" ] && [ ! -e "$output/.vgo-multiradius" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.vgo-multiradius"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

python="$root/training/.venv/bin/python"
cd "$root/training"
exec "$python" -m vgo_training.rl_loop \
  --output "$output" \
  --updates "$updates" \
  --ruleset vgo \
  --raster-kind compact-radius \
  --architecture ddrnet --norm-groups 8 --model-width 64 --blocks 16 \
  --context-attention-blocks 1 --attention-heads 8 \
  --resolution 256 --policy-resolution 128 \
  --radius 0.05555555555555555 \
  --coarse-pool 16 --generation-simulations 1600 \
  --widening-coefficient 4.0 --maximum-candidates 321 \
  --root-exploration-noise 0.0 \
  --board-mix 50:38 --board-mix 25:18 --board-mix 25:18-38 \
  --ply-sample-rate 0.25 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 2400 \
  --resign-window 5 --resign-minimum-ply 20 --resign-disable-fraction 0.0 \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 32 \
  --komi-low 0.017 --komi-high 0.137 \
  --dynamic-komi --komi-target-black-win-rate 0.5 \
  --komi-recenter-minimum-games 256 --komi-recenter-maximum-step 0.025 \
  --training-epochs 1 --training-batch 64 \
  --learning-rate 0.0005 --warm-learning-rate 0.0005 --value-weight 2.0 \
  --ownership-weight 0.0 --recency-decay 1.0 --drain-tail \
  --concurrent-generators 2 --maximum-prefetch-shards 2 \
  --schedule cosine --warmup-epochs 0 --compile --restore-optimizer \
  --full-adam \
  --training-device cuda --training-threads "${VGO_TRAINING_THREADS:-4}" \
  --report-every 1 --validation-fraction 0.1 \
  --actors "${VGO_ACTORS:-16}" --arena-actors "${VGO_ARENA_ACTORS:-32}" \
  --leaf-batch 4 \
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --no-overlap-actor-learner --retire-shards \
  --arena-komi 0.104 --telemetry-pairs 5 \
  --seed 42100001 --arena-seed 42105001 \
  2>&1 | tee -a "$output/run.log"
