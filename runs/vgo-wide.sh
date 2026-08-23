#!/usr/bin/env bash
# Reinforcement learning under our own rules, with everything the search and
# window work turned up.
#
#   ./runs/vgo-wide.sh                    # -> artifacts/vgo-wide
#   VGO_UPDATES=60 ./runs/vgo-wide.sh     # longer
#
# Run it from the project root; relative outputs are pinned to the invoking
# shell and the seed paths derive from it, so invoking elsewhere makes the run
# unresumable.
#
# ## What is different, and why
#
# Four measured changes, each of which cost a night to establish.
#
# `--replay-window 32` against the old 6. Replaying official-v2's updates
# offline at different widths, from the same checkpoint and the same Adam
# moments, moved the result from -104 to +23 Elo; live, a window-32 run held
# level over 40 updates where the window-6 one sat 111 Elo below its own seed.
# The window is seeded full, because a run that ramps 1, 2, 3... trains its
# earliest updates on a fraction of an update and never recovers.
#
# `--widening-coefficient 4.0` against SearchConfig's 2.0. Same model both
# seats, only this differing: +232 Elo at 800 simulations and +183 at 3200.
# The gain arrives entirely between 2.0 and 4.0 and then plateaus out to at
# least 32.0, so this is the cheap end of a wide basin rather than a peak that
# has to be hit. Below it the fall is steep -- 1.0 is -338, 0.5 is -708.
# This one constant beat every model-side lever combined: optimizer choice 0,
# schedule length 0, 4x capacity 0.
#
# `--maximum-candidates 321` because the ceiling can only hold widening back,
# and at 6400 simulations the formula asks for exactly that. It also sizes the
# replay record: 321 draws collapse to ~184 distinct cells, and v7 writes a
# capacity of 224 to match, at 2688 bytes a record against the old 2560.
#
# `--root-exploration-noise` is 0.0. It was 0.15 for this run's first three
# shards and then turned off, deliberately: it is the one change here with no
# measurement behind it, and leaving it on would have made any later result
# unattributable between three simultaneous changes. Turn it back on once the
# window and widening changes have a baseline to be compared against.
#
# The reasoning for wanting it stands. Candidates are sampled from the policy,
# so a placement it rates near zero is never proposed, never searched, and never
# enters the target -- from which the next policy learns the same blind spot.
# That is a fixed point a self-play loop can sit in indefinitely, and flooring
# the rate is what Dirichlet noise does for AlphaZero. What is unknown is the
# price: a noisy seat lost 4-8 in a twelve-game check, which is the expected
# sign, and no arena can show the target diversity it is supposed to buy.
#
# The original note, for when it goes back on: Candidates are sampled from the policy, so a placement it rates
# near zero is never proposed, never searched, and never enters the target --
# from which the next policy learns the same blind spot. This floors that rate
# the way Dirichlet noise does for AlphaZero. AlphaZero uses 0.25 over a few
# hundred enumerated moves; ours spreads uniform mass over ~844 effective
# cells, so the same epsilon is much weaker here and 0.15 is a judgement
# between the two. **Treat it as untested**: it is known to cost playing
# strength (a noisy seat lost 4-8 in a twelve-game check, which is the expected
# sign) and is hoped to pay for that in target diversity, which no arena can
# show. If this run disappoints, this is the first thing to turn off.
#
# ## Actors, and why 32
#
# Measured at these settings (scripts/actor-scaling.sh), 500-sample probes:
#
#     actors  samples/s   RSS     power    games per 1000 samples
#         24       1.30   16.2 GB  147 W                     17.2
#         48       1.56   31.7 GB  164 W                     17.8
#         72       1.69   47.2 GB  172 W                     17.6
#
# Marginal returns halve at every step: +0.26 samples/s for the first extra 24
# actors, +0.13 for the next, each costing 15.5 GB. Three times the actors buys
# 1.3x the throughput. Power tops out at 172 W of 300, so the GPU is not the
# limit -- with 32 cores and a 16,385-cell fine grid built per node, this is
# CPU-bound, and 32 actors is where that naturally tops out.
#
# Note what does *not* change: games per 1000 samples is flat across the range.
# Higher actor counts produce proportionally larger shards, not game-richer
# ones, because --drain-tail overshoots by whatever is in flight. Nothing here
# helps the value head, which learns from game-level labels and is the head
# this data serves worst.
#
# Each node caches that fine grid, and a tree holds about one node per
# simulation, so generation RAM scales with `actors * simulations`. At 32
# actors and 6400 simulations that is ~21 GB, against ~12 GB for the packed
# window and ~5 GB of desktop: about 38 of 60 GB, leaving real headroom. The
# machine became unstable at 55 GB earlier, which is the reason to bank it
# rather than spend it on the last few percent.
#
# ## Ruleset
#
# Ours, not voronoigo.com's, and compact-pass rather than compact-dead-zone.
# The two rulesets diverge rarely enough to be worth deferring: the official
# capture rule took 0 additional groups across 2,512 dead cells, and generation
# produced 0 self-captures in 3,641 games. Finetuning to the official rules is
# cheaper later than carrying two tracks now.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts/vgo-wide}"
case "$output" in
  /*) ;;
  *) output="$PWD/$output" ;;
esac

seed_checkpoint="${VGO_SEED_CHECKPOINT:-$root/artifacts/raster-ab/compact-pass/candidate.pt}"
seed_onnx="${VGO_SEED_ONNX:-$root/artifacts/raster-ab/compact-pass/candidate.onnx}"
updates="${VGO_UPDATES:-40}"

for required in "$seed_checkpoint" "$seed_onnx"; do
  [ -e "$required" ] || { echo "missing seed: $required" >&2; exit 1; }
done

if [ -e "$output/launch.sh" ] && [ ! -e "$output/.vgo-wide" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.vgo-wide"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

# The shards the seed model was trained on, copied in rather than referenced:
# retirement compresses a shard in place and deletes the original, so pointing
# --initial-replay at another run's replay directory rewrites that run's data.
seed_replay_root="$output/seed-replay"
mkdir -p "$seed_replay_root"
seed_replay=()
for source_run in ddrnet-deep-komi ddrnet-deep-search; do
  for shard in "$root"/artifacts/$source_run/replay/shard-*; do
    [ -e "$shard/dataset.vgo" ] || continue
    name="$source_run-$(basename "$shard")"
    destination="$seed_replay_root/$name"
    if [ ! -e "$destination/dataset.vgo" ]; then
      mkdir -p "$destination"
      cp "$shard/manifest.json" "$destination/manifest.json"
      [ -e "$shard/games.jsonl" ] && cp "$shard/games.jsonl" "$destination/games.jsonl"
      cp "$shard/dataset.vgo" "$destination/dataset.vgo"
    fi
  done
done
for shard in "$seed_replay_root"/*/; do
  [ -e "$shard/dataset.vgo" ] && seed_replay+=("${shard%/}")
done
echo "seeded ${#seed_replay[@]} shards into $seed_replay_root"

python="$root/training/.venv/bin/python"
cd "$root/training"
exec "$python" -m vgo_training.rl_loop \
  --output "$output" \
  --updates "$updates" \
  --initial-checkpoint "$seed_checkpoint" \
  --initial-onnx "$seed_onnx" \
  --initial-replay "${seed_replay[@]}" \
  --ruleset vgo \
  --raster-kind compact-pass \
  --architecture ddrnet --norm-groups 8 --model-width 64 --blocks 16 \
  --context-attention-blocks 1 --attention-heads 8 \
  --resolution 128 --policy-resolution 128 \
  --radius 0.05555555555555555 \
  --coarse-pool 16 --generation-simulations 6400 \
  --widening-coefficient 4.0 --maximum-candidates 321 \
  --root-exploration-noise 0.0 \
  --temperature 1.0 --temperature-plies 30 --maximum-plies 70 \
  --resign-target-false-positive 0.02 --resign-soft-simulations 2400 \
  --resign-window 5 --resign-minimum-ply 20 --resign-disable-fraction 0.0 \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 32 \
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
  --actors "${VGO_ACTORS:-32}" --arena-actors "${VGO_ARENA_ACTORS:-${VGO_ACTORS:-32}}" \
  --leaf-batch 4 \
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --no-overlap-actor-learner --retire-shards \
  --arena-komi 0.104 --telemetry-pairs 5 \
  --seed 40100001 --arena-seed 40105001 \
  2>&1 | tee -a "$output/run.log"
