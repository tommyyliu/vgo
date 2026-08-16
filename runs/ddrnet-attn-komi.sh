#!/usr/bin/env bash
# ddrnet-attn.sh continued from ddrnet-fresh-attn update 59, with the dynamic
# komi controller enabled.
#
#   ./runs/ddrnet-attn-komi.sh                       # -> artifacts/ddrnet-attn-komi
#   VGO_UPDATES=60 ./runs/ddrnet-attn-komi.sh        # longer
#   ./runs/ddrnet-attn-komi.sh artifacts/my-run      # somewhere else
#
# Run it from the project root. A relative output directory is pinned to the
# invoking shell below, and --initial-replay paths are derived from it; those
# paths are part of run identity, so invoking from a different directory with a
# different relative path would make the run unresumable.
#
# Why this is a new run rather than a continuation of ddrnet-fresh-attn.
# `dynamic_komi` is not in OPERATIONAL_CONFIG_FIELDS, and ddrnet-fresh-attn's
# pipeline-config.json predates the flag entirely, so compatible_config()
# backfills it to false. Turning it on is therefore an identity change and the
# coordinator refuses to resume -- deliberately; see the note at pipeline.py:220.
# Seeding from its final checkpoint keeps the 11.6 hours already spent while
# giving the controller a run it is allowed to steer.
#
# Why update 59. It is the run's final model and also its best rated checkpoint
# (1349 Elo), though 54 and 57 sit within the ~±59 measurement error, so this is
# "the newest one, which happens to lead" rather than a meaningful selection.
#
# Why the seed shards are copied instead of referenced. Retirement compresses a
# shard once it leaves the window and deletes the uncompressed original
# (pipeline.py:119). Pointed at the parent's replay directory, this run would
# rewrite ddrnet-fresh-attn's last six shards as a side effect of its own
# housekeeping. 147 MB of copies is the cost of not reaching into another run.
#
# Why --komi-low/--komi-high are unchanged even though the fit says +0.092.
# With replay seeded, _effective_komi_range (pipeline.py:1273) walks back to the
# newest manifest and takes its center -- the parent's +0.0340 -- so the
# configured range supplies only the width. Recentering it here would be inert.
# The controller closes the gap itself at its 0.025/shard cap, about three
# shards, which is the same delay a cold-centered range would have paid waiting
# for its first 256 eligible games. The measurement it is closing: over the
# parent's last six shards, Black won 383/584 (65.6%) and fit_komi_balance puts
# the 50% crossing at +0.0920.
#
# Why the replay window is seeded at all. Without it the first updates would
# train a model accustomed to a 31k-sample window on a single 5.2k shard. The
# seeded shards carry sequences -6..-1 and consumed_through_shard starts at -1,
# so none of them is pending: the first update still waits for fresh data and
# then trains on [-5..-1, 0]. They age out normally over the next six shards.
#
# Everything from --resolution down is byte-identical to ddrnet-attn.sh. The
# seeds are not: they are shifted so this run's games are independent of the
# parent's rather than replaying its schedule against a different model.
#
# Resignation stays at 0.02/800/0.0, which is what this run has generated every
# shard with. Do not change it here. All three are identity fields, so editing
# them does not reconfigure the run -- it makes the coordinator refuse to resume
# it, and the 62 updates already measured at +2.13 +/- 0.33 Elo/update would
# have to be abandoned to adopt them.
#
# This is not hypothetical. These lines were edited to 0.03/400/0.1 on
# 2026-08-14 at 02:10:27, 53 seconds after the run had already launched at
# 02:09:35. The running job never saw the change, and the next resume attempt
# failed with "learning configuration differs from the existing run" against an
# edit nobody remembered making. A recipe for a live run is a record of that
# run, not a place to stage the next experiment.
#
# The measurement that motivated the change, kept for whatever run does adopt
# it. Pooled over the six seed shards plus shard 0 (681 games, 37308 samples)
# at window 5, the counterfactual reads:
#
#   thresh  fired  wrong  w/fired  plies_saved
#   0.9      520     15     2.9%       31%
#   0.95     441      9     2.0%       21%
#   0.98     336      0     0.0%       10%
#
# The selector takes the lowest threshold clearing the target, so 0.02 buys
# 0.95-0.98 and 0.03 would buy 0.9. Soft simulations at 400 rather than 800 is
# the other half: `plies_saved` is a discount, not a skip, so search actually
# reclaimed is plies_saved * (1 - soft/generation) -- 21% * 0.5 = 10% here
# against 31% * 0.75 = 23% there. --resign-disable-fraction 0.1 would go with
# them: every soft-resigned game calibrates, but after a concession both sides
# play on at the reduced count, so a 0% error rate at high firing is partly the
# rule agreeing with a playout it shaped -- the ddrnet-resign failure mode. The
# exempt tenth is the only slice searched at full strength start to finish.
#
# Adopting all of that means a new run seeded from this one's best checkpoint,
# the same way this run was seeded from ddrnet-fresh-attn update 59.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/artifacts/$(basename "${BASH_SOURCE[0]}" .sh)}"
# Pin a relative argument to the invoking shell's directory; --output would
# otherwise resolve it against `cd "$root/training"` below and start a second
# run there. See the same guard in ddrnet-attn.sh.
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
mkdir -p "$out/logs"

parent="$root/artifacts/ddrnet-fresh-attn"
parent_update="$parent/updates/update-000059"
# Note that --initial-checkpoint below spells this path out as "$root/..."
# rather than reusing $parent_update. scripts/rate-checkpoints.py recovers the
# parent run by regex over this file (rate-checkpoints.py:98) and that pattern
# only understands a literal `root`; through a different variable the lineage
# silently reads as "no parent" and the child gets rated as if it started cold.

# Copied once. On a resume the state file already carries these shards and
# --initial-replay is ignored (pipeline.py only reads it when creating state),
# so re-copying would at best be wasted work and at worst restore a
# dataset.vgo this run has since retired to .zst.
seed="$out/seed-replay"
if [ ! -d "$seed" ]; then
  echo "=== seeding replay from $parent (147 MB)"
  mkdir -p "$seed"
  for shard in 000054 000055 000056 000057 000058 000059; do
    cp -r "$parent/replay/shard-$shard" "$seed/shard-$shard"
  done
fi

install -m 755 "${BASH_SOURCE[0]}" "$out/launch.sh"

exec > >(tee -a "$out/logs/run.log") 2>&1
echo "=== $(date -Is) starting, output $out"
echo "=== seeded from $parent_update, dynamic komi on"

cd "$root/training"
exec .venv/bin/python3 -m vgo_training.rl_loop \
  --output "$out" \
  --updates "${VGO_UPDATES:-40}" \
  --initial-checkpoint "$root/artifacts/ddrnet-fresh-attn/updates/update-000059/candidate.pt" \
  --initial-onnx "$root/artifacts/ddrnet-fresh-attn/updates/update-000059/candidate.onnx" \
  --initial-replay \
    "$seed/shard-000054" "$seed/shard-000055" "$seed/shard-000056" \
    "$seed/shard-000057" "$seed/shard-000058" "$seed/shard-000059" \
  --samples-per-shard 1600 --shards-per-update 1 --replay-window 6 \
  --resolution 128 --policy-resolution 128 --radius 0.055714285714285716 \
  --raster-kind compact --komi-low=-0.166 --komi-high=0.234 \
  --dynamic-komi --komi-target-black-win-rate 0.5 \
  --komi-recenter-minimum-games 256 --komi-recenter-maximum-step 0.025 \
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
  --inference-batch 32 --inference-delay-ms 1 \
  --inference-slots "${VGO_SLOTS:-2}" \
  --provider tensorrt --fp16 --warm-inference \
  --overlap-actor-learner --retire-shards \
  --arena-pairs 8 --arena-simulations 256 \
  --seed 21100001 --arena-seed 21105001
