#!/usr/bin/env bash
# Train the same model twice, once per six-plane raster, and change nothing else.
#
#   ./runs/raster-ab.sh                    # -> artifacts/raster-ab
#   VGO_EPOCHS=20 ./runs/raster-ab.sh      # longer
#   ./runs/raster-ab.sh artifacts/my-ab    # somewhere else
#
# Run it from the project root; a relative output directory is pinned to the
# invoking shell below.
#
# Do not name an output directory after an existing run. This script writes its
# own launch record into that directory before anything else looks at it, and
# doing so to a live run's directory destroys the record of how that run was
# started -- artifacts/ is gitignored, so there is no copy.
#
# ## What this compares
#
#   compact-pass       current_stones, opponent_stones, voronoi_ridge,
#                      settled, komi, previous_pass
#   compact-dead-zone  the same, with `dead_zone` in place of `settled`
#
# Slot 3 is the capture predicate and is the only difference: `settled` is this
# repository's rule -- a group lives while some future stone can still take area
# from it -- and `dead_zone` is voronoigo.com's, where a group lives while a
# stone could still be placed touching its territory. The second is strictly
# more aggressive; measured on real shards it covers 47.9% of the board against
# settled's 44.1%.
#
# Every other plane is bit-identical between the two, which is what makes this a
# one-plane A/B rather than a comparison of representations, and what lets a
# model warm-start from either into the other later.
#
# ## What this is not
#
# **This is supervised training on existing data, not a rules experiment.** The
# shards were generated under this repository's rules, so a model trained here
# on `compact-dead-zone` has seen the official rules' *capture field* but has
# never seen a game played under the official rules. What it settles is whether
# the encoding trains at all and which one fits the data better; playing csun's
# rules is the RL segment after this, and needs vgo-core taught his capture rule
# and his self-capture restriction first.
#
# It is also the seed for that segment. A new input width cannot be warm-started
# into, so the RL run needs a checkpoint *and* an exported ONNX at six channels
# before it can generate anything -- which is what this produces.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/artifacts/raster-ab}"
case "$output" in
  /*) ;;
  *) output="$PWD/$output" ;;
esac

epochs="${VGO_EPOCHS:-12}"
seed="${VGO_SEED:-1}"

# Adam on everything, not the Muon/Adam hybrid. Muon leads early and loses:
# over a full RL loop it was ahead until update 24 and Adam had overtaken by the
# end, which an offline A/B did not predict -- it spends its value headroom
# early. This is a supervised run short enough to sit entirely inside the window
# where Muon looks better, which is exactly why it should not be used here.
#
# Matches artifacts/ddrnet-deep-komi, the run that produced the current model, so
# the comparison is against a known architecture rather than the CLI's defaults.
width="${VGO_WIDTH:-64}"
blocks="${VGO_BLOCKS:-16}"
attention="${VGO_ATTENTION:-1}"

if [ -e "$output/launch.sh" ] && ! [ -e "$output/.raster-ab" ]; then
  echo "refusing: $output looks like another run's directory" >&2
  exit 1
fi
mkdir -p "$output"
touch "$output/.raster-ab"
install -m 0755 "${BASH_SOURCE[0]}" "$output/launch.sh"

# The renderer is a build artifact the Python loader shells out to, and its
# absence surfaces as a ValueError deep inside the first shard load rather than
# as a missing binary.
cargo build --release -p vgo-raster --example render_shard

# The newest two runs. Positions, not pictures, so the raster below is a
# training-time choice and these shards serve both arms unchanged.
shards=()
for run in ddrnet-deep-komi ddrnet-deep-search; do
  while IFS= read -r shard; do
    shards+=("$shard")
  done < <(find "$root/artifacts/$run/replay" -name dataset.vgo | sort)
done
if [ "${#shards[@]}" -eq 0 ]; then
  echo "no shards found under artifacts/{ddrnet-deep-komi,ddrnet-deep-search}/replay" >&2
  exit 1
fi
echo "training on ${#shards[@]} shards"

# The venv, explicitly. scripts/train-once.py's shebang finds the system
# interpreter, which on this box has a different torch entirely.
python="$root/training/.venv/bin/python"

for kind in compact-dead-zone; do
  arm="$output/$kind"
  mkdir -p "$arm"
  echo "=== $kind ==="
  # Same seed, same data, same architecture. The raster is the only difference.
  "$python" "$root/scripts/train-once.py" "${shards[@]}" \
    --output "$arm/candidate.pt" \
    --raster-kind "$kind" \
    --epochs "$epochs" \
    --seed "$seed" \
    --model-width "$width" \
    --blocks "$blocks" \
    --context-attention-blocks "$attention" \
    --architecture ddrnet \
    --value-weight 2.0 \
    --learning-rate 0.001 \
    --full-adam \
    2>&1 | tee "$arm/train.log"

  # Export now rather than later: the manifest records `raster_kind`, and a
  # serving path fed the wrong layout fails silently -- the channel count cannot
  # tell these two apart.
  "$python" -m vgo_training.export_onnx \
    --checkpoint "$arm/candidate.pt" \
    --output "$arm/candidate.onnx" \
    > "$arm/export.log" 2>&1
  echo "exported $arm/candidate.onnx"
done

echo
echo "done. Compare:"
echo "  grep -h 'val' $output/*/train.log | tail -20"
echo "  python3 -c \"import json;[print(k, json.load(open(f'$output/'+k+'/candidate.onnx.json'))['model']['raster_kind']) for k in ['compact-pass','compact-dead-zone']]\""
