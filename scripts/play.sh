#!/usr/bin/env bash
# Start the model move server for the JS client.
#
# The Rust binaries are built against ort with `load-dynamic`, so they dlopen
# libonnxruntime.so from ORT_DYLIB_PATH at runtime and need the venv's CUDA and
# TensorRT libraries on LD_LIBRARY_PATH. Without those they block in library
# loading with no output and no error -- the process simply appears to hang
# before it ever reaches the listen call. scripts/env/ort.sh sets it up.
#
#   ./scripts/play.sh                          # newest model of any run
#   ./scripts/play.sh path/to/candidate.onnx   # a specific model
#   SIMULATIONS=256 ./scripts/play.sh          # stronger, slower
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model="${1:-}"
if [[ -z "$model" ]]; then
  # Newest checkpoint of the newest run that has one, rather than a run named
  # here: a hardcoded path goes stale every time a run is superseded, and this
  # one had been pointing at a directory with no models in it.
  # Three layouts now: the continuous loop writes `<run>/models/update-N.onnx`,
  # while the older pipeline wrote `<run>/updates/update-N/candidate.onnx` with
  # or without a `model/` level.
  #
  # Collected first and sorted afterwards, because concatenating separate `ls -t`
  # runs sorts within each glob and not across them -- whichever pattern matched
  # first won regardless of age, which is how this came to serve a model from a
  # superseded run while a much newer one sat in `models/`. Unmatched globs are
  # passed through literally and make `ls` exit non-zero, hence the `|| true`.
  mapfile -t candidates < <(
    ls -1 "$root"/artifacts/*/models/update-*.onnx \
          "$root"/artifacts/*/updates/update-*/candidate.onnx \
          "$root"/artifacts/*/updates/update-*/model/candidate.onnx \
          2>/dev/null || true
  )
  #
  # Read into an array rather than piping to `head`. With this many candidates
  # `head` closes the pipe after one line, `ls` takes SIGPIPE, and `pipefail`
  # plus `set -e` kills the script with 141 and no output at all -- it exits
  # before printing so much as the model it chose.
  if (( ${#candidates[@]} > 0 )); then
    mapfile -t newest < <(ls -1t "${candidates[@]}" 2>/dev/null)
    model="${newest[0]:-}"
  fi
fi
if [[ -z "$model" || ! -f "$model" ]]; then
  echo "no model found; pass one explicitly: ./scripts/play.sh <candidate.onnx>" >&2
  exit 1
fi

source "$root/scripts/env/ort.sh"
if [[ -z "${ORT_DYLIB_PATH:-}" || ! -f "${ORT_DYLIB_PATH}" ]]; then
  # Refuse rather than proceed: a failed dlopen deadlocks in ort::api() instead
  # of returning an error, so the server would hang silently before listening.
  echo "libonnxruntime not found via the training venv (ORT_DYLIB_PATH=${ORT_DYLIB_PATH:-unset})." >&2
  echo "Is it installed?  cd training && uv sync --frozen --extra tensorrt" >&2
  exit 1
fi

# Read the raster shape from the model rather than hardcoding it. The server
# validates its --resolution and --policy-resolution against the exported
# contract, so a stale default fails at load: these were 96/32 from the
# ddrnet-fast3 era while every model since ddrnet-wide is 128/128.
#
# The channel layout comes from the export manifest beside the model, not from
# the channel count. Counting channels stopped identifying a layout when
# `compact-pass` and `compact-dead-zone` arrived: both are six planes, and they
# differ in the capture predicate, so guessing between them is a coin flip that
# fails silently at inference -- the shapes match, and the net simply reads a
# plane that means something else. `export_onnx` records the name in
# `<model>.json` for exactly this reason.
#
# Models exported before that field existed leave it null. Every one of those
# here is five-channel `compact`, which the count still resolves, so the count
# stays as a fallback -- but it refuses an ambiguous width rather than picking.
#
# RASTER_KIND overrides both, for a model whose manifest is missing or wrong.
read -r resolution policy_resolution raster_kind < <(
  "$root/training/.venv/bin/python3" - "$model" "${RASTER_KIND:-}" <<'PY'
import json, sys
from pathlib import Path

import onnx

model_path = Path(sys.argv[1])
override = sys.argv[2] or None

metadata = {p.key: p.value for p in onnx.load(model_path).metadata_props}
height = int(metadata["vgo.height"])
policy_size = int(metadata["vgo.policy_size"])
side = round((policy_size - 1) ** 0.5)
channels = int(metadata["vgo.channels"])

# Plane count per layout, mirroring RasterKind::channels in vgo-raster. Kept in
# sync by the check below: a layout whose width stops matching the model is
# refused rather than served.
widths = {
    "rgb": 3,
    "compact": 5,
    "compact-pass": 6,
    "compact-dead-zone": 6,
    "compact-connected": 9,
    "compact-radius": 7,
    "semantic": 12,
}
by_width: dict[int, list[str]] = {}
for name, width in widths.items():
    by_width.setdefault(width, []).append(name)

kind, source = override, "RASTER_KIND"
if kind is None:
    manifest_path = Path(f"{model_path}.json")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            manifest = {}
        kind = manifest.get("model", {}).get("raster_kind")
        source = manifest_path.name
if kind is None:
    candidates = by_width.get(channels, [])
    if len(candidates) == 1:
        kind, source = candidates[0], "channel count"
    elif candidates:
        sys.exit(
            f"{channels} channels is ambiguous between "
            f"{' and '.join(candidates)}, and {model_path.name}.json does not "
            "name one. Re-export the model, or set RASTER_KIND to the layout "
            "it was trained on."
        )
    else:
        # Ten channels is the pre-settled semantic set, which the rasterizer no
        # longer emits -- Semantic grew to twelve when settled and komi were
        # added. Such a model cannot be served at all, by this script or
        # otherwise; it would have to be retrained or re-exported.
        sys.exit(
            f"model reports {channels} channels; the rasterizer emits "
            f"{', '.join(sorted(widths))}. A ten-channel model predates the "
            "settled and komi channels and needs retraining."
        )

if kind not in widths:
    sys.exit(f"unsupported raster kind {kind!r} (from {source})")
if widths[kind] != channels:
    sys.exit(
        f"{source} says {kind}, which is {widths[kind]} planes, but the model "
        f"takes {channels}. A model fed the wrong layout plays blind; refusing."
    )
print(height, side, kind)
PY
)
# The coarse pool is a search setting, not part of the model contract, so it
# cannot be read back. 16 is what every 128-policy run trained against; a model
# searched with a different pool than it learned under plays worse.
coarse_pool=16
if (( policy_resolution <= 32 )); then
  coarse_pool=4
fi

echo "model:       ${model#"$root/"}"
echo "raster:      ${resolution}x${resolution} ${raster_kind}  policy ${policy_resolution}x${policy_resolution}  coarse-pool ${COARSE_POOL:-$coarse_pool}"
echo "simulations: ${SIMULATIONS:-1600}  widening ${WIDENING_COEFFICIENT:-6.0} cap ${MAXIMUM_CANDIDATES:-321}"
echo "first start builds a TensorRT engine for this model and takes ~30s."
echo

exec "$root/target/release/vgo-serve-move" \
  --model "$model" \
  --simulations "${SIMULATIONS:-1600}" \
  --coarse-pool "${COARSE_POOL:-$coarse_pool}" \
  --leaf-batch 4 \
  --widening-coefficient "${WIDENING_COEFFICIENT:-6.0}" \
  --maximum-candidates "${MAXIMUM_CANDIDATES:-321}" \
  --resolution "$resolution" \
  --policy-resolution "$policy_resolution" \
  --raster-kind "$raster_kind" \
  --cache-directory "$root/artifacts/onnx-cache" \
  --address "${ADDRESS:-127.0.0.1:8181}"
