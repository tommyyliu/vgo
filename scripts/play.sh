#!/usr/bin/env bash
# Start the model move server for the JS client.
#
# The Rust binaries are built against ort with `load-dynamic`, so they dlopen
# libonnxruntime.so from ORT_DYLIB_PATH at runtime and need the venv's CUDA and
# TensorRT libraries on LD_LIBRARY_PATH. Without those they block in library
# loading with no output and no error -- the process simply appears to hang
# before it ever reaches the listen call. rl_loop sets this environment for the
# stages it spawns, which is why arenas work inside the loop and not outside it.
#
#   ./artifacts/play.sh                          # newest ddrnet-fast3 model
#   ./artifacts/play.sh path/to/candidate.onnx   # a specific model
#   SIMULATIONS=256 ./artifacts/play.sh          # stronger, slower
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/training/.venv/lib/python3.14/site-packages"

model="${1:-}"
if [[ -z "$model" ]]; then
  # Newest checkpoint of the newest run that has one, rather than a run named
  # here: a hardcoded path goes stale every time a run is superseded, and this
  # one had been pointing at a directory with no models in it.
  # Both layouts, newest first. Run separately and concatenated because an
  # unmatched glob is passed through literally, `ls` then exits 2, and under
  # `set -o pipefail` that kills the script before it prints anything.
  model="$( { ls -1t "$root"/artifacts/*/updates/update-*/candidate.onnx 2>/dev/null || true
              ls -1t "$root"/artifacts/*/updates/update-*/model/candidate.onnx 2>/dev/null || true
            } | head -1 )"
fi
if [[ -z "$model" || ! -f "$model" ]]; then
  echo "no model found; pass one explicitly: ./artifacts/play.sh <candidate.onnx>" >&2
  exit 1
fi

ort="$venv/onnxruntime_trt/libonnxruntime.so"
if [[ ! -f "$ort" ]]; then
  # Fall back to the CUDA-only build if the TensorRT one is not installed.
  ort="$venv/onnxruntime/capi/libonnxruntime.so"
fi
if [[ ! -f "$ort" ]]; then
  echo "libonnxruntime.so not found under $venv; is the training venv installed?" >&2
  exit 1
fi

export ORT_DYLIB_PATH="$ort"
export LD_LIBRARY_PATH="$venv/onnxruntime_trt:$venv/tensorrt_libs:$venv/nvidia/cu13/lib:$venv/nvidia/cudnn/lib:$venv/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Read the raster shape from the model rather than hardcoding it. The server
# validates its --resolution and --policy-resolution against the exported
# contract, so a stale default fails at load: these were 96/32 from the
# ddrnet-fast3 era while every model since ddrnet-wide is 128/128.
#
# The channel layout comes from the same place. It cannot be defaulted: a model
# trained on the five compact channels and fed the twelve semantic ones reads a
# different input than it learned from and plays blind.
read -r resolution policy_resolution raster_kind < <(
  "$root/training/.venv/bin/python3" - "$model" <<'PY'
import sys, onnx
metadata = {p.key: p.value for p in onnx.load(sys.argv[1]).metadata_props}
height = int(metadata["vgo.height"])
policy_size = int(metadata["vgo.policy_size"])
side = round((policy_size - 1) ** 0.5)
# Channel count identifies the layout; these are the three the rasterizer emits.
kinds = {3: "rgb", 5: "compact", 12: "semantic"}
channels = int(metadata["vgo.channels"])
if channels not in kinds:
    # Ten channels is the pre-settled semantic set, which the rasterizer no
    # longer emits -- Semantic grew to twelve when settled and komi were
    # added. Such a model cannot be served at all, by this script or
    # otherwise; it would have to be retrained or re-exported.
    sys.exit(
        f"model reports {channels} channels; the rasterizer emits "
        f"{sorted(kinds)} (rgb/compact/semantic). A ten-channel model predates "
        "the settled and komi channels and needs retraining."
    )
print(height, side, kinds[channels])
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
echo "simulations: ${SIMULATIONS:-1600}"
echo "first start builds a TensorRT engine for this model and takes ~30s."
echo

exec "$root/target/release/vgo-serve-move" \
  --model "$model" \
  --simulations "${SIMULATIONS:-1600}" \
  --coarse-pool "${COARSE_POOL:-$coarse_pool}" \
  --leaf-batch 4 \
  --resolution "$resolution" \
  --policy-resolution "$policy_resolution" \
  --raster-kind "$raster_kind" \
  --cache-directory "$root/artifacts/onnx-cache" \
  --address "${ADDRESS:-127.0.0.1:8181}"
