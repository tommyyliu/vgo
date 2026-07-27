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
  model="$(ls -1 "$root"/artifacts/ddrnet-fast3/iteration-*/model/candidate.onnx 2>/dev/null | sort | tail -1)"
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

echo "model:       ${model#"$root/"}"
echo "simulations: ${SIMULATIONS:-128}"
echo "first start builds a TensorRT engine for this model and takes ~30s."
echo

exec "$root/target/release/vgo-serve-move" \
  --model "$model" \
  --simulations "${SIMULATIONS:-128}" \
  --coarse-pool 4 \
  --leaf-batch 4 \
  --resolution 96 \
  --policy-resolution 32 \
  --cache-directory "$root/artifacts/onnx-cache" \
  --address "${ADDRESS:-127.0.0.1:8181}"
