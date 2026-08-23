# The environment the pipeline builds for Rust binaries, for ad-hoc runs.
#
# Without ORT_DYLIB_PATH an ONNX binary hangs silently -- it looks exactly like
# a slow TensorRT engine build -- and without the rest the TensorRT provider
# cannot find libcublas.
#
# Lives in the repo rather than a scratch directory because /tmp is tmpfs here:
# a reboot takes every helper script with it, and the failure mode is a probe
# that reports zeros instead of an error.
#
#   source scripts/env/ort.sh
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for sp in "$root"/training/.venv/lib64/python3.14/site-packages \
          "$root"/training/.venv/lib/python3.14/site-packages; do
  [ -d "$sp" ] || continue
  export LD_LIBRARY_PATH="$sp/onnxruntime/capi:$sp/tensorrt_libs:$sp/nvidia/cu13/lib:$sp/nvidia/cudnn/lib:$sp/torch/lib:${LD_LIBRARY_PATH:-}"
  [ -n "${ORT_DYLIB_PATH:-}" ] || ORT_DYLIB_PATH=$(ls "$sp"/onnxruntime/capi/libonnxruntime.so.* 2>/dev/null | tail -1)
done
export ORT_DYLIB_PATH
