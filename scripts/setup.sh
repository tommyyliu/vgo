#!/usr/bin/env bash
# Take a fresh machine from a clean clone to a runnable one.
#
# Installs both toolchains, builds the Rust binaries, and verifies the pieces
# that fail confusingly rather than loudly. It stops there: it does not
# provision anything, hold credentials, or start a run.
#
#   ./scripts/setup.sh          # set up and verify
#   ./scripts/setup.sh --check  # verify only, install nothing
#
# On onnxruntime: the stock `onnxruntime-gpu` wheel is sufficient, including on
# Blackwell/sm_120. docs/NVRTX_HANDOFF.md describes building it from source with
# a userspace CUDA toolchain -- that was necessary for onnxruntime 1.24 and is
# not necessary now. Do not follow it on a new box.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
check_only=false
[[ "${1:-}" == "--check" ]] && check_only=true

step () { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok ()   { printf '    ok   %s\n' "$1"; }
warn () { printf '    warn %s\n' "$1"; }
die ()  { printf '    FAIL %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- the GPU
step "GPU"
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found. This needs an NVIDIA driver; the CUDA toolkit is not required."
# Split on the comma, not on whitespace: the GPU name contains spaces.
IFS=',' read -r gpu_name gpu_memory gpu_capability < <(
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1
)
gpu_name="${gpu_name#NVIDIA }"
gpu_memory="${gpu_memory// /}"
gpu_capability="${gpu_capability// /}"
ok "$gpu_name, ${gpu_memory} MiB, compute capability $gpu_capability"
# The tuned --actors/--inference-batch defaults assume a 16 GB card; a smaller
# one runs but wants those lowered, and finding that out mid-run is expensive.
if (( gpu_memory < 15000 )); then
  warn "under 16 GB: lower --actors and --inference-batch, or generation will OOM"
fi

# The job is CPU-bound -- generation saturates the CPU while the GPU sits near
# 60% -- so core count, not GPU class, sets throughput. Say so here because the
# natural instinct when renting is to buy GPU.
cores=$(nproc)
ok "$cores logical cores"
if (( cores < 16 )); then
  warn "generation is CPU-bound; under 16 cores this box will be slow regardless of GPU"
fi

# ------------------------------------------------------------ the toolchains
step "Python toolchain"
if ! command -v uv >/dev/null 2>&1; then
  $check_only && die "uv not found (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
if ! $check_only; then
  # --frozen so the lockfile decides, not the resolver; uv fetches Python 3.14
  # itself, so the host needs no system Python of any particular version.
  ( cd training && uv sync --frozen --extra tensorrt )
fi
[[ -x training/.venv/bin/python3 ]] || die "training/.venv missing; run without --check"
ok "venv at training/.venv ($(training/.venv/bin/python3 --version 2>&1))"

step "Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
  $check_only && die "cargo not found (https://rustup.rs)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
fi
ok "cargo $(cargo --version | awk '{print $2}') (rust-toolchain.toml pins the version)"

# The pipeline shells out to `cargo run --release` for every generation, arena
# and warmup stage rather than calling prebuilt binaries, so cargo has to be on
# PATH at run time too -- not just here.

# torch.compile is on by default and needs a C compiler for inductor. Minimal
# cloud images often ship without one, and the failure surfaces inside the first
# training step rather than at startup.
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || die "no C compiler. torch.compile needs one; install build-essential/gcc, or pass --no-compile."
ok "C compiler present"

# zstd is only used by shard retirement, and _retire_aged_shards swallows its
# failures (pipeline.py) -- so a box without it does not crash, it silently
# stops reclaiming disk. Warn rather than fail, but say what the consequence is.
if command -v zstd >/dev/null 2>&1; then
  ok "zstd $(zstd --version | grep -oP 'v\K[0-9.]+' | head -1)"
else
  warn "zstd not found: --retire-shards will fail silently and replay disk will grow unbounded"
fi

# ------------------------------------------------------------------ the build
if ! $check_only; then
  step "Build"
  # Building now rather than letting the first update do it: .cargo/config.toml
  # sets -C target-cpu=native, so this must happen on the machine that runs it,
  # and a cold cache would otherwise compile in the middle of the first shard.
  cargo build --release
fi
for binary in vgo-generate-demo vgo-arena vgo-tournament vgo-serve-move; do
  [[ -x "target/release/$binary" ]] || die "target/release/$binary missing; run without --check"
done
ok "release binaries present"

# ------------------------------------------------------------- the ORT dlopen
step "ONNX Runtime"
# This is the check worth having. The Rust side dlopens libonnxruntime from
# ORT_DYLIB_PATH, and a *failed* load hangs instead of erroring -- the error is
# built through ort::api(), which waits on the initialization lock the failing
# load still holds. The process then sits at 0% CPU with a few MB resident,
# indistinguishable from a deadlock. Catching it here turns hours of confusion
# into one line.
eval "$(
  training/.venv/bin/python3 -c "
import shlex, sys
sys.path.insert(0, 'training')
from vgo_training.pipeline import runtime_environment
environment = runtime_environment()
for key in ('ORT_DYLIB_PATH', 'LD_LIBRARY_PATH'):
    value = environment.get(key)
    if value:
        print(f'export {key}={shlex.quote(value)}')
"
)"
[[ -n "${ORT_DYLIB_PATH:-}" && -f "${ORT_DYLIB_PATH}" ]] \
  || die "ORT_DYLIB_PATH did not resolve (${ORT_DYLIB_PATH:-unset}); is the venv synced?"
ok "${ORT_DYLIB_PATH#"$root/"}"
providers=$(training/.venv/bin/python3 -c \
  "import onnxruntime; print(','.join(onnxruntime.get_available_providers()))")
ok "providers: $providers"
[[ "$providers" == *Tensorrt* ]] \
  || warn "no TensorRT provider; pass --provider cuda, as the pipeline defaults to tensorrt and aborts without it"

step "Ready"
cat <<'EOF'
    Next: prove the box end to end before committing hours to a run.

      ./scripts/smoke.sh

    Then launch a run (see runs/ for recipes):

      ./runs/ddrnet-attn.sh artifacts/my-run
EOF
