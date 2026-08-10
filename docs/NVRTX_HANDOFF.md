# Blackwell (sm_120) GPU inference — hand-off notes

> **Obsolete for setup. Do not follow this to provision a machine.**
>
> Everything below describes building onnxruntime from source because the
> prebuilt wheel had no sm_120 kernel image. That was true of onnxruntime
> 1.24 and is **no longer true**: the stock `onnxruntime-gpu` wheel pinned in
> `training/pyproject.toml` (1.28.0 at time of writing) ships working
> TensorRT and CUDA execution providers for sm_120. Verified 2026-08-10 on
> the same RTX 5070 Ti — full RL loops, 528-game tournaments, and generation
> all run on `--provider tensorrt` with zero failures and no custom build.
>
> To set up a machine: **`./scripts/setup.sh`**, then `./scripts/smoke.sh`.
> That is the whole procedure. Following the recipe below instead costs
> several hours assembling a userspace CUDA toolchain, a patched gcc-15 and
> TensorRT SDK headers to reproduce a library you already have.
>
> Kept because `runtime_environment()` still prefers `onnxruntime_trt/` and
> `onnxruntime_blackwell/` when present, so those names appear in stack
> traces and in this history — and because the `fuse_conv_bias` finding below
> is still live and still load-bearing.

Status updated 2026-07-24; command syntax refreshed 2026-07-27. Goal was "run
the RL loop for a few steps" on this machine (Linux / Fedora 44, RTX 5070 Ti /
Blackwell / sm_120). **RESOLVED — the RL loop runs end-to-end on `--provider
cuda`.** This records the GPU path. The measurements below predate the current
pipelined coordinator, persistent learner, and final coarse-to-fine replay-v3
design; use [`RL_LOOP.md`](RL_LOOP.md) as the operational source of truth.

## TL;DR (historical — see the notice above)

The prebuilt onnxruntime that the Rust `ort` crate downloaded **in 2026-07** had
**no sm_120 kernel image**, so `cuda`/`tensorrt` failed with
`cudaErrorNoKernelImageForDevice` and TensorRT-RTX (`nvrtx`) failed at engine
build. The fix at the time (per microsoft/onnxruntime#26177, confirmed on this
exact GPU) was to **build onnxruntime from source with
`CMAKE_CUDA_ARCHITECTURES=120`** and load it via `ort`'s `load-dynamic`. The
`load-dynamic` switch is still how inference works; the from-source build is
not needed on onnxruntime ≥ 1.28.

After that, GPU inference produced garbage (`invalid inference value` NaN in the
arena). **Root cause: `with_fuse_conv_bias(true)` in `cuda_provider()`
(`crates/vgo-inference/src/onnx.rs`).** On this build the fused conv+bias CUDA
kernel corrupts state across `Run()` calls: the first inference is correct, then
every subsequent call on the same session compounds (output maxabs 5 -> 103 ->
455 -> 815 -> 1279 -> ... -> NaN). MCTS does thousands of evals per session, so
it overflowed to NaN. **Fix: drop `with_fuse_conv_bias` (one line).** The other
CUDA options (`tf32`, `conv_max_workspace`) are fine. CPU parity confirms it:
`vgo-onnx-bench --compare-python` now reads ~1e-3 (was ~4000).

## What works now

- `training/.venv` has a userspace CUDA 13.0 toolchain (no sudo). See the
  `vgo-blackwell-cuda-toolchain` memory for the exact package pins and patches.
- onnxruntime v1.24.2 built from source with sm_120 kernels; the libs are
  installed at `training/.venv/lib/python3.14/site-packages/onnxruntime_blackwell/`
  (`libonnxruntime.so*`, `libonnxruntime_providers_cuda.so`,
  `libonnxruntime_providers_shared.so`).
- `crates/vgo-inference/Cargo.toml` uses `ort` with `default-features = false`
  and `features = ["load-dynamic", "std", "ndarray", "tracing", "api-24"]`. The
  crate dlopens `libonnxruntime.so` from `ORT_DYLIB_PATH` at runtime; EPs
  register through that library. (`api-24` is required — with `load-dynamic` the
  VitisAI EP path compiles and needs the `SessionOptionsAppendExecutionProvider_VitisAI`
  binding, which only exists at `api-18`+.)
- `rl_loop.runtime_environment()` now, on Linux, puts the onnxruntime + CUDA +
  cuDNN + torch lib dirs on `LD_LIBRARY_PATH` and sets `ORT_DYLIB_PATH` for every
  child process (both overridable from the environment).
- Verified: `vgo-arena --provider cuda` allocates ~2.4 GB of GPU memory and runs
  inference on the GPU. `nvidia-smi` shows nonzero utilization.

## Classic TensorRT works on Linux and is the fast path

TensorRT-RTX (`nvrtx`) failed at engine build, but **classic TensorRT
(`--provider tensorrt`) works** and is much faster than the plain CUDA provider.
It was never the blocker — only TRT-RTX was; they do not share the failure.

- Build: add `--use_tensorrt --tensorrt_home <TRT>` alongside `--use_cuda`. The
  SDK headers came from the public `NVIDIA/TensorRT` GitHub tag `v10.16` (sparse
  checkout of `include/`), layered over the `tensorrt-cu13-libs==10.16.1.11`
  runtime libs we already had. `TENSORRT_ROOT` = `include/` (headers) + `lib/`
  (the runtime `.so`s, with unversioned symlinks). Libs installed to
  `training/.venv/.../site-packages/onnxruntime_trt/` (a superset — also carries
  the CUDA provider). `runtime_environment()` prefers `onnxruntime_trt` and adds
  `tensorrt_libs` to `LD_LIBRARY_PATH`.
- Correct: parity vs Torch = policy 0.006 / value 0.0005 at fp16 (matches the
  Windows TensorRT parity). Full RL loop runs end-to-end on `--provider tensorrt`
  (a historical two-update validation, ~90s including one-time engine builds,
  zero failures, both candidates promoted).
- Fast: isolated 128x128 fp16 throughput (positions/s), same RTX 5070 Ti:

  | batch | Windows TensorRT | Linux CUDA EP | Linux TensorRT |
  | --- | --- | --- | --- |
  | 1  | 3022 | 2921 | 7423 |
  | 8  | 5136 | 5522 | 14419 |
  | 16 | 5732 | 4660 | 13696 |
  | 32 | 5747 | 3237 | 11877 |

  Linux TensorRT is ~2.4-2.8x the CUDA EP and scales up with batch instead of
  regressing. **Prefer `--provider tensorrt` for throughput; `cuda` still works.**

## The CUDA correctness bug (RESOLVED): fused conv+bias corrupts across calls

After the from-source build, `vgo-arena --provider cuda` failed with
`EvaluationError { "invalid inference value" }` (from
`crates/vgo-inference/src/protocol.rs` — the `Tanh` value head returned a
non-finite number), and `vgo-onnx-bench --compare-python` showed policy logits
off by ~4000 vs the Torch checkpoint while CPU matched to ~1e-6.

**Root cause: `with_fuse_conv_bias(true)` in `cuda_provider()`.** The fused
conv+bias kernel in this onnxruntime build keeps state that is not reset between
`Run()` calls. On a fresh session the first inference is correct; every call
after that compounds. Running one session repeatedly on the same input:

| call | output maxabs |
| --- | --- |
| 0 | 4.97 (correct) |
| 1 | 103.7 |
| 2 | 455.2 |
| 3 | 815.8 |
| 4 | 1279.0 |

MCTS does thousands of evaluations per game on one session, so it ran away to
inf/NaN — hence the arena's `invalid inference value`. Isolating the CUDA
options one at a time showed the drift comes *only* from `fuse_conv_bias`;
`tf32` and `conv_max_workspace` are stable. **Fix: drop `with_fuse_conv_bias`
(done).** After the fix: arena `failures: 0`, score matches CPU; parity ~1e-3
(normal fp32 GPU/CPU rounding on a barely-trained model with near-tied logits).

### How it was found (methodology, for the next weird GPU bug)
The fast oracle is `vgo-onnx-bench --compare-python` (ONNX vs the Torch
checkpoint; ~1e-3 when correct, thousands when broken). The decisive step was a
throwaway probe that runs a model on a chosen provider and prints the output sum
each call — running inference *5 times on one session* exposed the per-call
drift that single-shot tests hid. Key dead-ends that were ruled out first (all
correct in isolation): a single Conv, Conv→ReduceMean→Gemm, batch 1 vs 16,
random vs fixture input, opt level, tf32, max-workspace, and PyTorch's own CUDA
conv (bit-exact). The tell was that isolated ops were fine but the assembled
model drifted — pointing at a stateful fusion rather than a bad kernel.

## Current pipeline invocation

The loop sets `LD_LIBRARY_PATH` and `ORT_DYLIB_PATH` itself now. From `training/`:

```bash
uv run python -m vgo_training.rl_loop \
  --output ../artifacts/rl-cuda-demo \
  --updates 2 --samples-per-shard 256 --shards-per-update 1 \
  --replay-window 2 --maximum-prefetch-shards 1 \
  --resolution 128 --policy-resolution 32 --coarse-pool 8 \
  --generation-simulations 48 --arena-simulations 48 \
  --maximum-plies 64 \
  --training-epochs 40 --training-batch 32 \
  --warm-learning-rate 1e-4 --value-weight 0.1 \
  --training-device cuda --training-precision bfloat16 \
  --actors 16 --arena-actors 1 --arena-pairs 12 \
  --inference-batch 32 --provider cuda --inference-device-id 0 \
  --promotion-arena --promotion-score 0.52 \
  --maximum-truncation-rate 0.05
```

For a CPU-only smoke, use
`--provider cpu --training-device cpu --training-precision float32`.
The coordinator's default TensorRT engine warmup is skipped for both the CUDA
provider used above and the CPU provider.

The pipeline's `--coarse-pool` default is `4`, is forwarded to generation and
every arena, and must not exceed the policy resolution. In the command above,
the first generation still falls back to legacy candidates:
the naive bootstrap evaluator has no spatial policy grid. Supply both an initial
checkpoint and ONNX model to use coarse generation immediately; otherwise it
starts after the first accepted model. Coarse search uses the ordinary
visit-count progressive-widening schedule, with cumulative IID delta draws,
duplicate multiplicity accounting, and deterministic pass.

To drive `vgo-arena` directly, export the two env vars first (the loop does this
for you):

```bash
V="$PWD/training/.venv/lib/python3.14/site-packages"
export ORT_DYLIB_PATH="$V/onnxruntime_blackwell/libonnxruntime.so"
export LD_LIBRARY_PATH="$V/onnxruntime_blackwell:$V/nvidia/cu13/lib:$V/nvidia/cudnn/lib:$V/torch/lib"
```

## History: walls cleared to get here

1. No Rust toolchain → installed rustup 1.97.1.
2. `libcublasLt.so.13` not found → CUDA runtime libs live in the venv; put them
   on `LD_LIBRARY_PATH` (the Linux branch of `runtime_environment()` now does).
3. `cudaErrorNoKernelImageForDevice` → prebuilt onnxruntime has no sm_120 CUDA
   kernels. This is onnxruntime#26177 (CLOSED); fix is from-source with
   `CMAKE_CUDA_ARCHITECTURES=120`.
4. TensorRT-RTX (`nvrtx`) attempt: wired an `OnnxProvider::NvRtx` variant and
   installed `tensorrt-cu13-libs`; the provider loaded but its engine builder
   rejected a fused subgraph of our ONNX (`failed to create engine from network
   for fused node`), independent of fp16/batch/profile. Abandoned in favour of
   the CUDA-from-source route. The `nvrtx` code was **removed** when switching
   `ort` to `load-dynamic`; the current provider set is `cpu`/`cuda`/`tensorrt`.
5. From-source onnxruntime build snags: GCC 16 too new for CUDA 13 (used a
   userspace gcc-15); nvvm 13.3 vs ptxas 13.0 PTX-version mismatch (pinned nvvm
   13.0.88); the glibc `rsqrt`/`rsqrtf` `noexcept` header conflict (patched
   `nvidia/cu13/include/crt/math_functions.h`); cmake needed explicit
   `CMAKE_MAKE_PROGRAM`/`CMAKE_C[XX]_COMPILER`; missing CUDA driver stub
   (symlinked `/usr/lib64/libcuda.so`).

## What is committed vs. environment-only

Committed to the repo: the `ort` `load-dynamic` switch
(`crates/vgo-inference/Cargo.toml`), the `fuse_conv_bias` removal and
`load-dynamic`/`ORT_DYLIB_PATH` handling (`crates/vgo-inference/src/onnx.rs`,
`training/vgo_training/rl_loop.py`), and this doc. The abandoned
`OnnxProvider::NvRtx` (TensorRT-RTX) variant was removed — the story is kept
here for anyone who wants to retry it after a future onnxruntime/ort release.

Not committed (they lived in `training/.venv`, which is gitignored): the
userspace CUDA toolchain, the self-built `libonnxruntime.so` in
`onnxruntime_trt/` and `onnxruntime_blackwell/`, and the TensorRT SDK headers.

**A fresh checkout no longer needs any of that.** `uv sync --extra tensorrt`
installs the stock `onnxruntime-gpu` wheel, which carries sm_120 kernels and
both execution providers, and `runtime_environment()` falls through to it when
the self-built directories are absent — which is the path every run has taken
since. `./scripts/setup.sh` does this and verifies the result; `--extra
tensorrt` is required rather than optional, because `runtime_environment()`
puts `site-packages/tensorrt_libs` on `LD_LIBRARY_PATH` and the TensorRT
provider will not register without it.
