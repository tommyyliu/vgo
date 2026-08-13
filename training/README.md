# Python Training

This directory contains model definition, replay input, optimization, ONNX
export, and a retained inference service for protocol diagnostics. It
deliberately has no Rust package dependency.

The production self-play path loads exported ONNX models directly in Rust. The
Python service communicates through the versioned batch protocol described in
[`docs/INFERENCE_PROTOCOL.md`](../docs/INFERENCE_PROTOCOL.md) and remains useful
for output parity and transport benchmarks. Training reads replay shards written
by Rust.

No simulator or game-rule implementation belongs here.

The end-to-end replay, training, export, arena, promotion, and restart workflow
is documented in [`../docs/RL_LOOP.md`](../docs/RL_LOOP.md).

## Production RL learner

The RL coordinator starts `vgo_training.learner` once and speaks a strict
JSON-lines protocol to it for every update. The process retains model weights,
Adam moments, the optional compiled graph, prepared replay-window shards, pinned
host buffers, and CUDA staging buffers. Do not launch that service manually for
a normal run; from this directory use:

```powershell
uv run python -m vgo_training.rl_loop `
  --output ../artifacts/rl-run-NAME `
  --updates 20 --samples-per-shard 1024 --replay-window 8 `
  --architecture ddrnet --model-width 64 --blocks 8 `
  --training-device cuda --training-precision bfloat16 `
  --provider tensorrt --inference-device-id 0 --inference-slots 2 `
  --warm-inference
```

The pipeline defaults to BF16 autocast for CUDA training and FP16 TensorRT
inference. These are independent controls: use `--training-precision float32`
when BF16 is unsupported, and `--no-fp16` for an FP32 inference engine. The
standalone `train_demo` adapter below remains FP32 by default.

After each export, TensorRT warmup defaults on: one full configured batch builds
the model-digest-scoped engine cache while the current actor tail is running.
Use `--no-warm-inference` to disable it. CUDA and CPU providers skip the step.

The learner is persistent across updates, not across coordinator processes. On
restart the coordinator verifies durable replay/model identities, starts a new
service, reloads the accepted parent checkpoint, and rebuilds its in-memory
cache. Runtime controls such as update target, actor and inference-slot counts,
prefetch depth, device placement, compilation, inference warmup, and telemetry
capacity may change on restart; learning semantics and artifact contracts may
not. See the
configuration split and exact restart procedure in
[`../docs/RL_LOOP.md`](../docs/RL_LOOP.md#recovery-and-artifacts).

After a run, use `run.json.utilization` as the tuning feedback loop. It
aggregates overlap, active actors, inference fill, writer backpressure,
optimizer share, and prepared-replay cache reuse, with measured
generation/update counts so partial aggregates are visible. Field definitions
and tuning guidance are in
[`../docs/RL_LOOP.md`](../docs/RL_LOOP.md#utilization-feedback-loop).

## Raster training workflow

Generate a legacy bootstrap shard from the repository root:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-generate-demo -- `
  --samples 96 --resolution 128 --simulations 100 --output artifacts/raster-demo
```

The naive evaluator in that command has no spatial policy grid, so it cannot
exercise coarse-to-fine sampling. Given an exported model, generate spatial
replay with:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-generate-demo -- `
  --samples 96 --resolution 128 --simulations 100 --coarse-pool 8 `
  --runtime onnx `
  --model artifacts/PREV/updates/update-000019/candidate.onnx `
  --provider cpu --device-id 0 --output artifacts/raster-coarse-demo
```

`--coarse-pool` defaults to `0`, must not exceed `--policy-resolution`, and is
the only coarse-sampling control exposed by the generator.

Then run the small policy/value overfit experiment from this directory:

```powershell
uv run python -m vgo_training.train_demo `
  ../artifacts/raster-demo/dataset.vgo `
  --output ../artifacts/raster-demo/model.pt --device cuda `
  --model-width 32 --blocks 3
```

Three model families share the same checkpoint and inference contract:
`--architecture flat` is the original full-resolution residual tower,
`--architecture unet` moves most work into an encoder bottleneck, and
`--architecture ddrnet` uses persistent detail and context branches with two
bilateral fusions. The DDRNet option is a clean adaptation of
[DDRNet-23-slim](https://github.com/ydhongHIT/DDRNet) rather than a literal port:
VGO keeps the detail branch at 1/4 raster resolution and context at 1/8-1/16,
because the road-scene model's 1/8-1/64 schedule discards too much of a
96-128px board. The parameter-matched trial uses 2.66M parameters at width 48,
versus 2.80M for the established width-64 U-Net:

```powershell
uv run python -m vgo_training.train_demo `
  ../artifacts/raster-demo/dataset.vgo `
  --output ../artifacts/raster-demo/ddrnet.pt --device cuda `
  --architecture ddrnet --model-width 48 --blocks 8
```

The dual-resolution design and compact hierarchical pyramid context module are
based on Hong et al.,
["Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation
of Road Scenes"](https://arxiv.org/abs/2101.06085). The initial speed, export,
and replay-training measurements are recorded in
[`../benchmarks/results/2026-07-26-ddrnet-trial.json`](../benchmarks/results/2026-07-26-ddrnet-trial.json).
The standalone `train_demo` default remains `flat`; the production RL pipeline
defaults to `ddrnet` with width 64 and 8 blocks. Selecting DDRNet never changes
how older checkpoints are loaded.

An initial checkpoint always retains the architecture stored in that
checkpoint; model families do not share compatible parameter layouts. To switch
from an existing flat or U-Net checkpoint to DDRNet, start the DDRNet run
without `--initial-checkpoint`.

Replay v3 stores raw visits and beta followed by a `u32` proposal-count array.
The loader validates exact file size, policy/visit agreement, candidate and
proposal support, beta bounds, deterministic-pass metadata, finite values, and
tensor dimensions. Replay v1 and v2 remain compatible; missing proposal counts
are synthesized as zero.

For v3 spatial rows, placement target mass is
`visits * proposal_count / (K * beta)`, while deterministic pass keeps its raw
visit mass. Zero-count legacy rows use normalized raw visits. The full-legal
denominator comes from `legal_clearance` channel 7 plus pass and any sampled
boundary aliases. Training computes these targets and masks once in bounded CPU
batches, caches them for all epochs and metrics, and releases the consumed raw
visits, beta, and proposal-count tensors. The retained end-to-end audit is
[`../benchmarks/results/2026-07-24-coarse-policy-smoke.json`](../benchmarks/results/2026-07-24-coarse-policy-smoke.json).

## Inference service

The service is normally launched and supervised by `vgo-inference`. Its direct
form is useful for protocol debugging, but stdout is binary and must not be used
for logging:

```powershell
uv run python -m vgo_training.serve `
  --checkpoint ../artifacts/raster-demo/model.pt `
  --device cuda --compile --maximum-batch 8
```

Run the complete Rust-side boundary and actor smoke test from the repository
root with `vgo-model-smoke --resolution 128 --policy-resolution 32` through
Cargo. The active canary uses radius `1/6` and a 128x128 raster by default.
`--radius` and `--resolution` are independent so a small game can exercise a
larger inference tensor.

Measure the GPU-resident model without rasterization, framing, IPC, or transfer:

```powershell
uv run python -m vgo_training.benchmark_model `
  --checkpoint ../artifacts/raster-demo/model.pt --batches 1,8,16,32,64
```

Measure raster quantization error and policy/value sensitivity:

```powershell
uv run python -m vgo_training.benchmark_precision `
  --dataset ../artifacts/raster-demo/dataset.vgo `
  --checkpoint ../artifacts/raster-demo/model.pt
```

Export the checkpoint to a dynamic-batch ONNX artifact:

```powershell
uv run python -m vgo_training.export_onnx `
  --checkpoint ../artifacts/raster-demo/model.pt `
  --output ../artifacts/raster-demo/model.onnx --maximum-batch 32
```

The exported graph is self-describing: it records the raster schema, input
shape, policy size, supported maximum batch, and source checkpoint digest.
The Rust loader rejects a graph whose metadata does not match its configured
raster or batch range.

For local TensorRT inference, install the optional runtime libraries alongside
the training environment:

```powershell
uv sync --extra tensorrt
```

TensorRT is a native Rust inference path; it does not import Python. The
optional package is only a convenient way to install NVIDIA's matching native
DLLs. A deployed self-play worker can provide those libraries directly.

On Windows, expose the TensorRT and CUDA dependency directories before running
the benchmark or model smoke test:

```powershell
$env:PATH = "$PWD\training\.venv\Lib\site-packages\tensorrt_libs;" +
  "$PWD\training\.venv\Lib\site-packages\torch\lib;$env:CUDA_PATH\bin;$env:PATH"

cargo run --release -p vgo-inference --bin vgo-onnx-bench -- `
  --provider tensorrt --resolution 128 --policy-resolution 32 `
  --batch 32 --compare-python false

cargo run --release -p vgo-selfplay --bin vgo-model-smoke -- `
  --runtime onnx --provider tensorrt --fp16 true `
  --resolution 128 --policy-resolution 32
```

`vgo-onnx-bench` times input packing, ONNX Runtime, and output collection. Use
`--policy-resolution` to match a checkpoint with a decoupled policy head; it
defaults to `--resolution` for older same-size models. The production 96-to-32
models use `--resolution 96 --policy-resolution 32`. Use `--fp16 false` for an
FP32 TensorRT parity check. TensorRT engine and timing
caches are separated by model digest, precision, raster shape, and maximum
batch under `artifacts/onnx-cache`.

From the repository root, `vgo-stage-bench` separately measures Rust
rasterization, request framing, and the warm subprocess service:

```powershell
cargo run --release -p vgo-inference --bin vgo-stage-bench -- `
  --resolution 128 --policy-resolution 32
```

Windows uses the CUDA 13.0 PyTorch index and the matching `triton-windows`
compiler package. The service fails at startup if CUDA is requested but
unavailable. Compiled inference warms one fixed maximum-batch graph before
reading requests and pads partial batches to that shape, avoiding live graph
recompilation.
