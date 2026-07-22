# Python Training

This directory will contain model definition, inference serving, replay input,
and optimization code. It deliberately has no Rust package dependency.

The Python inference service communicates with the Rust self-play executable
through the versioned batch protocol described in
[`docs/SELFPLAY_ARCHITECTURE.md`](../docs/SELFPLAY_ARCHITECTURE.md). Training
reads immutable replay shards written by Rust.

No simulator or game-rule implementation belongs here.

## Raster training canary

Generate MCTS-labeled examples from the repository root:

```powershell
cargo run --release -p vgo-raster --bin vgo-generate-demo -- `
  --samples 96 --resolution 128 --simulations 100 --output artifacts/raster-demo
```

Then run the small policy/value overfit experiment from this directory:

```powershell
uv run python -m vgo_training.train_demo `
  ../artifacts/raster-demo/dataset.vgo `
  --output ../artifacts/raster-demo/model.pt --device cuda `
  --model-width 32 --blocks 3
```

The binary loader validates the schema, exact file size, policy normalization,
candidate mask, finite values, and tensor dimensions before training. The first
96-sample CPU canary reduced sampled-policy KL from `0.489` to `0.054`, reached
`81.2%` sampled-action top-1 agreement, and reduced value MAE from `0.997` to
`0.131`. Its retained metrics are in
[`../benchmarks/results/2026-07-21-raster-training-canary.json`](../benchmarks/results/2026-07-21-raster-training-canary.json).

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
root with `cargo run --release -p vgo-inference --bin vgo-model-smoke`. The
active canary uses radius `1/6` and a 128x128 raster by default. `--radius` and
`--resolution` are independent so a small game can exercise a larger inference
tensor.

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

From the repository root, `vgo-stage-bench` separately measures Rust
rasterization, request framing, and the warm subprocess service:

```powershell
cargo run --release -p vgo-inference --bin vgo-stage-bench
```

Windows uses the CUDA 13.0 PyTorch index and the matching `triton-windows`
compiler package. The service fails at startup if CUDA is requested but
unavailable. Compiled inference warms one fixed maximum-batch graph before
reading requests and pads partial batches to that shape, avoiding live graph
recompilation.
