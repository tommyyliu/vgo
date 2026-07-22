# Next Milestone: Closed Learning Loop

## Goal

Generate production self-play shards in Rust, train a checkpoint in Python,
export it to ONNX, and measure the candidate model against the incumbent in a
held-out Rust arena. The first quality gate is modest: on the radius-`1/6`
canary, a trained model should beat the naive spread evaluator under equal
search budgets.

## Completed foundation

- Rust owns exact gameplay, a canonical whole-game playout, deterministic
  progressive-widening MCTS, and parallel actors.
- The ten-channel 128x128 semantic raster and dense pixel policy contract are
  frozen and inspectable as RGB diagnostics.
- Python can load Rust demo data, train the residual policy/value network, and
  export a self-describing dynamic-batch ONNX graph.
- Rust validates and executes that graph in-process through ONNX Runtime with
  CUDA or TensorRT, while the Python subprocess remains available for parity.
- The batching broker derives raster and batch limits from the selected backend
  and rejects malformed or mismatched outputs.

## 1. Production replay shards

Move trajectory output from the demo writer into a generation runner. Shards
must be immutable and atomic, with a versioned header, model digest, checksums,
search configuration, seeds, per-ply masks/visits/actions, terminal utility, and
enough position identity to audit repetition handling. Interrupted shards must
be detectable and safely ignored.

## 2. Replay sampler and trainer

Load many shards without a Rust extension. Define train/validation separation,
uniform position sampling as the baseline, bounded replay retention, and metrics
for policy loss on sampled actions, value calibration, and held-out agreement.
Sampling policy belongs in Python and must be recorded in training metadata.

## 3. Checkpoint publication

Train to a temporary `.pt` path, export and validate ONNX, then publish the model
and manifest atomically. A generation run pins one model digest for its entire
lifetime. Workers never observe a partially written or mid-game replacement.

## 4. Iteration driver

Add an explicit driver for:

```text
generate -> validate replay -> train -> export -> smoke -> arena -> accept/reject
```

Each stage consumes immutable inputs and writes a machine-readable result. A
failed stage leaves the incumbent and prior replay usable.

## 5. Held-out arena

Compare candidate and incumbent with color-swapped games, fixed unseen seeds,
equal simulations, and confidence intervals. Keep the existing 1000-vs-10
naive canary as an engine/search regression; it is separate from model
promotion.

## Acceptance gate

- [ ] Rust writes checksummed, atomic replay shards with model and search
  provenance.
- [ ] Python validates and trains from multiple shards without simulator code.
- [ ] Exported ONNX output matches the source checkpoint on held-out examples.
- [ ] One command completes generation through arena evaluation and emits a
  reproducible run manifest.
- [ ] The first trained candidate beats the naive evaluator at equal search
  budgets on held-out seeds.
- [ ] Existing Rust, browser, inference-parity, and 1000-vs-10 canaries remain
  green.

Pipelined GPU slots, pinned I/O binding, richer replay prioritization, and model
promotion against a learned incumbent follow measurement of this simple closed
loop.
