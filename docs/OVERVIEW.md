# System overview

Start here. This is the map: what the system does, how the pieces fit, and why
the load-bearing decisions are what they are. Every section ends with where to
read more, so nothing here is duplicated in depth.

Everything below describes the system **as it runs today**. Where a document is
known to be out of date, that is called out inline.

---

## 1. What this is

An AlphaZero-style reinforcement learning system for **Voronoi Go**, a
continuous-action variant of Go. A stone is placed at any real coordinate, not
on a grid intersection; territory is decided by Voronoi cells. The rules are in
[`reference/RULES.md`](../reference/RULES.md), with a JavaScript implementation
in [`reference/`](../reference/README.md) that serves as the behavioural oracle.

Continuous placement is the fact that drives most design decisions below. There
is no finite action set, so:

- the board must be **rendered to a raster** for a convolutional model (§4);
- the policy cannot be a softmax over legal moves, so it is a **spatial map plus
  a coarse-to-fine sampler** (§5);
- move proposals come from **progressive widening**, not enumeration (§5).

## 2. The loop, at a glance

```
   self-play (Rust)  ->  replay shards  ->  training (Python)  ->  ONNX export
        ^                                                              |
        +--------------------------- new model <-----------------------+
                                         |
                                     arena / Elo
```

Four processes, run by a queue-driven pipeline rather than a serial barrier:

| stage | owner | what it produces |
|---|---|---|
| generation | Rust `vgo-generate-demo` | replay shards of (position, policy target, value, ownership) |
| training | Python `PersistentLearner` | a `.pt` checkpoint |
| export | Python `export_onnx` | a `.onnx` graph the Rust side serves |
| rating | Rust `vgo-arena` | pairwise results fit to Bradley-Terry Elo |

Generation and training **overlap** — the learner trains on shard N while actors
generate shard N+1. Details, including the utilization feedback loop and how
shards are retired, are in [`RL_LOOP.md`](RL_LOOP.md).

To actually run any of it, see [`RUNNING.md`](RUNNING.md). That document also
collects the failure modes that are confusing the first time (TensorRT library
paths, engine cache invalidation, stale resolution defaults).

## 3. Language split

**Rust owns everything in the hot loop**: rules, search, rasterization, native
inference, arenas, shard serialization. **Python owns training and export
only.** Rust does not import Python; they meet at two file formats — the replay
shard and the ONNX graph.

This split exists because self-play throughput is the binding constraint on
learning speed, and the per-move work (legality, Voronoi geometry, scoring) is
branch-heavy exact computation that does not vectorize.

Read [`SELFPLAY_ARCHITECTURE.md`](SELFPLAY_ARCHITECTURE.md) for the ownership
boundaries and [`ADR 0001`](adr/0001-native-simulator.md) for why the simulator
is native.

## 4. Board representation

A `Position` is rendered to a `[C, H, W]` f32 tensor by `vgo-raster`, sampling
the centre of every pixel. Two layouts exist:

- **semantic**, 12 channels — stones, Voronoi cells, distance fields, ridges,
  legality clearance, radius, pass state, settled mask, komi;
- **compact**, 5 channels — `current_stones`, `opponent_stones`,
  `voronoi_ridge`, `settled`, `komi`.

**Production runs compact at 128x128.** An ablation found the dropped channels
were recoverable from the kept ones, and 5 channels cut both shard size and the
per-batch staging copy.

> [`RASTER_REPRESENTATION.md`](RASTER_REPRESENTATION.md) still describes a
> 10-channel tensor. The channel *semantics* it documents are correct; the count
> is stale (now 12 semantic / 5 compact).

Shards store game records and re-render rasters on load rather than storing
pixels — see [`POSITION_SHARDS.md`](POSITION_SHARDS.md) for why (rendered pixels
were 6 GB per shard, 4 GB of it raster).

## 5. Search and the policy target

MCTS with progressive widening. Because the action space is continuous, the
policy head emits a **spatial map over a P x P grid** (`policy_resolution`),
plus one pass logit. Candidate moves are drawn coarse-to-fine: the coarse grid
picks a cell, then a fine draw picks a point within it.

This replaced a random-candidate sampler that could not train: with candidates
drawn independently of the board, the policy target carried no board-dependent
signal. That redesign is documented in [`POLICY_REDESIGN.md`](POLICY_REDESIGN.md).

`policy_resolution` is decoupled from the raster resolution on purpose — a
~9-across board does not need 128x128 of placement precision, and a coarser grid
concentrates a fixed number of proposal draws over fewer cells. Production
currently runs both at 128.

## 6. The model

DDRNet-inspired dual-resolution convolutional net in
[`training/vgo_training/model.py`](../training/vgo_training/model.py).
Production is `width=96, blocks=16` (~18.3M parameters, ~18.29M exported).

**Shape.** A stem downsamples 128 -> 32. A *detail* branch stays at 32x32 and
carries placement geometry; a *context* branch steps down 16 -> 8 and carries
global information. Two bilateral fusions exchange between them. The detail
branch dominates cost — three stages at ~1.9 ms each against ~1.3-1.5 ms for
context.

**Heads.** Policy (spatial map + pass), value, and ownership. Each exists twice:
a plain set reading raw trunk features, and a `_normed` set reading
batch-normalized features. The normalized set carries most of the training loss;
the plain set is what inference and the exported graph use. Without a norm in
front of *some* head, nothing penalizes weight magnitude, and trunk weights
inflate until activations overflow fp16. Keeping an unnormalized twin for
inference keeps BatchNorm running statistics out of the exported graph.

**Value is categorical.** Two logits — P(mover wins), P(mover loses) — collapsed
by `value_utility` to the [-1, 1] scalar the search consumes. A tanh scalar with
MSE was tried and abandoned: its `(1 - v^2)` gradient factor vanished exactly on
confidently-wrong positions, so the cases most needing correction learned
slowest. Ownership uses the same idea with the redundant logit dropped (BCE on
one logit is identical to two-class softmax CE).

**Ownership is auxiliary.** It predicts who holds each cell at game end. It is
spatial rather than scalar because a game's ~58 positions share one value label,
which a net this size memorizes by trajectory; ownership varies within a game so
it cannot collapse that way. It is training-only and never enters the exported
graph.

**Normalization: GroupNorm, 8 groups, two per residual block.** See §8 for the
measurements behind this and what is known about changing it.

Exact shapes, the parameter and time budget per stage, and the experimental
options are in [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).

## 7. Serving

The exported graph takes `states` and emits `policy_logits` and `values`. Rust
loads it through ONNX Runtime with the TensorRT execution provider (fp16) and
serves it behind a batching broker: actors submit evaluation requests, a broker
thread assembles them into batches up to `maximum_batch`, waits at most
`delay_ms`, and dispatches.

**Two execution slots can overlap host staging and GPU execution.** Each slot
owns its own session, thread, and staging buffer. A shared broker now builds
batches before selecting a slot, so the concurrency no longer divides arrivals
between independent queues. Three slots collapsed in the original measurement
because there is only one GPU and the third session added contention.

`inference_slots` defaults to 2 in `PipelineConfig`. With the shared broker, the
exact w64/b16 attention model measured 16,433 pos/s at batch 32, 13,343 at batch
64, and 14,255 with three slots at batch 16. A paired production-shaped short
self-play run improved from 20.67s to 18.92s at batch 32 while averaging
31.8/32 positions. Batch size is a resumable serving control rather than search
identity; the effective ceiling remains recorded per shard. A resumed ceiling
must fit the current ONNX artifact, so lowering 64 to 32 works directly while
raising beyond an old export requires re-exporting it.

The protocol and the evaluator interface are in
[`INFERENCE_PROTOCOL.md`](INFERENCE_PROTOCOL.md). The Blackwell/sm_120 toolchain
situation — why onnxruntime is built from source and loaded via `ORT_DYLIB_PATH`
— is in [`NVRTX_HANDOFF.md`](NVRTX_HANDOFF.md).

## 8. Decisions worth knowing, with evidence

These are the choices a newcomer is most likely to want to re-litigate. Each was
measured.

**Two GroupNorms per residual block, 8 groups.**
Grouping buys nothing representationally here: across all 42 norm sites of a
trained model, the spread *across* group means is 1.24x while the spread *within*
groups is 3.35x, so the groups are statistically interchangeable and LayerNorm
loses no information. LayerNorm also trains equivalently (policy_kl 1.004x,
value_mae 0.98x). The reason to keep grouping is throughput: 1 group is ~43%
slower on TensorRT, while 4/8/12 groups are within 0.4% of each other and the
curve is flat to 96. So 8 is correct, for a reason unrelated to why it was
originally chosen.

**One norm per block is a known, unadopted win.** Halving the norms (one after
the second conv instead of one after each) is **+11% end-to-end inference and
-10% training wall time**, and it does *not* destabilize: peak validation
activation 1.04x, fp16 headroom 747x, deepest-block peak actually 21% *lower*.
Removing normalization entirely is faster per batch but **slower end to end** —
it finishes batches faster than the broker can feed them and drops the GPU to
78-85%. Not yet adopted because strength was never measured in an arena; the
change is one line (`ResidualBlock`) plus a fresh run to validate.

**Value head: categorical, not tanh+MSE.** Measured median gradient damping of
0.0004 on real positions under tanh+MSE — a 2500x weaker signal precisely where
the model was confidently wrong.

**Ownership: BCE, not MSE.** MSE has no finite per-cell optimum against +/-1
targets, so it keeps pulling long after the sign settles; it drove 16.6% of cells
past +/-1, a magnitude that means nothing for a bounded quantity.

**Komi is sampled per game, from a normal.** Checkpoints are rated at the komi
they trained on, since strength is komi-dependent.

**Resignation is soft, with an adaptive threshold** chosen from measured error
rather than a constant, and disabled when no threshold meets the false-positive
target.

**Few large arena matches, not many small ones.** Cost per game falls ~3.3x as a
match grows (3.08 s/game at 4 pairs, 0.92 at 60) because concurrency is capped by
game count and a small match never fills an inference batch.

## 9. Where the numbers live

Measured results that are not in the code:

- [`benchmarks/`](../benchmarks/README.md) — workload definitions and retained
  JSON results.
- [`RL_LOOP.md`](RL_LOOP.md) — loop-level throughput and the utilization
  feedback loop.
- Git history is unusually informative here; commit messages carry the reasoning
  for individual changes (`git log --oneline` and read the ones that sound like
  decisions).

## 10. Reading order

- **Running it:** [`RUNNING.md`](RUNNING.md).
- **Changing the loop:** [`RL_LOOP.md`](RL_LOOP.md), then
  [`SELFPLAY_ARCHITECTURE.md`](SELFPLAY_ARCHITECTURE.md).
- **Changing the model:** §6 above, then
  [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md), then `model.py`, then
  [`POLICY_REDESIGN.md`](POLICY_REDESIGN.md) for why the policy is shaped as it
  is.
- **Changing inference:** §7 above, then
  [`INFERENCE_PROTOCOL.md`](INFERENCE_PROTOCOL.md), then
  [`NVRTX_HANDOFF.md`](NVRTX_HANDOFF.md) if the toolchain misbehaves.
- **Changing the representation:** §4 above, then
  [`RASTER_REPRESENTATION.md`](RASTER_REPRESENTATION.md) (note the stale channel
  count) and [`POSITION_SHARDS.md`](POSITION_SHARDS.md).

Historical or superseded: [`RGB_REPRESENTATION_EXPERIMENT.md`](RGB_REPRESENTATION_EXPERIMENT.md)
records a representation that was tried and dropped.
[`NEXT_MILESTONE.md`](NEXT_MILESTONE.md) describes a milestone that has since
been met.
