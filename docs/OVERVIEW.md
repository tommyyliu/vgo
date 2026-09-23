# System overview

Start here. This is the map: what the system does, how the pieces fit, and why
the load-bearing decisions are what they are. Each section ends with where to
read more.

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

## 2. The loop

```
  vgo-generate-continuous  --one game per directory-->  games/gen-N-<sha>/
          ^  (holds the incumbent model throughout)           |
          |                                                   v
   new generator on the  <--  export_onnx  <--  train-once.py (from scratch,
   exported model                                on the newest window)
                                                              |
                                          every few updates:  v
                                     vgo-arena vs sampled earlier models
                                     -> anchor.jsonl -> scripts/ratings.py
```

[`scripts/bulk-loop.sh`](../scripts/bulk-loop.sh) runs it, and its header is the
design document. In short:

- **Generation never stops.** One generator writes each finished game as its
  own directory. When a new model is exported, a new generator starts on it and
  the old one drains the games it holds, so the GPU never idles on a
  shard-completion tail.
- **Training is from scratch, not incremental.** When `turnover` of the
  `window` (default 15% of 150k samples) is new, a fresh model trains for
  `epochs` (8) on the newest window and is adopted without a gate. A
  warm-started incremental chain lost 41-7 (+307 Elo) to a from-scratch model
  on the same corpus; retraining breaks the chain, so a bad round only affects
  the games it produces.
- **Rating is a graph, not a gate.** Every `anchor_every` updates the new model
  plays short matches against a few sampled earlier models. `scripts/ratings.py`
  fits everything at once (Bradley-Terry), anchored at `sl-w64b16 = 0`.
- **Resignation is soft and self-calibrating.** A conceded game plays on at a
  cheaper search budget, so every game still reaches a real result. The
  threshold is re-picked before each generator from the measured false
  positives (`scripts/resign-calibration.py`).
- **Boards are mixed.** Generation draws 50% at radius 1/38, 25% at 1/18, and
  25% uniformly between them, with komi set per board (a fixed number of points,
  as in Go). The loop's own rating runs at 1/18, and
  [`scripts/bigboard-curve.sh`](../scripts/bigboard-curve.sh) re-rates
  checkpoints at 1/38.

To run any of it, see [`RUNNING.md`](RUNNING.md).

## 3. Language split

**Rust owns everything in the hot loop**: rules, search, rasterization, native
inference, arenas, game serialization. **Python owns training and export
only.** They meet at two file formats: the game dataset (`dataset.vgo` plus a
`manifest.json`) and the ONNX graph.

The split exists because self-play throughput is the binding constraint on
learning speed, and the per-move work (legality, Voronoi geometry, scoring) is
branch-heavy exact computation that does not vectorize.

Read [`SELFPLAY_ARCHITECTURE.md`](SELFPLAY_ARCHITECTURE.md) for the ownership
boundaries and [`ADR 0001`](adr/0001-native-simulator.md) for why the simulator
is native.

## 4. Board representation

A `Position` is rendered to a `[C, H, W]` tensor by `vgo-raster`, sampling the
centre of every pixel. The layouts that remain:

| kind | planes | used by |
|---|---|---|
| `compact-radius` | 7: current/opponent stones, voronoi ridge, settled, komi, previous pass, radius | **the loop**, 256x256 |
| `compact-pass` | 6: the same without radius | the browser client's model |
| `compact-dead-zone` | 6: dead zone in place of settled, for the official rules | [`OFFICIAL_RULES.md`](OFFICIAL_RULES.md) |
| `semantic` | 12 engineered channels | shard tooling, legality masks, tests |

The radius plane exists because board size *is* the radius (the board is always
the unit square). An empty board renders identically at every radius without
it.

Game datasets store positions, not pixels, and are rasterized at load time by
`vgo-render-shard`. See [`POSITION_SHARDS.md`](POSITION_SHARDS.md) for why, and
[`RASTER_REPRESENTATION.md`](RASTER_REPRESENTATION.md) for the channel semantics.

The `settled` plane is 92-96% of raster cost. It is computed by a distance
transform ([`research/SETTLED_REGION_PROBLEM.md`](research/SETTLED_REGION_PROBLEM.md)).

## 5. Search and the policy target

MCTS with progressive widening. The policy head emits a **spatial map over a
128x128 grid** plus one pass logit. Candidate moves are drawn coarse-to-fine:
a coarse cell (16x16 pooling) is picked from the map, then a fine draw picks a
point within it. Uniform mass (10%) is mixed into the root's proposal so the
loop can try moves its policy does not already favour.

This replaced a random-candidate sampler that could not train: with candidates
drawn independently of the board, the policy target carried no board-dependent
signal. See [`POLICY_REDESIGN.md`](POLICY_REDESIGN.md).

The widening coefficient matters more than any model-side lever: raising it
from 2 was worth about +240 Elo. The loop generates at 6.0 and rates at 4.0.

## 6. The model

DDRNet-inspired dual-resolution convolutional net in
[`training/vgo_training/model.py`](../training/vgo_training/model.py), and the only
architecture left. The loop trains `width=64, blocks=16`, one attention block in
each context stage, GroupNorm with 8 groups: about 8.2M parameters.

**Shape.** A stem downsamples by 4. A *detail* branch stays at that resolution
and carries placement geometry; a *context* branch steps down twice more and
carries global information. Two bilateral fusions exchange between them.

**Heads.** Policy (spatial map + pass), value, and ownership. Each exists twice:
a plain set reading raw trunk features, and a `_normed` set reading
batch-normalized features. The normalized set carries most of the training loss;
the plain set is what inference and the exported graph use. Without a norm in
front of *some* head, nothing penalizes weight magnitude, and trunk weights
inflate until activations overflow fp16.

**Value is categorical.** Two logits, P(mover wins) and P(mover loses), are
collapsed by `value_utility` to the [-1, 1] scalar the search consumes. A tanh
scalar with MSE was abandoned: its `(1 - v^2)` gradient factor vanished exactly
on confidently wrong positions.

**Ownership is auxiliary** and currently weighted 0. It is training-only and
never enters the exported graph.

**Optimizer: Adam.** Muon led early in A/Bs and was overtaken by update 24 in the
loop. Bigger nets are a dead end at this compute: one doubling of simulations is
worth about 61 Elo, which is the exchange rate to judge capacity results by.

Exact shapes and the per-stage budget are in
[`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).

## 7. Serving

Models are exported with a **packed input**: binary planes as bits, constant
planes as one scalar each, the rest fp16, expanded inside the graph. At 256x256
that stages 152 KB per position instead of 1792 KB, which doubled measured
throughput (1.72k -> 3.7k positions/s) with bit-identical outputs.

Rust loads the graph through ONNX Runtime's TensorRT provider (fp16) and serves
it behind a batching broker. Two inference slots overlap host staging with GPU
execution. `--inference-slots 2 --maximum-batch 32 --leaf-batch 4` is the
known-good combination: at 4 slots the inference threads livelock.

The generator asks CUDA to block rather than spin while waiting on the GPU,
which frees two cores that were otherwise pinned at 100%.

The protocol and evaluator interface are in
[`INFERENCE_PROTOCOL.md`](INFERENCE_PROTOCOL.md).

## 8. Decisions worth knowing, with evidence

**Two GroupNorms per residual block, 8 groups.** The groups are statistically
interchangeable (spread across group means 1.24x against 3.35x within), and
LayerNorm trains equivalently. Grouping is kept for throughput: 1 group is ~43%
slower on TensorRT, and 4-96 groups are within 0.4% of each other.

**One norm per block is a known, unadopted win**: +11% inference and -10%
training wall time, with no activation growth (peak validation activation
1.04x). Its strength was never measured in an arena.

**Ownership: BCE, not MSE.** MSE has no finite per-cell optimum against +/-1
targets and drove 16.6% of cells past +/-1.

**Few large arena matches, not many small ones.** Cost per game falls ~3.3x as a
match grows, because concurrency is capped by game count and a small match never
fills an inference batch.

**Training telemetry is not evidence of strength.** `best_epoch` and validation
levels are not comparable across rounds. Judge by play.

## 9. Where to read next

- **Running it:** [`RUNNING.md`](RUNNING.md).
- **Changing the loop:** the header of [`scripts/bulk-loop.sh`](../scripts/bulk-loop.sh),
  then [`SELFPLAY_ARCHITECTURE.md`](SELFPLAY_ARCHITECTURE.md).
- **Changing the model:** §6, then [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md)
  and `model.py`.
- **Changing the representation:** §4, then
  [`RASTER_REPRESENTATION.md`](RASTER_REPRESENTATION.md) and
  [`POSITION_SHARDS.md`](POSITION_SHARDS.md).
- **Geometry performance research** (locality bounds, support certificates,
  iteration lab): [`research/`](research).
- **History:** commit messages carry the reasoning for individual changes, and
  everything removed in the 2026-09-23 prune is on `archive/pre-prune`.
