# Next Milestone: Model Boundary Canary

## Goal

Replace the in-process naive evaluator with a versioned, batched model boundary
without changing deterministic search behavior. Rust continues to own games,
candidate generation, MCTS, and replay production. Python receives tensors and
candidate features, returns policy/value predictions, and can train from replay
files without importing the Rust simulator.

This milestone validates the system boundary before model quality becomes a
variable.

## Completed foundation

- The player-relative ten-channel contract and scalar pixel-center rasterizer
  are implemented in `vgo-raster`.
- RGB overview and per-channel diagnostics are derived from the tensor itself.
- Version 2 demo shards carry raster states, dense policy targets, sampled-pixel
  masks, and current-player terminal values.
- Python loads the Rust format without a simulator dependency, and a 59,555
  parameter residual CNN successfully fits the first 96-sample canary.

See [`RASTER_REPRESENTATION.md`](RASTER_REPRESENTATION.md) for the frozen state
and policy-grid semantics. The remaining work in this milestone is production
batching, transport, broader fixtures, and integration into MCTS.

## 1. Freeze the model contract

Write a versioned schema covering:

- player-relative board tensors, dimensions, channels, dtypes, and coordinate
  orientation;
- dense placement logits on the raster grid plus one pass logit;
- sampled candidate pixels and masks used to gather inference logits and define
  training normalization;
- value on a documented current-player `[-1, 1]` utility scale;
- request IDs, model version, and explicit error responses.

Keep schema semantics independent of transport so local pipes can later become
shared memory without changing the model API.

## 2. Build the Rust encoder

Implement scalar CPU reference encoding first, with a batch-oriented public API.
Use analytic geometry at pixel centers rather than rendering APIs. Add golden
fixtures for empty, boundary-heavy, capture, self-capture, and finished states,
including checksums and symmetry tests.

Benchmark positions per second across candidate raster sizes and batch sizes.
Choose the initial resolution from measured cost and information loss rather
than assuming one upfront.

## 3. Add an evaluator abstraction and broker - complete

Make MCTS depend on an evaluator trait rather than the current hard-coded
policy/value functions. Implement:

- an in-process naive evaluator preserving today's behavior;
- a deterministic fake batched evaluator for broker tests;
- a broker that batches requests by size or a short latency deadline, applies
  backpressure, and routes responses by request ID.

Keep search randomness derived from game and position seeds, never request
arrival order.

## 4. Connect the Python service - boundary complete

Create a small service with no simulator dependency. Its first implementation
reproduces the naive spread logits and neutral nonterminal value from supplied
features. Compare its outputs and selected root actions against the in-process
evaluator on fixed fixtures and games.

The offline raster policy/value network already exists. Only after service
parity is established should its outputs replace the naive evaluator in search.

## 5. Write auditable replay shards

Rust writes immutable shards containing encoded tensors, candidate features,
root visits, selected action, player perspective, terminal outcome, search
budget, seed, schema version, and model version. Python validates and loads a
shard using only its training dependencies.

Retain enough raw position data and checksums to diagnose encoder changes, but
do not require Python to reconstruct game rules.

## Acceptance gate

- [x] Direct Python and framed-service checkpoint outputs match.
- [x] Out-of-order response IDs route to the correct callers.
- [x] At least 16 concurrent actors complete games without deadlock or mismatched
  responses.
- [x] Broker metrics report queue and combined transport/evaluation time.
- [ ] Split encoding, transport, and model execution into separate timings.
- [x] Python loads a Rust-written demo shard and verifies shapes, masks, and
  offsets without a Rust extension.
- [ ] Add shard checksums and production self-play trajectory metadata.
- [x] The existing 1000-vs-10 canary remains above its acceptance threshold through
  the in-process evaluator path.

## Following milestone

Run a closed learning loop on the radius-`1/6` board scale. Its acceptance test
is that a trained checkpoint beats the naive evaluator under equal search
budgets, with a held-out arena rather than training-set fit as the decision
metric.
