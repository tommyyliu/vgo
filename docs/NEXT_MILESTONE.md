# Next Milestone: Model Boundary Canary

## Goal

Replace the in-process naive evaluator with a versioned, batched model boundary
without changing deterministic search behavior. Rust continues to own games,
candidate generation, MCTS, and replay production. Python receives tensors and
candidate features, returns policy/value predictions, and can train from replay
files without importing the Rust simulator.

This milestone validates the system boundary before model quality becomes a
variable.

## 1. Freeze the model contract

Write a versioned schema covering:

- player-relative board tensors, dimensions, channels, dtypes, and coordinate
  orientation;
- variable-length candidate batches with offsets, coordinates, source, and
  baseline features;
- policy logits aligned one-to-one with candidates;
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

## 3. Add an evaluator abstraction and broker

Make MCTS depend on an evaluator trait rather than the current hard-coded
policy/value functions. Implement:

- an in-process naive evaluator preserving today's behavior;
- a deterministic fake batched evaluator for broker tests;
- a broker that batches requests by size or a short latency deadline, applies
  backpressure, and routes responses by request ID.

Keep search randomness derived from game and position seeds, never request
arrival order.

## 4. Connect the Python service

Create a small service with no simulator dependency. Its first implementation
reproduces the naive spread logits and neutral nonterminal value from supplied
features. Compare its outputs and selected root actions against the in-process
evaluator on fixed fixtures and games.

Only after parity is established, replace the naive computation with a tiny
candidate-conditioned policy/value network.

## 5. Write auditable replay shards

Rust writes immutable shards containing encoded tensors, candidate features,
root visits, selected action, player perspective, terminal outcome, search
budget, seed, schema version, and model version. Python validates and loads a
shard using only its training dependencies.

Retain enough raw position data and checksums to diagnose encoder changes, but
do not require Python to reconstruct game rules.

## Acceptance gate

- Fixed-seed in-process and Python-naive runs choose identical moves.
- Batched and unbatched fake evaluation produce identical search results.
- At least 16 concurrent actors complete games without deadlock or mismatched
  responses.
- Broker metrics separate queue, encoding, transport, and evaluation time.
- Python loads a Rust-written replay shard and verifies shapes, offsets, and
  checksums without a Rust extension.
- The existing 1000-vs-10 canary remains above its acceptance threshold through
  the in-process evaluator path.

## Following milestone

Train a tiny model on the radius-`1/6` board scale. First prove it can overfit a
small replay shard and reproduce stored value and visit targets. Then run a
closed learning loop whose acceptance test is that the trained checkpoint beats
the naive evaluator under equal search budgets.
