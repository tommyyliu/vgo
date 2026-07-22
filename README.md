# Voronoi Go

This repository is being prepared for search and machine-learning experiments
over the continuous-action game defined in [`reference/RULES.md`](reference/RULES.md).

## Repository areas

- [`reference/`](reference/README.md) contains the clean, working JavaScript
  implementation, mathematical rules, proofs, architecture notes, and browser
  regression suite. It is the behavioral oracle for future implementations.
- [`docs/`](docs/SELFPLAY_ARCHITECTURE.md) contains the native simulator decision
  and the Rust self-play/Python inference architecture. The immediate execution
  plan is [`docs/NEXT_MILESTONE.md`](docs/NEXT_MILESTONE.md).
- [`benchmarks/`](benchmarks/README.md) defines implementation-neutral workloads
  used to compare candidate simulator backends.
- [`crates/vgo-core/`](crates/vgo-core) is the exact Rust rules engine,
  [`crates/vgo-inference/`](crates/vgo-inference) batches model evaluations
  through the versioned Python subprocess protocol,
  [`crates/vgo-raster/`](crates/vgo-raster) produces the canonical semantic
  tensor and RGB diagnostics,
  [`crates/vgo-search/`](crates/vgo-search) provides deterministic candidates and
  progressive-widening MCTS, and [`crates/vgo-selfplay/`](crates/vgo-selfplay)
  contains the paired canary arena.
- [`training/`](training) is reserved for Python model training and the future
  batched inference service; Rust self-play does not depend on it.
- [`todo/`](todo/README.md) tracks known rule, geometry, and visualization work.

Run the complete Rust verification with `cargo test --workspace` and the
self-play canary using the command in [`benchmarks/README.md`](benchmarks/README.md).
The model-facing channel contract is documented in
[`docs/RASTER_REPRESENTATION.md`](docs/RASTER_REPRESENTATION.md).
The live Rust/Python boundary is documented in
[`docs/INFERENCE_PROTOCOL.md`](docs/INFERENCE_PROTOCOL.md).
