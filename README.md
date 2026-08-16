# Voronoi Go

This repository is being prepared for search and machine-learning experiments
over the continuous-action game defined in [`reference/RULES.md`](reference/RULES.md).

**New here? Read [`docs/OVERVIEW.md`](docs/OVERVIEW.md).** It maps the whole
system top-down — the RL loop, the language split, the board representation, the
model, and the serving path — and records the load-bearing decisions with the
measurements behind them, so you do not have to reconstruct them from a dozen
files.

**Want to put the bot on a website?** [`client/`](client/README.md) is a
self-contained browser opponent: one 360 KB ES module with the engine compiled
to WebAssembly, no server and no GPU to operate. `createBot`, then `chooseMove`.

## Repository areas

- [`reference/`](reference/README.md) contains the clean, working JavaScript
  implementation, mathematical rules, proofs, architecture notes, and browser
  regression suite. It is the behavioral oracle for future implementations.
- [`docs/`](docs/SELFPLAY_ARCHITECTURE.md) contains the native simulator decision
  and the Rust self-play/inference architecture. The immediate execution
  plan is [`docs/NEXT_MILESTONE.md`](docs/NEXT_MILESTONE.md).
- [`benchmarks/`](benchmarks/README.md) defines implementation-neutral workloads
  used to compare candidate simulator backends.
- [`crates/vgo-core/`](crates/vgo-core) is the exact Rust rules engine,
  [`crates/vgo-inference/`](crates/vgo-inference) batches model evaluations
  through native ONNX Runtime/TensorRT or a diagnostic Python subprocess,
  [`crates/vgo-raster/`](crates/vgo-raster) produces the canonical semantic
  tensor and RGB diagnostics,
  [`crates/vgo-search/`](crates/vgo-search) provides deterministic candidates and
  progressive-widening MCTS, and [`crates/vgo-selfplay/`](crates/vgo-selfplay)
  owns complete playouts, the paired canary arena, model smoke tests, and demo
  trajectory generation.
- [`client/`](client/README.md) is the embeddable browser bot: `crates/vgo-wasm`
  compiled to WebAssembly plus the JavaScript that drives its search loop, built
  into a single module a site can serve. Because inference is asynchronous in a
  browser, the search hands its loop out
  ([`SteppedSearch`](crates/vgo-search/src/stepped.rs)), which is also what lets
  it think for a time budget rather than a fixed simulation count. Design notes
  and measurements: [`docs/CLIENT_BOT.md`](docs/CLIENT_BOT.md).
- [`training/`](training) owns Python model training, checkpoint export, replay
  loading, and the retained protocol-debug service. Rust self-play does not
  import it.
- [`todo/`](todo/README.md) tracks known rule, geometry, and visualization work.

[`docs/RUNNING.md`](docs/RUNNING.md) collects every command for tests, the
reinforcement-learning loop, strength measurement and the benchmarks, along with
the failure modes that are confusing the first time.
The model-facing channel contract is documented in
[`docs/RASTER_REPRESENTATION.md`](docs/RASTER_REPRESENTATION.md).
The model execution boundaries are documented in
[`docs/INFERENCE_PROTOCOL.md`](docs/INFERENCE_PROTOCOL.md).
