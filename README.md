# Voronoi Go

AlphaZero-style self-play for the continuous-action game defined in
[`reference/RULES.md`](reference/RULES.md): stones go anywhere in the unit
square, and territory is decided by Voronoi cells.

**New here? Read [`docs/OVERVIEW.md`](docs/OVERVIEW.md)**, then
[`docs/RUNNING.md`](docs/RUNNING.md) for commands.

**Want the bot on a website?** [`client/`](client/README.md) is a self-contained
browser opponent: the engine compiled to WebAssembly plus a small ES module, with
no server and no GPU to operate.

## Layout

| path | what it is |
|---|---|
| [`crates/vgo-core`](crates/vgo-core) | exact rules engine: legality, Voronoi geometry, captures, scoring |
| [`crates/vgo-search`](crates/vgo-search) | progressive-widening MCTS with coarse-to-fine candidate sampling |
| [`crates/vgo-raster`](crates/vgo-raster) | position -> model input tensor; `vgo-render-shard`/`vgo-pack-shard` for the Python loader |
| [`crates/vgo-inference`](crates/vgo-inference) | ONNX Runtime / TensorRT sessions behind a batching broker |
| [`crates/vgo-selfplay`](crates/vgo-selfplay) | `vgo-generate-continuous`, `vgo-arena`, `vgo-serve-move` |
| [`crates/vgo-wasm`](crates/vgo-wasm) | the engine for the browser client |
| [`training/`](training) | PyTorch model, learner, ONNX export |
| [`scripts/`](scripts) | the loop (`bulk-loop.sh`) and its tools |
| [`client/`](client/README.md) | the embeddable browser bot |
| [`reference/`](reference/README.md) | JavaScript reference implementation, rules and proofs: the behavioural oracle |
| [`docs/`](docs) | system docs; [`docs/research/`](docs/research) holds geometry and search research write-ups |
| [`todo/`](todo/README.md) | backlog |

Code for experiments that were concluded (the shard pipeline, Muon, the flat and
U-Net nets, RGB rasters, tournament tooling and the rest) was removed on
2026-09-23. It is all on the `archive/pre-prune` branch.
