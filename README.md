# Voronoi Go

AlphaZero-style self-play for a continuous-action game: stones go anywhere in the unit
square, and territory is decided by Voronoi cells.

| Read this | For |
|---|---|
| [`spec/RULES.md`](spec/RULES.md) | the game, in one page |
| [`DIRECTION.md`](DIRECTION.md) | the goal, active bets and working rules |
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | how the system fits together |
| [`spec/ALGORITHMS.md`](spec/ALGORITHMS.md) | the core ideas, separate from the code |
| [`docs/RUNNING.md`](docs/RUNNING.md) | commands |

Run `scripts/check.sh` before merging.

## Layout

| path | what it is |
|---|---|
| [`spec/`](spec/RULES.md) | rules, formal rules, lemmas with proofs, algorithms |
| [`crates/vgo-core`](crates/vgo-core) | exact rules engine: legality, Voronoi geometry, captures, scoring |
| [`crates/vgo-search`](crates/vgo-search) | progressive-widening MCTS with coarse-to-fine candidate sampling |
| [`crates/vgo-raster`](crates/vgo-raster) | position to model input; `vgo-render-shard`, `vgo-pack-shard` |
| [`crates/vgo-inference`](crates/vgo-inference) | ONNX Runtime sessions behind a batching broker |
| [`crates/vgo-selfplay`](crates/vgo-selfplay) | `vgo-generate-continuous`, `vgo-arena`, `vgo-serve-move`, `vgo-bench` |
| [`crates/vgo-wasm`](crates/vgo-wasm) | the engine for the browser client |
| [`training/`](training) | PyTorch model, learner, ONNX export |
| [`web-bot/`](web-bot/README.md) | the embeddable browser bot: engine as WebAssembly, no UI |
| [`scripts/`](scripts) | the loop (`bulk-loop.sh`), checks and tools |
| [`js-app/`](js-app/README.md) | the playable game, whose JS engine is also the behavioural oracle |
| [`docs/`](docs) | system docs; [`docs/research/`](docs/research/README.md) holds dated, frozen notes |
| [`todo/`](todo/README.md) | backlog tickets |
| `benchmarks/` | fixture positions, timing history, golden digests |
| `artifacts/` | run output, not tracked. `artifacts/old/` is the archive |

## Tech stack

| Area | Part | Tools |
|---|---|---|
| Self-play | Rules, search, rasterization | Rust |
| | Inference | Rust, ONNX Runtime through `ort` (CUDA, TensorRT) |
| Training | Rasterization | Rust binaries called from Python, with a NumPy fallback |
| | ML loop | Python, PyTorch, ONNX export, `uv`; driven by Bash |
| Website | Game | Plain JavaScript and HTML |
| | Browser bot | Rust compiled to WebAssembly, `onnxruntime-web`; no server |
| Evaluation | Arena and ratings | Rust `vgo-arena`, Python Bradley-Terry |
| Oracle | JS engine in the game | Tests run through PowerShell |

Code for experiments that were concluded (the shard pipeline, Muon, the flat and
U-Net nets, RGB rasters, tournament tooling and the rest) was removed on
2026-09-23. It is all on the `archive/pre-prune` branch.
