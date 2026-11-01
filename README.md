# Voronoi Go

AlphaZero-style self-play for a continuous-action game: stones go anywhere in the unit
square, and territory is decided by Voronoi cells.

| Read this | For |
|---|---|
| [`spec/RULES.md`](spec/RULES.md) | the game, in one page |
| [`docs/BACKLOG.md`](docs/BACKLOG.md) | the goal and what is next |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | what we have learned and how results are decided |
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | how the system fits together |
| [`spec/ALGORITHMS.md`](spec/ALGORITHMS.md) | the core ideas, separate from the code |
| [`docs/RUNNING.md`](docs/RUNNING.md) | commands |

Run `correctness/check.sh` before merging.

## Layout

| path | what it is |
|---|---|
| [`spec/`](spec/RULES.md) | rules, formal rules, lemmas with proofs, algorithms |
| [`crates/vgo-core`](crates/vgo-core) | exact rules engine: legality, Voronoi geometry, captures, scoring |
| [`crates/vgo-search`](crates/vgo-search) | progressive-widening MCTS with coarse-to-fine candidate sampling |
| [`crates/vgo-raster`](crates/vgo-raster) | position to model input; `vgo-render-shard`, `vgo-pack-shard` |
| [`crates/vgo-inference`](crates/vgo-inference) | ONNX Runtime sessions behind a batching broker |
| [`crates/vgo-selfplay`](crates/vgo-selfplay) | `vgo-generate-continuous`, `vgo-arena`, `vgo-serve-move` |
| [`crates/vgo-bench`](crates/vgo-bench) | `vgo-bench` timings, `vgo-verify` output digests |
| [`crates/vgo-wasm`](crates/vgo-wasm) | the engine for the browser client |
| [`training/`](training) | PyTorch model, learner, ONNX export; [`loop/`](training/loop) runs the RL loop |
| [`evaluation/`](evaluation) | ratings and checkpoint curves |
| [`web-bot/`](web-bot/README.md) | the embeddable browser bot: engine as WebAssembly, no UI |
| [`scripts/`](scripts) | setup, environment and `play.sh` |
| [`js-app/`](js-app/README.md) | the playable game, whose JS engine is also the behavioural oracle |
| [`docs/`](docs) | overview, commands, backlog, [`reference/`](docs/reference) for interfaces, [`history/`](docs/history/README.md) for dated records |
| [`performance/`](performance/README.md) | timing runner, fixture positions and history |
| [`correctness/`](correctness) | `check.sh`, spec and link checks, golden digests |
| [`dashboard/`](dashboard/README.md) | performance and research dashboards: `python3 -m dashboard` |
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
