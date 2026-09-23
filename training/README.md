# Python training

The model, the learner and ONNX export. No game rules live here: games are
generated and rasterized by the Rust side, and this package reads what it
writes.

| module | role |
|---|---|
| `model.py` | the DDRNet policy/value/ownership net, `build_model`, `load_model` |
| `attention.py` | board transformer blocks used in the context branch |
| `learner.py` | `PersistentLearner`: replay window, batch staging, optimization, checkpointing |
| `supervision.py` | policy targets, losses, metrics, dihedral augmentation, LR schedule |
| `dataset.py` | reads `dataset.vgo` game records; rasterizes via `target/release/vgo-render-shard` |
| `packed_states.py`, `packed_policy.py` | compact in-memory forms of the replay window |
| `packed_input.py` | the in-graph unpacking used by `export_onnx --packed-input` |
| `export_onnx.py` | checkpoint -> ONNX plus a `.json` manifest |
| `bradley_terry.py` | the rating fit used by `scripts/ratings.py` |

The entry point for training is [`scripts/train-once.py`](../scripts/train-once.py),
which the loop calls once per round. See [`docs/RUNNING.md`](../docs/RUNNING.md).

```bash
uv sync --frozen --extra tensorrt
.venv/bin/python -m unittest discover -s tests
```

The dataset loader shells out to `vgo-render-shard` and `vgo-pack-shard`, so
build the workspace (`cargo build --release`) before training or running the
dataset tests.

Only DDRNet checkpoints with `norm_groups` load. Older architectures need the
`archive/pre-prune` branch.
