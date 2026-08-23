#!/usr/bin/env python3
"""Replay official-v2's last eight updates at a different replay-window width.

The run declined ~110 Elo against the supervised model it was seeded from, and
bulk training on the same shards ties that seed -- so the loss is in the
training regime, not the data or the rules. The regime's distinguishing feature
is that each update sees a six-shard slice where the seed saw thirty-six.

This replays updates 32..39 from update-31's checkpoint, which carries the
optimizer state the loop had at that point, so the arms start identical in
weights *and* Adam moments. Update `k` trains on shards `k-window+1 .. k`,
matching the pipeline: update 39 trained on shards 34..39.

The only difference between arms is how far back the window reaches. No
generation is involved, so the shards are fixed and every arm sees the same
eight new ones -- what changes is the tail behind them.

One learner per arm, eight sequential updates, because ReplayCache lives on the
learner: a fresh process per update would re-render every shard and lose the
cache the real loop relies on. The cache evicts shards that leave the window
before loading the entering one, so peak host memory is one window, not the
union of all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "training"))
from vgo_training.learner import LearnerConfig, LearnerUpdate, PersistentLearner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True,
                        help="staging directory holding shard-NNNNNN/dataset.vgo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--recency-decay", type=float, default=1.0)
    parser.add_argument("--first-update", type=int, default=32)
    parser.add_argument("--last-update", type=int, default=39)
    parser.add_argument("--seed", type=int, default=30100001)
    arguments = parser.parse_args()

    # Matches artifacts-official/official-v2's per-update training exactly --
    # verified against update-000039/publication.json -- except for the two
    # fields under test. Warm-starting from a checkpoint takes the architecture
    # from the checkpoint, so the shape fields here only have to be consistent.
    config = LearnerConfig(
        epochs=1,
        batch_size=256,
        learning_rate=0.001,
        value_weight=2.0,
        ownership_weight=0.0,
        model_width=64,
        blocks=16,
        architecture="ddrnet",
        raster_kind="compact-dead-zone",
        norm_groups=8,
        context_attention_blocks=1,
        attention_heads=8,
        full_adam=True,
        recency_decay=arguments.recency_decay,
        threads=4,
        device="cuda",
        precision="bfloat16",
        seed=arguments.seed,
        compile=True,
        restore_optimizer=True,
        schedule="cosine",
        warmup_epochs=0.0,
        report_every=1,
        validation_fraction=0.1,
        augment=True,
    )

    checkpoint = arguments.initial_checkpoint
    reports: list[dict] = []
    learner = PersistentLearner(defaults=config)
    try:
        for update in range(arguments.first_update, arguments.last_update + 1):
            low = max(0, update - arguments.window + 1)
            datasets = [
                arguments.shards / f"shard-{index:06d}" / "dataset.vgo"
                for index in range(low, update + 1)
            ]
            missing = [str(p) for p in datasets if not p.exists()]
            if missing:
                raise SystemExit(f"missing staged shards: {missing[:3]}")

            destination = arguments.output / f"update-{update:06d}" / "candidate.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            report = learner.update(
                LearnerUpdate(
                    datasets=tuple(datasets),
                    output=destination,
                    initial_checkpoint=checkpoint,
                    config=config,
                )
            )
            validation = report.get("final_validation", {})
            print(
                f"update {update:3d}  shards {low:3d}..{update:<3d} "
                f"({len(datasets):2d})  samples {report.get('samples')}  "
                f"policy_kl {validation.get('policy_kl'):.5f}  "
                f"top1 {validation.get('policy_top1'):.3f}  "
                f"value_mae {validation.get('value_mae'):.5f}",
                flush=True,
            )
            reports.append({"update": update, "shards": [low, update], **report})
            checkpoint = destination
    finally:
        learner.close()

    (arguments.output / "reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"final checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
