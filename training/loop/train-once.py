#!/usr/bin/env python3
"""Train one model on a window of games: the training step of the RL loop.

Usage:

    training/loop/train-once.py --games-root artifacts/vgo-bulk/games \\
        --window-samples 1200000 --output artifacts/scratch/model.pt \\
        --raster-kind compact-radius --epochs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.learner import LearnerConfig, LearnerUpdate, PersistentLearner, ReplayCache  # noqa: E402
from vgo_training.window import window_from_games  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="*")
    parser.add_argument(
        "--games-root", type=Path, default=None,
        help="directory of per-generation game directories, as written by "
        "vgo-generate-continuous. Selects the window instead of listing "
        "datasets by hand.",
    )
    parser.add_argument(
        "--window-samples", type=int, default=0,
        help="with --games-root, train on the most recent games totalling at "
        "least this many samples.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepared-cache", type=Path, default=None,
        help="keep prepared shards here between runs, so the next update loads "
        "the part of its window this one rendered instead of rendering it "
        "again. Files for shards outside this run's window are deleted.",
    )
    parser.add_argument(
        "--map-cache", action="store_true",
        help="hold the window as memory maps of --prepared-cache files rather "
        "than in RAM, for a window larger than memory.",
    )
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=None,
        help="warm start from this checkpoint; omit to train from scratch, in "
        "which case --model-width/--blocks/--norm-groups take "
        "effect (with a checkpoint, its own shape is used instead)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--value-weight", type=float, default=2.0)
    parser.add_argument("--ownership-weight", type=float, default=0.0)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument(
        "--raster-kind",
        default="compact-radius",
        choices=("semantic", "compact", "compact-radius"),
        help="which planes to render from each game; the loop uses compact-radius",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="square input size to render at; default is each game's own, "
        "which is what generation ran at",
    )
    parser.add_argument("--norm-groups", type=int, default=8)
    parser.add_argument("--norms-per-block", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stem-stride", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--shape", type=json.loads, default=None,
        help="JSON of departures from the default shape; see SHAPE_KEYS in model.py",
    )
    parser.add_argument(
        "--context-attention-blocks",
        type=int,
        default=0,
        help="trailing residual blocks in each ddrnet context stage to replace "
        "with transformer blocks; 0 is the plain convolutional net",
    )
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--restore-optimizer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument(
        "--warmup-epochs", type=float, default=0,
        help="wsd only: epochs ramping to full rate. 0 -- the data has already "
        "been through warmup in the run this continues, and a nonzero value "
        "here just re-ramps every call, exactly what --epochs > 1 avoids.",
    )
    parser.add_argument("--decay-fraction", type=float, default=0.2)
    parser.add_argument("--final-learning-rate-fraction", type=float, default=0.01)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--tail-keep", type=float, default=1.0,
        help="share of tail plies to train on. Every ply stays on disk.",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--value-stride", type=int, default=1,
        help="train the value head on every Nth position of each game only",
    )
    parser.add_argument(
        "--value-labels-per-game", type=int, default=0,
        help="train the value head on this many positions per game, spread "
        "evenly, with the value loss scaled back to the unthinned total",
    )
    parser.add_argument(
        "--history-plies", type=int, default=0,
        help="add the stones of this many previous positions as input planes "
        "(training only: such a net cannot be exported for play)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW weight decay on weight matrices; 0 is plain Adam",
    )
    arguments = parser.parse_args()
    if arguments.games_root is not None:
        if arguments.window_samples <= 0:
            parser.error("--games-root needs a positive --window-samples")
        datasets, total = window_from_games(
            arguments.games_root, arguments.window_samples, arguments.tail_keep
        )
        if not datasets:
            parser.error(f"no complete games under {arguments.games_root}")
        print(
            f"[window] {len(datasets)} games, {total:,} samples "
            f"(asked for {arguments.window_samples:,})",
            flush=True,
        )
        arguments.datasets = datasets
    elif not arguments.datasets:
        parser.error("pass dataset paths, or --games-root with --window-samples")

    config = LearnerConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        value_weight=arguments.value_weight,
        ownership_weight=arguments.ownership_weight,
        model_width=arguments.model_width,
        blocks=arguments.blocks,
        raster_kind=arguments.raster_kind,
        resolution=arguments.resolution,
        context_attention_blocks=arguments.context_attention_blocks,
        attention_heads=arguments.attention_heads,
        norm_groups=arguments.norm_groups,
        norms_per_block=arguments.norms_per_block,
        stem_stride=arguments.stem_stride,
        shape=arguments.shape,
        threads=arguments.threads,
        device=arguments.device,
        precision=arguments.precision,
        seed=arguments.seed,
        compile=arguments.compile,
        restore_optimizer=arguments.restore_optimizer,
        schedule=arguments.schedule,
        warmup_epochs=arguments.warmup_epochs,
        decay_fraction=arguments.decay_fraction,
        final_learning_rate_fraction=arguments.final_learning_rate_fraction,
        report_every=arguments.report_every,
        validation_fraction=arguments.validation_fraction,
        tail_keep=arguments.tail_keep,
        augment=arguments.augment,
        value_stride=arguments.value_stride,
        value_labels_per_game=arguments.value_labels_per_game,
        history_plies=arguments.history_plies,
        weight_decay=arguments.weight_decay,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    learner = PersistentLearner(
        defaults=config,
        replay_cache=ReplayCache(
            spill_directory=arguments.prepared_cache, map_spill=arguments.map_cache
        ),
    )
    try:
        report = learner.update(
            LearnerUpdate(
                datasets=tuple(arguments.datasets),
                output=arguments.output,
                initial_checkpoint=arguments.initial_checkpoint,
                config=config,
            )
        )
    finally:
        learner.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
