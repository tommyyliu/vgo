#!/usr/bin/env python3
"""Train one model on a window of games: the training step of the RL loop.

Usage:

    scripts/train-once.py --games-root artifacts/vgo-continuous/games \\
        --window-samples 1200000 --output artifacts/scratch/model.pt \\
        --raster-kind compact-radius --epochs 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.learner import LearnerConfig, LearnerUpdate, PersistentLearner  # noqa: E402


def window_from_games(root: Path, window_samples: int) -> tuple[list[Path], int]:
    """The most recent games totalling at least ``window_samples`` samples.

    Samples, not games and not shards, because they are the only unit that means
    the same thing twice. A game runs 70 plies on an 18-unit board and 312 on a
    38-unit one, so "the last N games" is a quantity of data that swings by a
    factor of four with the board mix; shards from the old generator varied 2.5x
    in sample count for the same reason. What the learner actually spends, and
    what the gradient signal scales with, is samples.

    Recency is the game number, not directory order. Generation labels restart
    at `gen-000000` for every run and collide across them: the current tree has
    six unrelated `gen-000001-*` directories, and one `gen-000000-seed` holding
    games from two different runs. Sorting by label therefore put the newest
    run *first*, and a window took the tail of the oldest generations still on
    disk -- three consecutive bulk updates trained on the identical 759 stale
    games while ~2,000 fresh ones sat unread, which reads as a flat loop rather
    than as a bug.

    Game numbers order correctly across runs because the generator assigns them
    globally (`--first-game`, advanced past everything a previous process could
    claim). They agree with file mtime on 98.8% of sampled pairs; the remainder
    is 32 actors finishing games out of order within one batch, which is
    unordered anyway. A directory that is not `game-<digits>` predates the
    per-game writer and sorts oldest rather than being dropped.

    Counts come from each manifest, so choosing a window costs a few hundred
    small reads rather than loading any data.
    """

    entries: list[tuple[int, str, Path, int]] = []
    for generation in sorted(p for p in root.iterdir() if p.is_dir()):
        for game in sorted(p for p in generation.iterdir() if p.is_dir()):
            manifest = game / "manifest.json"
            dataset = game / "dataset.vgo"
            if not manifest.is_file() or not dataset.is_file():
                continue  # a game still being written, or a staging leftover
            try:
                samples = int(json.loads(manifest.read_text())["samples"])
            except (ValueError, KeyError, OSError):
                continue
            if samples <= 0:
                continue
            number = re.fullmatch(r"game-(\d+)", game.name)
            entries.append(
                (int(number.group(1)) if number else -1, game.name, dataset, samples)
            )
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    chosen: list[Path] = []
    total = 0
    for _, _, dataset, samples in reversed(entries):
        if total >= window_samples:
            break
        chosen.append(dataset)
        total += samples
    chosen.reverse()  # oldest first, so the window reads in play order
    return chosen, total


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
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    if arguments.games_root is not None:
        if arguments.window_samples <= 0:
            parser.error("--games-root needs a positive --window-samples")
        datasets, total = window_from_games(
            arguments.games_root, arguments.window_samples
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
        augment=arguments.augment,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    learner = PersistentLearner(defaults=config)
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
