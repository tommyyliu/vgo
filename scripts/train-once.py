#!/usr/bin/env python3
"""One-off training run on existing shards, outside the RL loop.

`train_demo.py`'s CLI is a thin adapter over `LearnerConfig` / `PersistentLearner`
that only forwards a subset of fields -- notably not `norm_groups`,
`ownership_weight`, or `raster_kind` (`resolution` is still taken from the
dataset itself rather than configured). This calls the same
`PersistentLearner.update` used by the RL loop and `train_demo.py`, with full
field coverage, so a checkpoint's architecture and loss weights can be matched
exactly instead of falling back to `LearnerConfig` defaults.

Usage:

    scripts/train-once.py \\
        artifacts/ddrnet-tonight/replay/shard-*/dataset.vgo \\
        --output artifacts/ddrnet-tonight/manual-updates/update-000/candidate.pt \\
        --initial-checkpoint artifacts/ddrnet-tonight/updates/update-000038/candidate.pt \\
        --epochs 10 --warmup-epochs 0
"""

from __future__ import annotations

import argparse
import json
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

    Recency is directory order: generation labels sort chronologically and games
    are numbered within them. Counts come from each manifest, so choosing a
    window costs a few hundred small reads rather than loading any data.
    """

    entries: list[tuple[Path, int]] = []
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
            if samples > 0:
                entries.append((dataset, samples))

    chosen: list[Path] = []
    total = 0
    for dataset, samples in reversed(entries):
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
        "which case --model-width/--blocks/--architecture/--norm-groups take "
        "effect (with a checkpoint, its own shape is used instead)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    # Without these the optimizer silently follows LearnerConfig's defaults,
    # which is Muon -- so a script meant to hold the optimizer fixed across an
    # A/B would quietly pick one of the two arms being compared.
    parser.add_argument("--muon-learning-rate", type=float, default=0.01)
    parser.add_argument("--full-adam", action="store_true",
                        help="put every parameter on Adam instead of Muon-on-trunk")
    parser.add_argument(
        "--optimizer", choices=("adam", "muon", "ranger21"), default=None,
        help="overrides --full-adam when given. ranger21 needs the optional "
        "'optimizers' dependency group and runs with its own warmup/warmdown "
        "off, so --schedule drives every arm and the comparison is about the "
        "optimizer rather than about two different learning-rate curves",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--value-weight", type=float, default=2.0)
    parser.add_argument("--ownership-weight", type=float, default=0.0)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--architecture", default="ddrnet")
    parser.add_argument(
        "--raster-kind",
        default=None,
        choices=("semantic", "compact", "compact-pass", "compact-dead-zone", "compact-connected", "compact-radius", "rgb"),
        help=(
            "which planes to render from each shard. A property of the model "
            "rather than of the data: shards store positions, so the raster is "
            "produced at load time and two runs over the same shards can train "
            "different encodings. Omit to fall back to the shard header, which "
            "cannot distinguish compact-pass from compact-dead-zone -- both are "
            "six planes and differ only in the capture predicate"
        ),
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
        architecture=arguments.architecture,
        raster_kind=arguments.raster_kind,
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
        muon_learning_rate=arguments.muon_learning_rate,
        full_adam=arguments.full_adam,
        optimizer=arguments.optimizer,
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
