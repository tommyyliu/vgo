"""Which games a training window holds, counted in the samples training will keep.

Samples, not games and not shards, because they are the only unit that means the
same thing twice. A game runs 70 plies on an 18-unit board and 312 on a 38-unit
one, so "the last N games" is a quantity of data that swings by a factor of four
with the board mix. What the learner spends, and what the gradient signal scales
with, is samples.

Tail plies are thinned at load, so they count here at the rate they're kept.
Otherwise a 150k window would quietly train on far fewer samples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator


def kept_samples(manifest: dict, tail_keep: float) -> float:
    samples = int(manifest["samples"])
    tail = int(manifest.get("tail_samples", 0))
    return samples - (1.0 - tail_keep) * tail


def finished_games(root: Path) -> Iterator[tuple[int, str, Path, dict]]:
    """Every complete game under `root`, unordered.

    Counts come from each manifest, so this costs a few hundred small reads and
    loads no data. A directory that is not `game-<digits>` predates the per-game
    writer and sorts oldest rather than being dropped.
    """
    for generation in sorted(p for p in root.iterdir() if p.is_dir()):
        for game in sorted(p for p in generation.iterdir() if p.is_dir()):
            path, dataset = game / "manifest.json", game / "dataset.vgo"
            if not path.is_file() or not dataset.is_file():
                continue  # a game still being written, or a staging leftover
            try:
                manifest = json.loads(path.read_text())
                if int(manifest["samples"]) <= 0:
                    continue
            except (ValueError, KeyError, OSError):
                continue
            number = re.fullmatch(r"game-(\d+)", game.name)
            yield int(number.group(1)) if number else -1, game.name, dataset, manifest


def window_from_games(
    root: Path, window_samples: int, tail_keep: float = 1.0
) -> tuple[list[Path], int]:
    """The most recent games totalling at least `window_samples` kept samples.

    Recency is the game number, not directory order. Generation labels restart
    at `gen-000000` for every run and collide across them: sorting by label once
    put the newest run first, and a window took the tail of the oldest generations
    still on disk. Three consecutive updates trained on the identical 759 stale
    games while about 2,000 fresh ones sat unread, which reads as a flat loop
    rather than as a bug.

    Game numbers order correctly across runs because the generator assigns them
    globally (`--first-game`, advanced past everything a previous process could
    claim). They agree with file mtime on 98.8% of sampled pairs, and the remainder
    is actors finishing out of order within one batch.
    """
    games = sorted(finished_games(root), key=lambda game: game[:2])
    chosen: list[Path] = []
    total = 0.0
    for _, _, dataset, manifest in reversed(games):
        if total >= window_samples:
            break
        chosen.append(dataset)
        total += kept_samples(manifest, tail_keep)
    chosen.reverse()  # oldest first, so the window reads in play order
    return chosen, int(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the kept samples on disk.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--tail-keep", type=float, default=1.0)
    arguments = parser.parse_args()
    print(int(sum(
        kept_samples(game[3], arguments.tail_keep) for game in finished_games(arguments.root)
    )))


if __name__ == "__main__":
    main()
