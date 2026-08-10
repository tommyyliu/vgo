#!/usr/bin/env python3
"""Measure how varied a shard's self-play games are.

Elo says which model wins; it says nothing about whether the games it produces
are worth training on. A policy can get sharper -- winning faster, by more, and
in the same way every time -- and a run whose self-play collapses onto a narrow
set of positions starves its own learner while still looking healthy on
validation loss, because the validation set collapses with it.

Three angles, all from `games.jsonl`:

*Occupancy entropy* bins every final stone onto a coarse grid and takes the
Shannon entropy of the resulting distribution, in bits, normalised by the
maximum for that grid. Games that keep finding different parts of the board
score near 1; games that keep replaying the same shapes score lower.

*Margin* is how lopsided the finishes are. A rising mean |margin| with falling
ply count is the signature of a policy converting faster rather than exploring.

*Cap and resign rates* say how often a game failed to resolve, or ended early.

    scripts/shard-diversity.py artifacts/crosstrain/data-adam ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def games(directory: Path) -> list[dict]:
    path = directory / "games.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def occupancy_entropy(rows: list[dict], grid: int) -> float:
    """Normalised Shannon entropy of final stone positions over a grid x grid map."""
    counts = [0] * (grid * grid)
    total = 0
    for row in rows:
        for stone in row.get("final_stones") or []:
            x, y = stone[0], stone[1]
            i = min(int(x * grid), grid - 1)
            j = min(int(y * grid), grid - 1)
            counts[j * grid + i] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy / math.log2(grid * grid)


def pairwise_similarity(rows: list[dict], grid: int, limit: int = 120) -> float:
    """Mean Jaccard overlap of occupied cells between pairs of games.

    Aggregate entropy is nearly blind to collapse: a hundred games that each
    cover the board differently and a hundred that all cover it the *same* way
    both integrate to a flat occupancy map. Comparing games to each other is
    what separates them -- rising similarity is the documented signature of
    self-play diversity collapse.
    """
    sets = []
    for row in rows[:limit]:
        cells = set()
        for stone in row.get("final_stones") or []:
            i = min(int(stone[0] * grid), grid - 1)
            j = min(int(stone[1] * grid), grid - 1)
            cells.add((i, j))
        if cells:
            sets.append(cells)
    if len(sets) < 2:
        return 0.0
    total, count = 0.0, 0
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            union = len(sets[a] | sets[b])
            if union:
                total += len(sets[a] & sets[b]) / union
                count += 1
    return total / count if count else 0.0


def summarize(directory: Path, grid: int) -> dict | None:
    rows = games(directory)
    if not rows:
        return None
    n = len(rows)
    margins = [abs(float(r.get("margin", 0.0))) for r in rows]
    plies = [int(r.get("plies", 0)) for r in rows]
    stones = [len(r.get("final_stones") or []) for r in rows]
    return {
        "games": n,
        "entropy": occupancy_entropy(rows, grid),
        "jaccard": pairwise_similarity(rows, grid),
        "margin": sum(margins) / n,
        "plies": sum(plies) / n,
        "stones": sum(stones) / n,
        "capped": sum(1 for r in rows if r.get("reached_ply_cap")) / n,
        "resigned": sum(1 for r in rows if r.get("resigned")) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--grid", type=int, default=8)
    arguments = parser.parse_args()

    print(f"{'source':<34}{'games':>6}{'entropy':>9}{'jaccard':>9}"
          f"{'|margin|':>10}{'plies':>7}{'stones':>8}{'capped':>8}")
    for directory in arguments.directories:
        summary = summarize(directory, arguments.grid)
        if summary is None:
            print(f"{str(directory):<34}  no games.jsonl")
            continue
        print(f"{str(directory)[-34:]:<34}{summary['games']:>6}"
              f"{summary['entropy']:>9.4f}{summary['jaccard']:>9.4f}"
              f"{summary['margin']:>10.4f}{summary['plies']:>7.1f}"
              f"{summary['stones']:>8.1f}{summary['capped']:>8.3f}")


if __name__ == "__main__":
    main()
