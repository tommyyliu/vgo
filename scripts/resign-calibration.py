#!/usr/bin/env python3
"""Pool the resignation counterfactual and show what each threshold would cost.

Generation writes one `resign-calibration.jsonl` per game: for every candidate
(threshold, window), whether the rule would have conceded that game, whether the
concession would have been wrong, and how many plies it would have skipped.
Under soft resignation every game calibrates, because the game plays on to a
real terminal state -- the counterfactual is known, not sampled.

This pools those rows over the most recent games and prints the trade. Pick the
lowest threshold whose false-positive rate is under your budget: lower fires
more often and saves more search, so the cheapest setting that clears the bound
is the right one.

Only recent games count. Calibration describes the model that generated them,
and a model still learning invalidates its own history -- one recorded run fired
on 15 of 1625 games over fifteen shards and 440 of 1686 over the next sixteen.

    scripts/resign-calibration.py [games-root] [--games N] [--window W]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _calibration_files(root: Path) -> list[tuple[int, Path]]:
    """Every game's calibration rows, oldest first by game number."""
    found = []
    for generation in root.iterdir():
        if not generation.is_dir():
            continue
        for game in generation.iterdir():
            f = game / "resign-calibration.jsonl"
            m = re.fullmatch(r"game-(\d+)", game.name)
            if f.is_file() and m:
                found.append((int(m.group(1)), f))
    found.sort()
    return found


def main() -> None:
    if "--pick" in sys.argv:
        argv = sys.argv[1:]
        root = Path(argv[0]) if not argv[0].startswith("--") \
            else Path("artifacts/vgo-continuous/games")
        def opt(name, default):
            return type(default)(argv[argv.index(name) + 1]) if name in argv else default
        print(pick(root, opt("--games", 400), opt("--window", 5),
                   opt("--target", 0.03), opt("--fallback", 0.99)))
        return
    argv = sys.argv[1:]
    root = Path(argv[0]) if argv and not argv[0].startswith("--") \
        else Path("artifacts/vgo-continuous/games")
    recent = int(argv[argv.index("--games") + 1]) if "--games" in argv else 400
    only_window = int(argv[argv.index("--window") + 1]) if "--window" in argv else None

    games = _calibration_files(root)
    if not games:
        print(f"  no resign-calibration.jsonl under {root}")
        print("  (only games generated after the calibration write was added have it)")
        return
    games = games[-recent:]

    # (threshold, window) -> [fired, wrong, plies_saved]
    totals: dict[tuple[float, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, f in games:
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if only_window is not None and int(row["window"]) != only_window:
                continue
            cell = totals[(float(row["threshold"]), int(row["window"]))]
            cell[0] += int(row["fired"])
            cell[1] += int(row["wrong"])
            cell[2] += int(row["plies_saved"])

    print(f"  pooled over {len(games)} games\n")
    print(f"  {'threshold':>9} {'window':>7} {'fired':>7} {'wrong':>6} "
          f"{'false pos':>10} {'plies saved':>12}")
    for (threshold, window) in sorted(totals):
        fired, wrong, saved = totals[(threshold, window)]
        # A handful of firings can read 0% by luck; the picker requires 30.
        rate = f"{100*wrong/fired:8.1f}%" if fired >= 30 else \
               (f"{100*wrong/fired:7.1f}%?" if fired else "        -")
        print(f"  {threshold:>9.3f} {window:>7} {fired:>7} {wrong:>6} {rate:>10} {saved:>12,}")
    print("\n  '?' means fewer than 30 firings -- not enough to trust the rate.")


def pick(root: Path, recent: int, window: int, target: float,
         fallback: float, minimum_fires: int = 30) -> float:
    """Lowest threshold whose measured false-positive rate clears `target`.

    Lower fires more often and saves more search, so the cheapest setting that
    stays under the bound is the right one. Returns `fallback` when nothing
    qualifies -- which is the conservative direction, not the aggressive one:
    a threshold nothing has vouched for should not be trusted just because it
    is the only candidate left.

    `minimum_fires` guards against a handful of firings reading 0% by luck.
    """
    totals: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for _, f in _calibration_files(root)[-recent:]:
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if int(row["window"]) != window:
                continue
            cell = totals[float(row["threshold"])]
            cell[0] += int(row["fired"])
            cell[1] += int(row["wrong"])
    for threshold in sorted(totals):
        fired, wrong = totals[threshold]
        if fired >= minimum_fires and wrong / fired <= target:
            return threshold
    return fallback


if __name__ == "__main__":
    main()
