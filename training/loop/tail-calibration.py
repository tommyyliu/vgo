#!/usr/bin/env python3
"""Build the search tail's schedule from recent games.

The schedule says, for each board band and stage of a game, how far behind a side
must be before its game gets cheaper search. It comes from one measurement: when a
side trailed like this, how often did it go on to win or tie? The lowest value
whose comeback rate is clearly under the target wins the cell.

A cell without enough games has a wide bound, so it stays closed, which is why a
new run starts with full search and needs no special case. With a 5% target, a cell
needs at least 73 games even with no comebacks at all.

The target is a plain setting. Nothing derives it, and a tail that can end if a
game recovers makes a wrong call cheap.

Games still running hold back the longest boards, and values after a tail starts
come from cheaper search, so read these rates as estimates.

    training/loop/tail-calibration.py artifacts/vgo-bulk/games --output schedule.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "vgo.tail-schedule.v1"
# Board bands as widest units across. None takes every larger board.
BOARDS = (18.5, 37.5, None)
# Where each stage starts, as a share of the ply cap. No game starts a tail in the
# first quarter: at 128 input, 38-unit boards were wrong one time in five there.
STAGES = (0.25, 0.5, 0.75)
# Trailing root value, loosest first. 0.8 is a 10% win chance.
VALUES = (0.80, 0.90, 0.95, 0.99)


def board_band(radius: float) -> int:
    units = 1.0 / radius
    return next(
        index for index, limit in enumerate(BOARDS) if limit is None or units <= limit
    )


def stage_band(ply: int, maximum_plies: int) -> int | None:
    progress = ply / maximum_plies
    found = None
    for index, start in enumerate(STAGES):
        if progress >= start:
            found = index
    return found


def recent_games(root: Path, count: int) -> list[dict]:
    """The newest games a model played that recorded root values, oldest first.

    The naive first generation is skipped: its values are a heuristic, and a
    schedule built from them would say nothing about a net.
    """
    found: list[tuple[int, Path]] = []
    for generation in root.iterdir():
        if not generation.is_dir():
            continue
        for game in generation.iterdir():
            number = re.fullmatch(r"game-(\d+)", game.name)
            if number and (game / "manifest.json").is_file():
                found.append((int(number.group(1)), game))
    found.sort()
    games = []
    for _, game in reversed(found):
        if len(games) >= count:
            break
        if "behavior_model_sha256" not in json.loads((game / "manifest.json").read_text()):
            continue
        record = json.loads((game / "games.jsonl").read_text().splitlines()[0])
        if "root_values" in record:
            games.append(record)
    games.reverse()
    return games


def first_entries(
    values: list[float], maximum_plies: int, threshold: float, window: int
) -> dict[tuple[int, int], int]:
    """The ply each seat first held a streak in each stage, keyed by (seat, stage).

    Black moves on even plies, so a seat's plies are that seat's own turns, and a
    streak counts those. Root values are from Black's view.
    """
    streak = [0, 0]
    entries: dict[tuple[int, int], int] = {}
    for ply, black_value in enumerate(values):
        seat = ply % 2
        value = black_value if seat == 0 else -black_value
        stage = stage_band(ply, maximum_plies)
        streak[seat] = streak[seat] + 1 if stage is not None and value <= -threshold else 0
        if streak[seat] >= window:
            entries.setdefault((seat, stage), ply)
    return entries


def comeback_table(games: list[dict], window: int) -> dict[tuple[int, int, float], list[int]]:
    """(board, stage, value) to [comebacks, games].

    A seat counts once per cell, because plies within a game aren't independent.
    """
    table: dict[tuple[int, int, float], list[int]] = defaultdict(lambda: [0, 0])
    for game in games:
        board = board_band(game["radius"])
        for threshold in VALUES:
            entries = first_entries(
                game["root_values"], game["maximum_plies"], threshold, window
            )
            for seat, stage in entries:
                utility = game["black_utility"] if seat == 0 else -game["black_utility"]
                cell = table[(board, stage, threshold)]
                cell[0] += int(utility >= 0)
                cell[1] += 1
    return table


def upper_bound(successes: int, trials: int) -> float:
    """Upper end of a 95% Wilson interval. With no trials, nothing is known."""
    if trials == 0:
        return 1.0
    z = 1.96
    rate = successes / trials
    centre = rate + z * z / (2 * trials)
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
    return (centre + spread) / (1 + z * z / trials)


def schedule(table: dict, target: float) -> dict:
    """The lowest value in each cell whose comeback rate is clearly under target."""
    boards = []
    for board, limit in enumerate(BOARDS):
        thresholds = []
        for stage in range(len(STAGES)):
            passing = (
                value
                for value in VALUES
                if upper_bound(*table.get((board, stage, value), (0, 0))) <= target
            )
            thresholds.append(next(passing, None))
        boards.append({"up_to_units": limit, "thresholds": thresholds})
    return {"schema": SCHEMA, "stages": list(STAGES), "boards": boards}


def describe(games: list[dict], table: dict, built: dict) -> list[str]:
    """One line per board, for the log."""
    lines = []
    for index, board in enumerate(built["boards"]):
        played = [game for game in games if board_band(game["radius"]) == index]
        plies = sum(game["plies"] for game in played)
        tail = sum(end - start for game in played for start, end in game.get("tail_spans", []))
        share = f"{100 * tail / plies:.0f}%" if plies else "none"
        cells = []
        for stage, start in enumerate(STAGES):
            threshold = board["thresholds"][stage]
            if threshold is None:
                cells.append(f"{start:.0%}+ closed")
                continue
            comebacks, trials = table[(index, stage, threshold)]
            cells.append(f"{start:.0%}+ at {threshold} ({comebacks}/{trials} came back)")
        limit = board["up_to_units"]
        name = f"up to {limit:g}" if limit else "larger"
        lines.append(f"[tail] board {name}: {len(played)} games, tail {share} of plies; " + ", ".join(cells))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=5)
    arguments = parser.parse_args()
    games = recent_games(arguments.root, arguments.games)
    table = comeback_table(games, arguments.window)
    built = schedule(table, arguments.target)
    built["games"] = len(games)
    built["target"] = arguments.target
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(built, indent=1) + "\n")
    for line in describe(games, table, built):
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
