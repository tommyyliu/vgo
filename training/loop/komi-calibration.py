#!/usr/bin/env python3
"""Recentre self-play komi on recent results.

Fits Black's win rate on recent model games as a logistic in board size and komi:

    logit P(Black wins) = w0 + w1 * size + w2 * points

with size the units across (1/radius) and points the komi in points (komi area
/ 4r^2, the convention in which coefficient 28 is 7 points). Where the fit says
more komi costs Black -- w2 clearly negative -- the even point on a board is
-(w0 + w1 * size) / w2, and the centre moves toward it at sizes 18 and 38 by at
most --max-step points per refit. Where it does not, the centre stays put: early
in a run the agents barely respond to komi, and an even point read off that fit
is noise.

The spread is fixed in points. It exists so the value head sees komi vary and
learns what it is worth, not to follow how noisy the results are. The fitted
slope is recorded each time: once it is clear, a spread chosen to cover, say,
25-75% Black wins would be the principled one.

    training/loop/komi-calibration.py artifacts/vgo-bulk/games --output komi.json \\
        [--previous earlier.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SCHEMA = "vgo.komi-schedule.v1"
# The fixed centre this replaces: coefficient 28 is 7 points on every board, and
# the old range of 0.017 to 0.137 around a 0.077 mean was +-78% of it.
DEFAULT_CENTRE = 7.0
DEFAULT_SPREAD = 0.78 * DEFAULT_CENTRE
# The two sizes the centre is set at; the line through them gives the rest.
SIZES = (18.0, 38.0)
# How far the centre may ever go, in points.
LIMITS = (-10.0, 20.0)


def recent_games(root: Path, count: int) -> list[dict]:
    """The newest games a model played, oldest first; the naive generation is skipped."""
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
        try:
            games.append(json.loads((game / "games.jsonl").read_text().splitlines()[0]))
        except (json.JSONDecodeError, IndexError):
            continue  # a game still being written
    games.reverse()
    return games


def rows(games: list[dict]) -> list[tuple[float, float, float]]:
    """(size, komi in points, Black's result: 1 win, 0 loss, 0.5 tie) per game."""
    out = []
    for g in games:
        radius = float(g["radius"])
        points = float(g["komi"]) / (4.0 * radius * radius)
        utility = float(g["black_utility"])
        out.append((1.0 / radius, points, 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5))
    return out


def fit(data: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    """Logistic regression by Newton's method: weights and their standard errors."""
    w = [0.0, 0.0, 0.0]
    covariance = [[0.0] * 3 for _ in range(3)]
    for _ in range(50):
        gradient = [0.0, 0.0, 0.0]
        hessian = [[1e-6 if i == j else 0.0 for j in range(3)] for i in range(3)]
        for size, points, y in data:
            x = (1.0, size, points)
            z = sum(a * b for a, b in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            for i in range(3):
                gradient[i] += (y - p) * x[i]
                for j in range(3):
                    hessian[i][j] += p * (1.0 - p) * x[i] * x[j]
        covariance = invert(hessian)
        step = [sum(covariance[i][j] * gradient[j] for j in range(3)) for i in range(3)]
        w = [a + b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return w, [math.sqrt(max(covariance[i][i], 0.0)) for i in range(3)]


def invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    rows_ = [row[:] + [float(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows_[r][col]))
        rows_[col], rows_[pivot] = rows_[pivot], rows_[col]
        scale = rows_[col][col]
        rows_[col] = [v / scale for v in rows_[col]]
        for r in range(size):
            if r != col and rows_[r][col]:
                factor = rows_[r][col]
                rows_[r] = [v - factor * c for v, c in zip(rows_[r], rows_[col])]
    return [row[size:] for row in rows_]


def centre_at(schedule: dict, size: float) -> float:
    return schedule["centre_points"] + schedule["points_per_unit"] * size


def recentre(games: list[dict], previous: dict, minimum_games: int, max_step: float,
             minimum_z: float) -> dict:
    data = rows(games)
    old = [centre_at(previous, s) for s in SIZES]
    built = {"schema": SCHEMA, "games": len(data), "spread_points": previous["spread_points"]}
    if data:
        built["black_win_rate"] = sum(y for _, _, y in data) / len(data)
    reason, new = "too few games", old
    if len(data) >= minimum_games:
        w, se = fit(data)
        z = w[2] / se[2] if se[2] > 0 else 0.0
        built.update({"slope_per_point": w[2], "slope_z": z})
        if z < -minimum_z:
            fair = [-(w[0] + w[1] * s) / w[2] for s in SIZES]
            built["fair_points"] = {f"{s:.0f}": f for s, f in zip(SIZES, fair)}
            new = [min(LIMITS[1], max(LIMITS[0], o + max(-max_step, min(max_step, f - o))))
                   for o, f in zip(old, fair)]
            reason = "moved toward the even point"
        else:
            reason = "komi does not yet move the result clearly"
    slope = (new[1] - new[0]) / (SIZES[1] - SIZES[0])
    built.update({
        "centre_points": new[0] - slope * SIZES[0],
        "points_per_unit": slope,
        "centre_at": {f"{s:.0f}": c for s, c in zip(SIZES, new)},
        "reason": reason,
    })
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=None,
                        help="the last schedule; without one the centre starts at 7 points")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--minimum-games", type=int, default=300)
    parser.add_argument("--max-step", type=float, default=1.0,
                        help="most the centre may move per refit, in points")
    parser.add_argument("--minimum-z", type=float, default=2.0,
                        help="how clearly komi must lower Black's win rate before moving")
    arguments = parser.parse_args()
    previous = {"centre_points": DEFAULT_CENTRE, "points_per_unit": 0.0,
                "spread_points": DEFAULT_SPREAD}
    if arguments.previous and arguments.previous.is_file():
        previous = json.loads(arguments.previous.read_text())
    built = recentre(recent_games(arguments.root, arguments.games), previous,
                     arguments.minimum_games, arguments.max_step, arguments.minimum_z)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(built, indent=2) + "\n")
    at = built["centre_at"]
    print(f"[komi] {built['games']} games, Black won {built.get('black_win_rate', float('nan')):.0%}; "
          f"{built['reason']}; centre {at['18']:+.2f} points at 18 units, {at['38']:+.2f} at 38"
          + (f"; slope {built['slope_per_point']:+.3f} per point (z {built['slope_z']:+.1f})"
             if "slope_per_point" in built else ""))


if __name__ == "__main__":
    main()
