#!/usr/bin/env python3
"""Fit the big-board checkpoint curve from `artifacts/bigboard-curve/matches.jsonl`.

`scripts/ratings.py` anchors at sl-w64b16, which has never played on the 1/38
board. Nothing in this graph has a path to it, so that fitter would report every
model as floating and on its own scale. This anchors at the *earliest* checkpoint
present instead, which is the question being asked: how much did the run gain,
on the board it actually trains on, since it started.

Ratings are therefore gains over that checkpoint, not comparable with any number
measured at 1/18. Differences within this table are what mean something.

    scripts/bigboard-ratings.py [artifacts/bigboard-curve/matches.jsonl]
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from vgo_training.bradley_terry import fit_ratings  # noqa: E402


def model_id(path: str) -> int | None:
    match = re.fullmatch(r"update-(\d+)", Path(path).name.replace(".onnx", ""))
    return int(match.group(1)) if match else None


def wilson(wins: float, games: int) -> tuple[float, float]:
    """95% interval for a score rate, so a flat link is visibly flat."""
    if games <= 0:
        return (0.0, 1.0)
    z, p, n = 1.959964, wins / games, games
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "artifacts/bigboard-curve/matches.jsonl")
    if not path.exists():
        print(f"  no matches yet at {path}")
        return
    records = [json.loads(m) for m in
               re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", path.read_text())]

    matches, rows, skipped = [], [], 0
    for record in records:
        a = model_id(record.get("candidate_model", ""))
        b = model_id(record.get("opponent_model", ""))
        if a is None or b is None:
            skipped += 1
            continue
        wins = record["candidate_wins"]
        losses = record["candidate_losses"]
        draws = record.get("draws", 0)
        games = wins + losses + draws
        matches.append({"a": a, "b": b, "a_wins": wins, "b_wins": losses,
                        "draws": draws})
        rows.append((a, b, wins + 0.5 * draws, games, record))

    if not matches:
        print(f"  no usable matches in {path}")
        return

    # Anchor at the earliest checkpoint in the graph. `fit_ratings` pins its
    # anchor to 0 and regularizes everything else against a rating-0 phantom, so
    # an undefeated checkpoint stays finite rather than running to +inf.
    anchor = min(min(m["a"], m["b"]) for m in matches)
    ratings = fit_ratings(matches, anchor=anchor)

    # A rating only means something relative to what it has a match path to.
    adjacency: dict[int, set[int]] = defaultdict(set)
    for m in matches:
        adjacency[m["a"]].add(m["b"])
        adjacency[m["b"]].add(m["a"])
    connected, stack = set(), [anchor]
    while stack:
        node = stack.pop()
        if node in connected:
            continue
        connected.add(node)
        stack.extend(adjacency[node] - connected)

    played: dict[int, int] = defaultdict(int)
    for a, b, _, games, _ in rows:
        played[a] += games
        played[b] += games

    total = sum(games for _, _, _, games, _ in rows)
    print(f"  {len(rows)} links, {total} games, {len(ratings)} checkpoints"
          f"   (anchor update-{anchor:06d} = 0)")
    if skipped:
        print(f"  {skipped} record(s) skipped for missing model names")

    print(f"\n  {'link':<30} {'score':>7} {'95% CI':>16} {'games':>6}"
          f" {'plies':>6} {'capped':>7}")
    for a, b, score, games, record in rows:
        low, high = wilson(score, games)
        capped = record.get("reached_ply_cap", 0)
        print(f"  update-{a:06d} vs update-{b:06d}   {score / games:>7.3f}"
              f"  [{low:>5.3f}, {high:>5.3f}] {games:>6}"
              f" {record.get('average_plies', 0):>6.0f} {capped:>7}")

    print(f"\n  {'checkpoint':<20} {'rating':>8} {'games':>7}")
    for identifier, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        if identifier not in connected:
            continue
        print(f"  update-{identifier:06d}{'':<6} {rating:>+8.0f}"
              f" {played.get(identifier, 0):>7}")

    floating = sorted(set(ratings) - connected)
    if floating:
        print(f"\n  NOT COMPARABLE -- no match path to update-{anchor:06d}:")
        for identifier in floating:
            print(f"    update-{identifier:06d} {ratings[identifier]:>+8.0f}"
                  f" {played.get(identifier, 0):>7}  (floating)")

    # Reaching the cap is the normal ending on this board -- these bots do not
    # resign, and by 312 plies the board is full and the position settled, so
    # the game is decided on area. Reported as context, not as a warning; the
    # number to watch is `truncated`, which is a game that produced no result.
    capped = sum(r.get("reached_ply_cap", 0) for _, _, _, _, r in rows)
    truncated = sum(r.get("truncated", 0) for _, _, _, _, r in rows)
    print(f"\n  {capped}/{total} games decided by area at the ply cap"
          f"{f', {truncated} produced no result' if truncated else ''}")


if __name__ == "__main__":
    main()
