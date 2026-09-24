#!/usr/bin/env python3
"""Fit every model's rating from the accumulated arena history.

The loop no longer measures against one fixed opponent. A fixed anchor stops
measuring anything once it is outclassed -- sl-w64b16 lost 48-0 twice, which
bounds the candidate from below and says nothing else. Instead each update plays
a short match against a few sampled earlier models, and the ratings come from
fitting all of those results at once.

Any single 6-game match is noise. The graph is not: every match constrains two
players, re-fitting from scratch lets old matches inform new ratings, and the
whole history sharpens as it grows. `prior_games` in the fitter is what keeps an
undefeated model finite rather than diverging to +inf.

Ratings are anchored at sl-w64b16 = 0 when it appears in the history, so they
are comparable with every number measured against it directly. Otherwise the
first non-update model seen -- usually the run's seed -- is the anchor, and
failing that the earliest update.

    scripts/ratings.py [artifacts/vgo-continuous/anchor.jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from collections import defaultdict  # noqa: E402

from vgo_training.bradley_terry import fit_ratings  # noqa: E402

# Fixed reference points get ids outside the update range so they cannot collide
# with one. sl-w64b16 is the anchor and holds rating 0 by definition.
ANCHOR_ID = 0
NAMED = {"sl-w64b16": ANCHOR_ID, "control": -1}


def model_id(path: str) -> int:
    stem = Path(path).name.replace(".onnx", "")
    match = re.fullmatch(r"update-(\d+)", stem)
    if match:
        # Update 0 would collide with the anchor's id, so updates are offset by one.
        return int(match.group(1)) + 1
    # Any other model -- a seed, a control, a model from another run -- gets its
    # own id below every update. These used to be dropped as "missing names",
    # which silently removed a from-scratch run's matches against its seed.
    if stem not in NAMED:
        NAMED[stem] = min(NAMED.values()) - 1
    return NAMED[stem]


def label(identifier: int) -> str:
    for name, value in NAMED.items():
        if value == identifier:
            return name
    return f"update-{identifier - 1:06d}"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "artifacts/vgo-continuous/anchor.jsonl")
    raw = path.read_text()
    records = [json.loads(m) for m in
               re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw)]

    matches, skipped = [], 0
    for record in records:
        # The candidate is not named in the record, so it comes from the file's
        # own ordering: every arena run writes its opponents in sequence for one
        # candidate. `candidate_model` was added later; fall back to it when set.
        candidate = record.get("candidate_model")
        opponent = record.get("opponent_model")
        if not candidate or not opponent:
            skipped += 1
            continue
        a, b = model_id(candidate), model_id(opponent)
        matches.append({"a": a, "b": b,
                        "a_wins": record["candidate_wins"],
                        "b_wins": record["candidate_losses"],
                        "draws": record.get("draws", 0)})

    if not matches:
        print(f"  no usable matches in {path}"
              + (f" ({skipped} records lacked a candidate_model field)" if skipped else ""))
        return

    # A rating is only meaningful relative to players it is connected to. The
    # fitter anchors one id at zero and fits everything else against the prior's
    # phantom opponent, so a component with no path to the anchor still gets
    # numbers -- on a different scale, and indistinguishable from real ones.
    # Measured: sampling only recent models split the graph into {anchor, 28,
    # 33, 36} and {31, 32, 34, 39}, and update 39 printed +28 against update
    # 33's +676, which reads as a collapse and was an artifact.
    adjacency: dict[int, set[int]] = defaultdict(set)
    for m in matches:
        adjacency[m["a"]].add(m["b"])
        adjacency[m["b"]].add(m["a"])
    players = {m["a"] for m in matches} | {m["b"] for m in matches}
    anchor = ANCHOR_ID if ANCHOR_ID in players else min(players)
    anchored, stack = set(), [anchor]
    while stack:
        node = stack.pop()
        if node in anchored:
            continue
        anchored.add(node)
        stack.extend(adjacency[node] - anchored)

    ratings = fit_ratings(matches, anchor=anchor)
    floating = sorted(set(ratings) - anchored)
    games = {}
    for m in matches:
        n = m["a_wins"] + m["b_wins"] + m["draws"]
        games[m["a"]] = games.get(m["a"], 0) + n
        games[m["b"]] = games.get(m["b"], 0) + n
    print(f"  {len(matches)} matches, {sum(games.values()) // 2} games, "
          f"{len(ratings)} models   (anchor {label(anchor)} = 0)")
    if skipped:
        print(f"  {skipped} record(s) skipped for missing model names")
    print(f"\n  {'model':<20} {'rating':>8} {'games':>7}")
    for identifier, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        if identifier in floating:
            continue
        print(f"  {label(identifier):<20} {rating:>+8.0f} {games.get(identifier, 0):>7}")
    if floating:
        print(f"\n  NOT COMPARABLE -- no match path to {label(anchor)}, so these are")
        print("  fitted against the prior rather than the anchor and sit on their own")
        print("  scale. They need a match against an anchored model to join.")
        for identifier in sorted(floating, key=lambda i: -ratings[i]):
            print(f"    {label(identifier):<18} {ratings[identifier]:>+8.0f} "
                  f"{games.get(identifier, 0):>7}  (floating)")


if __name__ == "__main__":
    main()
