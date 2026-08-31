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

Ratings are anchored at sl-w64b16 = 0, so they are comparable with every number
measured against it directly.

    scripts/ratings.py [artifacts/vgo-continuous/anchor.jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from vgo_training.bradley_terry import fit_ratings  # noqa: E402

# Fixed reference points get ids outside the update range so they cannot collide
# with one. sl-w64b16 is the anchor and holds rating 0 by definition.
ANCHOR_ID = 0
NAMED = {"sl-w64b16": ANCHOR_ID, "control": -1}


def model_id(path: str) -> int | None:
    stem = Path(path).name.replace(".onnx", "")
    if stem in NAMED:
        return NAMED[stem]
    match = re.fullmatch(r"update-(\d+)", stem)
    # Update 0 would collide with the anchor's id, so updates are offset by one.
    return int(match.group(1)) + 1 if match else None


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
        if a is None or b is None:
            skipped += 1
            continue
        matches.append({"a": a, "b": b,
                        "a_wins": record["candidate_wins"],
                        "b_wins": record["candidate_losses"],
                        "draws": record.get("draws", 0)})

    if not matches:
        print(f"  no usable matches in {path}"
              + (f" ({skipped} records lacked a candidate_model field)" if skipped else ""))
        return

    ratings = fit_ratings(matches, anchor=ANCHOR_ID)
    games = {}
    for m in matches:
        n = m["a_wins"] + m["b_wins"] + m["draws"]
        games[m["a"]] = games.get(m["a"], 0) + n
        games[m["b"]] = games.get(m["b"], 0) + n
    print(f"  {len(matches)} matches, {sum(games.values()) // 2} games, "
          f"{len(ratings)} models   (anchor sl-w64b16 = 0)")
    if skipped:
        print(f"  {skipped} record(s) skipped for missing model names")
    print(f"\n  {'model':<20} {'rating':>8} {'games':>7}")
    for identifier, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {label(identifier):<20} {rating:>+8.0f} {games.get(identifier, 0):>7}")


if __name__ == "__main__":
    main()
