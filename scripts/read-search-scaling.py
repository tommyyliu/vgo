#!/usr/bin/env python3
"""Read a search-scaling sweep: Elo against simulation count.

`runs/search-scaling.sh` plays one network against itself at two budgets, so
each record is "N simulations scored X against 800". That is not a rating —
two seats at different budgets are two different players — so this deliberately
does not fit a Bradley-Terry model over the field or touch ratings.json. Each
point stands alone as a direct measurement against the reference budget.

The number the client-side bot needs is the Elo cost of the budget a browser can
afford, which is read straight off the table. The fitted slope is secondary and
mostly a sanity check: engines usually lose 100-150 Elo per halving of search,
and a much steeper slope says the policy prior is carrying little of the load.

    scripts/read-search-scaling.py artifacts/search-scaling/records-scaling.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def records(text: str) -> list[dict]:
    """Every top-level JSON object in `text`, by brace depth."""
    found, depth, start = [], 0, None
    for position, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start:position + 1])
                except json.JSONDecodeError:
                    continue
                if value.get("schema") == "vgo.arena.v1":
                    found.append(value)
    return found


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400 * math.log10(score / (1 - score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    arguments = parser.parse_args()

    found = records(arguments.records.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit(f"no vgo.arena.v1 records in {arguments.records}")

    rows = []
    for record in found:
        budget = int(record["simulations_per_move"])
        reference = int(record["opponent_simulations_per_move"])
        wins = int(record["candidate_wins"])
        losses = int(record["candidate_losses"])
        draws = int(record.get("draws", 0))
        played = wins + losses + draws
        if played == 0:
            continue
        score = (wins + 0.5 * draws) / played
        error = math.sqrt(max(score * (1 - score), 1e-12) / played)
        # A record with no wins (or no losses) has no finite maximum-likelihood
        # rating: the fit keeps sliding as long as the count grows, so a point
        # estimate would be an artifact of wherever the clamp sits. Report the
        # one-sided Clopper-Pearson bound instead, which is what the games
        # actually support. Same reason naive cannot anchor a rating scale.
        if wins + 0.5 * draws == 0:
            bound = 1 - 0.05 ** (1 / played)
            rows.append({
                "budget": budget, "reference": reference,
                "wins": wins, "losses": losses, "games": played,
                "score": score, "elo": None, "bound": elo(bound),
            })
            continue
        rows.append({
            "budget": budget, "reference": reference,
            "wins": wins, "losses": losses, "games": played,
            "score": score, "elo": elo(score), "bound": None,
            "low": elo(score - 1.96 * error), "high": elo(score + 1.96 * error),
        })
    rows.sort(key=lambda row: row["budget"])

    references = {row["reference"] for row in rows}
    if len(references) != 1:
        print(f"warning: mixed reference budgets {sorted(references)}; "
              "these points are not on one axis")
    reference = rows[0]["reference"]

    print(f"one network against itself, reference seat {reference} simulations")
    print()
    print(f"  {'sims':>5} {'record':>9} {'score':>7} {'Elo vs ' + str(reference):>14}"
          f"  {'95% CI':>18}")
    for row in rows:
        if row["elo"] is None:
            print(f"  {row['budget']:>5} {row['wins']:>4}-{row['losses']:<4} "
                  f"{row['score'] * 100:>6.1f}% {'unbounded':>13}  "
                  f"{'< ' + format(row['bound'], '+.0f') + ' (95%)':>18}")
            continue
        interval = f"[{row['low']:+.0f}, {row['high']:+.0f}]"
        print(f"  {row['budget']:>5} {row['wins']:>4}-{row['losses']:<4} "
              f"{row['score'] * 100:>6.1f}% {row['elo']:>+13.0f}  {interval:>18}")

    # Slope per halving, from the points below the reference. A straight line in
    # log2(simulations) is the usual first-order model; it is reported only to be
    # compared against the 100-150 Elo/halving that engines typically show.
    below = [row for row in rows if row["budget"] < reference]
    if len([r for r in below if r["elo"] is not None]) >= 2:
        below = [row for row in below if row["elo"] is not None]
        xs = [math.log2(row["budget"]) for row in below]
        ys = [row["elo"] for row in below]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator > 0:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
            print()
            print(f"slope over the {n} points below the reference: "
                  f"{slope:+.0f} Elo per doubling of simulations")
            print("  (engines typically show 100-150; much steeper means the "
                  "policy prior is carrying little)")

    print()
    print("Not a rating. Two seats at different budgets are different players, so")
    print("these records must never be pooled into the Elo scale.")


if __name__ == "__main__":
    main()
