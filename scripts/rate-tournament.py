#!/usr/bin/env python3
"""Fit one Elo scale from `vgo-tournament` output, with uncertainty.

`vgo-tournament` writes `vgo.arena.v1` records keyed by *model path*, while
`fit_ratings` keys on integers, so this maps each path back to the run and
update it came from and offsets one run by 1000 -- the same convention
`joint-arena.py` uses, and for the same reason: both runs number their updates
from zero.

The output file is named `.jsonl` but is not line-delimited. The tournament
pretty-prints each record across several lines, so records are recovered by
brace depth rather than by splitting on newlines. That also means a partially
written file parses fine, which is what an incremental writer would produce.

Emits the same JSON shape `h2h-elo.py` does, so the dashboard reads either.

    scripts/rate-tournament.py artifacts/joint-tournament/records.jsonl
    scripts/rate-tournament.py records.jsonl --json elo.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bt_stats import (  # noqa: E402
    components, covariance_of, difference, standard_errors,
)
from vgo_training.bradley_terry import fit_ratings  # noqa: E402

OFFSET = 1000
_UPDATE = re.compile(r"update-(\d+)")


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


NAIVE = -1


def identify(path: str) -> tuple[str, int]:
    """(run name, update number) for a checkpoint path.

    The naive evaluator is written as the bare token "naive" rather than a
    path. It gets its own id so it can anchor the fit: it is the only player
    whose strength is independent of any run, which is what makes ratings
    comparable across tournaments instead of relative to whichever checkpoint
    happened to be weakest in the field.
    """
    if path == "naive":
        return "naive", NAIVE
    parts = Path(path).parts
    match = _UPDATE.search(path)
    if "updates" not in parts or match is None:
        raise SystemExit(f"cannot place checkpoint path: {path}")
    return parts[parts.index("updates") - 1], int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--prior-games", type=float, default=0.25)
    arguments = parser.parse_args()

    found = records(arguments.records.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit(f"no vgo.arena.v1 records in {arguments.records}")

    # Run order is discovered rather than assumed: the launch script interleaves
    # the two runs, so first appearance is what fixes which takes the offset.
    order: list[str] = []
    for record in found:
        for side in ("candidate_model", "opponent_model"):
            run, _ = identify(record[side])
            if run != "naive" and run not in order:
                order.append(run)
    if len(order) > 2:
        raise SystemExit(f"expected at most two runs, found {order}")

    def key(path: str) -> int:
        run, version = identify(path)
        # Naive is not part of any run, so it keeps a fixed id outside the
        # per-run offset space.
        return NAIVE if run == "naive" else version + order.index(run) * OFFSET

    # "muon"/"adam" read better on a chart than the artifact directory names.
    def nickname(run: str) -> str:
        return "Muon" if "muon" in run.lower() else "Adam"

    labels: dict[int, str] = {}
    matches, undecided = [], 0
    for record in found:
        if int(record["completed"]) == 0:
            undecided += 1
            continue
        a, b = key(record["candidate_model"]), key(record["opponent_model"])
        for side, identifier in ((record["candidate_model"], a),
                                 (record["opponent_model"], b)):
            run, version = identify(side)
            labels[identifier] = ("naive" if run == "naive"
                                  else f"{nickname(run)} v{version}")
        matches.append({"a": a, "b": b,
                        "a_wins": int(record["candidate_wins"]),
                        "b_wins": int(record["candidate_losses"]),
                        "draws": int(record["draws"]),
                        "games": int(record["games"])})

    games = sum(m["a_wins"] + m["b_wins"] + m["draws"] for m in matches)
    print(f"records   : {len(found)} pairings, {games} decided games"
          + (f" ({undecided} undecided)" if undecided else ""))

    groups = components(matches)
    stranded = groups[1:]
    for g in stranded:
        print(f"  stranded: {', '.join(sorted(labels[p] for p in g))}")
    main_group = groups[0]
    kept = [m for m in matches if m["a"] in main_group]
    # Anchor on naive when it played: it is the only fixed point, so ratings
    # from different tournaments are then on one absolute scale. Without it the
    # zero lands on whichever checkpoint happened to be weakest, which moves
    # every time the field changes and makes runs incomparable across sessions.
    anchored_on_naive = NAIVE in main_group
    anchor = NAIVE if anchored_on_naive else min(main_group)
    print("anchor    : " + ("naive (absolute scale)" if anchored_on_naive
                            else f"{labels[anchor]} (relative -- no naive in field)"))

    ratings = fit_ratings(kept, anchor=anchor, prior_games=arguments.prior_games)
    covariance, index = covariance_of(kept, ratings, anchor, arguments.prior_games)
    errors = standard_errors(covariance, index)

    # With naive anchored at 0 the scale already means something; shifting it
    # would throw that away. Only re-zero when there is no absolute reference.
    shift = 0.0 if anchored_on_naive else -min(ratings.values())
    rows = []
    for p in sorted(ratings):
        played = [m for m in kept if p in (m["a"], m["b"])]
        total = sum(m["a_wins"] + m["b_wins"] + m["draws"] for m in played)
        rows.append({"id": p, "label": labels[p],
                     "run": "Naive" if p == NAIVE else labels[p].split()[0],
                     "version": None if p == NAIVE else p - (p // OFFSET) * OFFSET,
                     "elo": ratings[p] + shift, "se": errors[p],
                     "games": total})

    if anchored_on_naive:
        naive_row = next(r for r in rows if r["id"] == NAIVE)
        print(f"\nnaive     0 Elo by definition, {naive_row['games']} games played")
    for index_of_run, run in enumerate(order):
        print(f"\n{nickname(run)}  ({run})")
        print(f"{'update':>8}{'elo':>8}{'±1se':>7}{'games':>7}")
        for r in rows:
            if r["id"] == NAIVE or r["id"] // OFFSET != index_of_run:
                continue
            print(f"{r['version']:>8}{r['elo']:>8.0f}{r['se']:>7.0f}{r['games']:>7}")

    # Matched updates: same amount of training, both runs, one scale -- the
    # comparison that answers which optimizer learns faster.
    # Which run carries the offset is discovered from first appearance, so the
    # subtraction has to be ordered by *nickname*, not by id. Ordering it by id
    # silently reports Adam minus Muon whenever Adam happens to appear first,
    # which reverses every sign on the chart.
    muon_slot = next((i for i, run in enumerate(order) if "muon" in run.lower()), None)
    contrasts = []
    if muon_slot is not None and len(order) == 2:
        adam_slot = 1 - muon_slot
        for version in sorted({r["version"] for r in rows
                               if r["version"] is not None}):
            muon = version + muon_slot * OFFSET
            adam = version + adam_slot * OFFSET
            if muon not in ratings or adam not in ratings:
                continue
            gap, se = difference(muon, adam, ratings, covariance, index)
            contrasts.append({"label": f"update {version}", "gap": gap, "se": se,
                              "sigma": abs(gap) / se if se else 0.0})
    print("\nMuon minus Adam at matched updates")
    for c in contrasts:
        print(f"  {c['label']:<14}{c['gap']:>7.0f} ± {c['se']:>4.0f}"
              f"   ({c['sigma']:.1f} sigma)")

    if arguments.json:
        arguments.json.write_text(json.dumps(
            {"rows": rows, "contrasts": contrasts,
             "stranded": [sorted(labels[p] for p in g) for g in stranded],
             "games_per_match": 8}, indent=2) + "\n", encoding="utf-8")
        print(f"\n-> {arguments.json}")


if __name__ == "__main__":
    main()
