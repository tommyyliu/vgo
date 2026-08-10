#!/usr/bin/env python3
"""Fit Elo ratings, with standard errors, from head-to-head match records.

`fit_ratings` gives the point estimates; this adds the two things needed to
read them safely.

*Standard errors.* The Bradley-Terry Fisher information is the weighted graph
Laplacian of the match network, so inverting it (with the anchor dropped, and
the prior's phantom opponent on the diagonal) gives each rating's variance. A
checkpoint with two 24-game matches carries an interval wide enough that most
apparent gaps between neighbours are not real, which is exactly the failure
mode single matches produced all night.

*Connectivity.* Ratings are only comparable within a connected component of the
match graph. The prior makes every rating finite, so a component that touches
the rest only through the prior still prints a plausible number -- one that came
from regularization rather than from any game played. Those are reported
separately rather than plotted on the same scale.

    scripts/h2h-elo.py h2h.json --games 24 --output elo.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.bradley_terry import ELO_SCALE, fit_ratings  # noqa: E402


def components(matches: list[dict]) -> list[set]:
    """Connected components of the match graph (the prior is not an edge)."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in matches:
        a, b = find(m["a"]), find(m["b"])
        if a != b:
            parent[a] = b
    groups: dict = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return sorted(groups.values(), key=len, reverse=True)


def standard_errors(matches: list[dict], ratings: dict, anchor,
                    prior_games: float) -> dict:
    """Approximate SE per rating, on the Elo scale.

    The information matrix for Bradley-Terry in log-strength is
    I[a][a] += n*p*(1-p), I[a][b] -= n*p*(1-p) per match -- a Laplacian, hence
    singular until the anchor is removed. The prior contributes n_prior*q*(1-q)
    to the diagonal only, since the phantom opponent is fixed rather than fitted.
    """
    ids = sorted(ratings)
    index = {p: i for i, p in enumerate(ids)}
    theta = {p: ratings[p] / ELO_SCALE for p in ids}
    size = len(ids)
    info = np.zeros((size, size))

    for m in matches:
        a, b = m["a"], m["b"]
        if a not in index or b not in index:
            continue
        n = float(m["a_wins"]) + float(m["b_wins"]) + float(m.get("draws", 0))
        p = 1.0 / (1.0 + math.exp(-(theta[a] - theta[b])))
        weight = n * p * (1.0 - p)
        i, j = index[a], index[b]
        info[i, i] += weight
        info[j, j] += weight
        info[i, j] -= weight
        info[j, i] -= weight

    for p in ids:
        q = 1.0 / (1.0 + math.exp(-theta[p]))
        info[index[p], index[p]] += prior_games * q * (1.0 - q)

    # Only *differences* are identified, so the covariance has to be taken with
    # the anchor dropped. Reporting each SE against the anchor would then hand
    # the anchor an interval of exactly zero -- an artifact of which checkpoint
    # happened to be picked, not a claim that it is known perfectly. Each rating
    # is instead reported against the field mean, which treats every checkpoint
    # alike and leaves the intervals directly comparable.
    keep = [i for p, i in index.items() if p != anchor]
    reduced = np.linalg.pinv(info[np.ix_(keep, keep)])
    covariance = np.zeros((size, size))
    covariance[np.ix_(keep, keep)] = reduced

    out = {}
    for p in ids:
        contrast = np.full(size, -1.0 / size)
        contrast[index[p]] += 1.0
        var = float(contrast @ covariance @ contrast)
        out[p] = math.sqrt(max(var, 0.0)) * ELO_SCALE
    return out, covariance, index


def difference(x, y, ratings, covariance, index):
    """Elo gap between two checkpoints and its SE.

    Two ratings from one fit are correlated, so the gap's variance is
    c'Cc rather than the sum of the two marginal variances -- adding them
    would overstate the error and hide differences that are real.
    """
    contrast = np.zeros(covariance.shape[0])
    contrast[index[x]] = 1.0
    contrast[index[y]] = -1.0
    se = math.sqrt(max(float(contrast @ covariance @ contrast), 0.0)) * ELO_SCALE
    return ratings[x] - ratings[y], se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h2h", type=Path)
    parser.add_argument("--games", type=int, default=24,
                        help="games per recorded match (scores are fractions)")
    parser.add_argument("--prior-games", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    raw = json.loads(arguments.h2h.read_text(encoding="utf-8"))
    # Muon keeps its own update numbers; Adam is offset, the same convention
    # joint-arena.py uses so one integer space holds both runs.
    OFFSET = 1000
    labels: dict[int, str] = {}
    matches = []
    for muon_version, opponent, score in raw:
        adam_version = int(opponent.split("v")[-1])
        a, b = muon_version, adam_version + OFFSET
        labels[a] = f"Muon v{muon_version}"
        labels[b] = f"Adam v{adam_version}"
        wins = round(score * arguments.games)
        matches.append({"a": a, "b": b, "a_wins": wins,
                        "b_wins": arguments.games - wins, "draws": 0,
                        "games": arguments.games})

    groups = components(matches)
    main_group = groups[0]
    stranded = [g for g in groups[1:]]
    print(f"matches   : {len(matches)}, {len(labels)} checkpoints")
    print(f"components: {len(groups)}  (largest {len(main_group)})")
    for g in stranded:
        print(f"  stranded: {', '.join(sorted(labels[p] for p in g))}"
              f" -- only played each other, not on the main scale")

    kept = [m for m in matches if m["a"] in main_group]
    anchor = min(main_group)
    ratings = fit_ratings(kept, anchor=anchor, prior_games=arguments.prior_games)
    errors, covariance, index = standard_errors(
        kept, ratings, anchor, arguments.prior_games)

    # Matched updates are the comparison that answers "which optimizer learns
    # faster": same amount of training, both runs, one scale.
    contrasts = []
    for version in sorted({p for p in ratings if p < OFFSET}):
        if version + OFFSET not in ratings:
            continue
        gap, se = difference(version, version + OFFSET, ratings, covariance, index)
        contrasts.append({"label": f"update {version}", "gap": gap, "se": se,
                          "sigma": abs(gap) / se if se else 0.0})
    best_muon = max((p for p in ratings if p < OFFSET), key=lambda p: ratings[p])
    best_adam = max((p for p in ratings if p >= OFFSET), key=lambda p: ratings[p])
    gap, se = difference(best_muon, best_adam, ratings, covariance, index)
    contrasts.append({"label": f"best vs best ({labels[best_muon].split()[1]}"
                               f" / {labels[best_adam].split()[1]})",
                      "gap": gap, "se": se,
                      "sigma": abs(gap) / se if se else 0.0})

    # Anchor on the weakest rated checkpoint so the whole scale is positive and
    # reads like a progress curve rather than a signed offset.
    shift = -min(ratings.values())
    rows = []
    for p in sorted(ratings, key=lambda k: ratings[k]):
        played = sum(m["games"] for m in kept if p in (m["a"], m["b"]))
        rows.append({"id": p, "label": labels[p],
                     "run": "Adam" if p >= OFFSET else "Muon",
                     "version": p - OFFSET if p >= OFFSET else p,
                     "elo": ratings[p] + shift,
                     "se": errors.get(p, float("nan")),
                     "games": played})

    print(f"\n{'checkpoint':<12}{'elo':>8}{'±1se':>8}{'games':>7}")
    for r in sorted(rows, key=lambda r: -r["elo"]):
        print(f"{r['label']:<12}{r['elo']:>8.0f}{r['se']:>8.0f}{r['games']:>7}")

    print("\nMuon minus Adam at matched updates")
    for c in contrasts:
        print(f"  {c['label']:<28}{c['gap']:>7.0f} ± {c['se']:>4.0f}"
              f"   ({c['sigma']:.1f} sigma)")

    if arguments.output:
        arguments.output.write_text(json.dumps(
            {"rows": rows, "contrasts": contrasts,
             "stranded": [sorted(labels[p] for p in g) for g in stranded],
             "games_per_match": arguments.games}, indent=2) + "\n",
            encoding="utf-8")
        print(f"\n-> {arguments.output}")


if __name__ == "__main__":
    main()
