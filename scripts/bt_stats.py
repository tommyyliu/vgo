"""Uncertainty and connectivity for Bradley-Terry fits.

`fit_ratings` returns point estimates only, which is not enough to read a
rating table safely: it says nothing about which gaps the games actually
resolve, and it happily prints a number for a checkpoint no match connects to
the field. Both scripts that fit ratings share these helpers.
"""

from __future__ import annotations

import math

import numpy as np

from vgo_training.bradley_terry import ELO_SCALE


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


def information(matches: list[dict], ratings: dict, prior_games: float):
    """Fisher information in log-strength, plus the id->index map.

    I[a][a] += n*p*(1-p) and I[a][b] -= n*p*(1-p) per match, which is the
    weighted graph Laplacian of the match network -- singular until an anchor
    is removed. The prior contributes to the diagonal only, the phantom
    opponent being fixed rather than fitted.
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
    return info, index


def covariance_of(matches: list[dict], ratings: dict, anchor,
                  prior_games: float):
    """Covariance of the fitted ratings, with the anchor's row/column zeroed."""
    info, index = information(matches, ratings, prior_games)
    size = info.shape[0]
    keep = [i for p, i in index.items() if p != anchor]
    full = np.zeros((size, size))
    full[np.ix_(keep, keep)] = np.linalg.pinv(info[np.ix_(keep, keep)])
    return full, index


def standard_errors(covariance, index) -> dict:
    """Per-rating SE on the Elo scale, taken against the field mean.

    Only differences are identified, so an SE has to be an SE *of something*.
    Measuring each against the anchor would hand the anchor an interval of
    exactly zero -- an artifact of which checkpoint was picked. The field mean
    treats every checkpoint alike and leaves the intervals comparable.
    """
    size = covariance.shape[0]
    out = {}
    for p, i in index.items():
        contrast = np.full(size, -1.0 / size)
        contrast[i] += 1.0
        var = float(contrast @ covariance @ contrast)
        out[p] = math.sqrt(max(var, 0.0)) * ELO_SCALE
    return out


def difference(x, y, ratings: dict, covariance, index) -> tuple[float, float]:
    """Elo gap between two checkpoints and its SE.

    Two ratings from one fit are correlated, so the gap's variance is c'Cc
    rather than the sum of the two marginal variances -- adding them overstates
    the error and hides differences that are real.
    """
    contrast = np.zeros(covariance.shape[0])
    contrast[index[x]] = 1.0
    contrast[index[y]] = -1.0
    se = math.sqrt(max(float(contrast @ covariance @ contrast), 0.0)) * ELO_SCALE
    return ratings[x] - ratings[y], se
