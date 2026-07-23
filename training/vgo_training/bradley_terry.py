"""Batch Bradley-Terry rating fit for the arena Elo pool.

Given a match history (each match records wins/losses/draws between two
generations), solve every generation's rating by maximum-likelihood, anchored so
the anchor generation has rating 0. Re-run from scratch each iteration; as
matches accumulate every generation's rating gets more accurate.

Ratings are reported on the Elo scale (400 / ln(10) per natural-log unit). The
solver is the standard Bradley-Terry minorization-maximization (MM) iteration,
which is stable and monotonically increases the likelihood.
"""
from __future__ import annotations

import math
from collections import defaultdict

ELO_SCALE = 400.0 / math.log(10.0)


def fit_ratings(
    matches: list[dict],
    anchor: int = 0,
    iterations: int = 1000,
    tolerance: float = 1e-9,
    prior_games: float = 2.0,
) -> dict[int, float]:
    """Fit Elo-scale ratings from a match history.

    Each match dict has integer generation ids "a" and "b" and the counts
    "a_wins", "b_wins", "draws". A draw is half a win to each side.

    `prior_games` adds that many virtual even games against a rating-0 phantom
    opponent for every player, which regularizes generations with few or lopsided
    matches (otherwise an undefeated net would diverge to +inf). Returns
    {generation_id: elo_rating}, anchor forced to 0.
    """
    # w[p] = total (fractional) wins by p; n[p][o] = games played between p and o.
    w: dict[int, float] = defaultdict(float)
    n: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    players: set[int] = {anchor}
    for match in matches:
        a, b = int(match["a"]), int(match["b"])
        players.add(a)
        players.add(b)
        aw = float(match.get("a_wins", 0)) + 0.5 * float(match.get("draws", 0))
        bw = float(match.get("b_wins", 0)) + 0.5 * float(match.get("draws", 0))
        games = aw + bw
        w[a] += aw
        w[b] += bw
        n[a][b] += games
        n[b][a] += games

    # gamma[p] = exp(theta_p); update multiplicatively. Phantom opponent has gamma 1.
    gamma: dict[int, float] = {p: 1.0 for p in players}
    ids = sorted(players)
    for _ in range(iterations):
        max_rel = 0.0
        new_gamma: dict[int, float] = {}
        for p in ids:
            if p == anchor:
                new_gamma[p] = 1.0
                continue
            # numerator: wins by p, plus half of the prior_games phantom matches.
            numerator = w[p] + 0.5 * prior_games
            # denominator: sum over opponents of games_po / (gamma_p + gamma_o),
            # plus the phantom term prior_games / (gamma_p + 1).
            denom = prior_games / (gamma[p] + 1.0)
            for o, games in n[p].items():
                denom += games / (gamma[p] + gamma[o])
            if denom <= 0.0:
                new_gamma[p] = gamma[p]
                continue
            g = numerator / denom
            max_rel = max(max_rel, abs(g - gamma[p]) / (gamma[p] + 1e-12))
            new_gamma[p] = g
        gamma = new_gamma
        if max_rel < tolerance:
            break

    anchor_gamma = gamma.get(anchor, 1.0)
    return {
        p: math.log(gamma[p] / anchor_gamma) * ELO_SCALE for p in ids
    }
