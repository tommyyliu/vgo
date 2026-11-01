"""Bradley-Terry ratings with uncertainty, for arena results.

Matches vgo's own fitter (same prior, same scale), and adds what it doesn't report:
a standard error, the connected component, and whether a rating is only a bound.
"""

from __future__ import annotations

import math
from collections import defaultdict

ELO = 400.0 / math.log(10.0)
PRIOR_GAMES = 2.0


def fit(matches: list[dict], anchor) -> dict[object, float]:
    """Minorization-maximization fit. Returns Elo ratings with anchor at 0."""
    wins: dict = defaultdict(float)
    games: dict = defaultdict(lambda: defaultdict(float))
    players = {anchor}
    for m in matches:
        a, b = m["a"], m["b"]
        players |= {a, b}
        wins[a] += m["a_wins"] + 0.5 * m.get("draws", 0)
        wins[b] += m["b_wins"] + 0.5 * m.get("draws", 0)
        total = m["a_wins"] + m["b_wins"] + m.get("draws", 0)
        games[a][b] += total
        games[b][a] += total
    gamma = {p: 1.0 for p in players}
    for _ in range(2000):
        change = 0.0
        updated = {}
        for p in players:
            if p == anchor:
                updated[p] = 1.0
                continue
            denominator = PRIOR_GAMES / (gamma[p] + 1.0)
            denominator += sum(n / (gamma[p] + gamma[o]) for o, n in games[p].items())
            value = (wins[p] + 0.5 * PRIOR_GAMES) / denominator
            change = max(change, abs(value - gamma[p]) / gamma[p])
            updated[p] = value
        gamma = updated
        if change < 1e-9:
            break
    return {p: ELO * math.log(g) for p, g in gamma.items()}


def standard_errors(matches: list[dict], ratings: dict, anchor) -> dict[object, float]:
    """Elo SE of each rating relative to the anchor, from the Fisher information."""
    players = sorted((p for p in ratings if p != anchor), key=str)
    index = {p: i for i, p in enumerate(players)}
    size = len(players)
    if not size:
        return {}
    info = [[0.0] * size for _ in range(size)]

    def weight(a: float, b: float) -> float:
        p = 1.0 / (1.0 + math.exp((b - a) / ELO))
        return p * (1.0 - p)

    for p in players:
        info[index[p]][index[p]] += PRIOR_GAMES * weight(ratings[p], 0.0)
    for m in matches:
        a, b = m["a"], m["b"]
        total = m["a_wins"] + m["b_wins"] + m.get("draws", 0)
        w = total * weight(ratings[a], ratings[b])
        if a != anchor:
            info[index[a]][index[a]] += w
        if b != anchor:
            info[index[b]][index[b]] += w
        if a != anchor and b != anchor:
            info[index[a]][index[b]] -= w
            info[index[b]][index[a]] -= w
    inverse = invert(info)
    return {p: ELO * math.sqrt(max(inverse[i][i], 0.0)) for p, i in index.items()}


def invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    rows = [row[:] + [float(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for r in range(size):
            if r != col and rows[r][col]:
                factor = rows[r][col]
                rows[r] = [v - factor * c for v, c in zip(rows[r], rows[col])]
    return [row[size:] for row in rows]


def connected(matches: list[dict], anchor) -> set:
    neighbours: dict = defaultdict(set)
    for m in matches:
        neighbours[m["a"]].add(m["b"])
        neighbours[m["b"]].add(m["a"])
    seen, stack = set(), [anchor]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(neighbours[node] - seen)
    return seen


def only_a_bound(matches: list[dict], player) -> bool:
    """True when a player never lost or never won, so the data can't place it."""
    won = lost = 0.0
    for m in matches:
        if player == m["a"]:
            won, lost = won + m["a_wins"], lost + m["b_wins"]
        elif player == m["b"]:
            won, lost = won + m["b_wins"], lost + m["a_wins"]
    return won == 0 or lost == 0


def wilson(wins: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 1.0)
    z, p = 1.959964, wins / games
    d = 1.0 + z * z / games
    centre = (p + z * z / (2 * games)) / d
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))
