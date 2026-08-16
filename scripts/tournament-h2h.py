#!/usr/bin/env python3
"""Read one `vgo-tournament` field: head-to-head grid, Elo gaps, and two checks.

`rate-tournament.py` fits the same games onto the shared dense-curve scale.
This is for reading a single hand-picked field on its own terms: who beat whom,
by how much, and whether the spread is real. Three things it does that the
shared fit does not.

**Gaps, not ratings.** Each row is the gap to one reference player with *that
gap's* standard error. Two ratings from one fit are correlated, so the error on
their difference is c'Cc; adding two marginal errors overstates it and can hide
a real difference. `rate-tournament.py` reports each rating against the field
mean, which is the right choice for a curve and the wrong one for "is A better
than B".

**Is any of it real.** A likelihood-ratio test against "every player is equal
strength". A ten-player field can show a 137 Elo spread and still not reject
that null, which is worth knowing before reading the ordering.

**Is it measuring skill at all.** If komi is set where colour decides the game,
colour-swapped pairs return N/2 regardless of strength and every rating is
compressed toward zero. The spread of pairing scores separates that case from a
genuinely flat field: it is ~0 when colour decides, and ~sqrt(games/4) when the
games are near-even coin flips.

    scripts/tournament-h2h.py artifacts/run-finals/records-fixed-rules.jsonl
    scripts/tournament-h2h.py records.jsonl --anchor ddrnet-attn-komi:61
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "training"))
sys.path.insert(0, str(_ROOT / "scripts"))
from bt_stats import components, covariance_of, difference  # noqa: E402
from vgo_training.bradley_terry import ELO_SCALE, fit_ratings  # noqa: E402

_UPDATE = re.compile(r"update-(\d+)")
# Long artifact directory names do not fit a grid header. Shorten only the ones
# that recur; anything else keeps its directory name so it is never ambiguous.
_SHORT = {
    "ddrnet-attn-komi": "komi",
    "ddrnet-fresh-attn": "attn",
    "ddrnet-fresh-muon": "muon",
    "shard-sweep-15000": "sw15k",
    "shard-sweep-10000": "sw10k",
    "shard-sweep-5000": "sw5k",
}


def records(text: str) -> list[dict]:
    """Every top-level JSON object in `text`, by brace depth.

    The tournament pretty-prints each record across several lines and flushes
    it as its pairing finishes, so this parses a file that is still being
    appended to -- which is what makes it usable mid-run.
    """
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


def identify(path: str) -> tuple[str, int]:
    if path == "naive":
        return "naive", -1
    parts = Path(path).parts
    match = _UPDATE.search(path)
    if "updates" not in parts or match is None:
        raise SystemExit(f"cannot place checkpoint path: {path}")
    return parts[parts.index("updates") - 1], int(match.group(1))


def label(player: tuple[str, int]) -> str:
    run, version = player
    if run == "naive":
        return "naive"
    return f"{_SHORT.get(run, run)} {version}"


def chi2_survival(x: float, degrees: int) -> float:
    """Wilson-Hilferty tail, accurate enough to read a p-value off."""
    from statistics import NormalDist
    cube = (x / degrees) ** (1 / 3)
    mean = 1 - 2 / (9 * degrees)
    sigma = math.sqrt(2 / (9 * degrees))
    return 1 - NormalDist().cdf((cube - mean) / sigma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--anchor", default=None,
                        help="reference player as run:update; default is the "
                             "strongest in the field")
    parser.add_argument("--prior-games", type=float, default=0.25)
    arguments = parser.parse_args()

    found = records(arguments.records.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit(f"no vgo.arena.v1 records in {arguments.records}")

    players: dict[tuple[str, int], int] = {}
    names: dict[int, tuple[str, int]] = {}

    def key(path: str) -> int:
        player = identify(path)
        if player not in players:
            players[player] = len(players)
            names[players[player]] = player
        return players[player]

    matches, undecided = [], 0
    scores: dict[tuple[int, int], float] = {}
    played: dict[tuple[int, int], int] = {}
    for record in found:
        if int(record["completed"]) == 0:
            undecided += 1
            continue
        a, b = key(record["candidate_model"]), key(record["opponent_model"])
        wins, losses = int(record["candidate_wins"]), int(record["candidate_losses"])
        draws = int(record["draws"])
        matches.append({"a": a, "b": b, "a_wins": wins,
                        "b_wins": losses, "draws": draws})
        total = wins + losses + draws
        scores[(a, b)] = wins + 0.5 * draws
        scores[(b, a)] = total - (wins + 0.5 * draws)
        played[(a, b)] = played[(b, a)] = total

    games = sum(played[p] for p in scores) // 2
    print(f"{len(matches)} pairings, {games} decided games"
          + (f" ({undecided} undecided)" if undecided else ""))

    groups = components(matches)
    for stranded in groups[1:]:
        print("  stranded: " + ", ".join(sorted(label(names[p]) for p in stranded)))
    main_group = groups[0]
    kept = [m for m in matches if m["a"] in main_group and m["b"] in main_group]

    if arguments.anchor:
        run, _, version = arguments.anchor.partition(":")
        wanted = (run, int(version))
        if wanted not in players:
            raise SystemExit(f"--anchor {arguments.anchor} is not in this field: "
                             + ", ".join(sorted(f"{r}:{v}" for r, v in players)))
        anchor = players[wanted]
    else:
        # Provisional fit to find the strongest, then refit anchored on it.
        provisional = fit_ratings(kept, anchor=min(main_group),
                                  prior_games=arguments.prior_games)
        anchor = max(provisional, key=provisional.get)
    if anchor not in main_group:
        raise SystemExit(f"reference {label(names[anchor])} is not connected")

    ratings = fit_ratings(kept, anchor=anchor, prior_games=arguments.prior_games)
    covariance, index = covariance_of(kept, ratings, anchor, arguments.prior_games)
    order = sorted(ratings, key=lambda p: -ratings[p])

    width = max(len(label(names[p])) for p in order)
    print()
    print("Head-to-head. Each cell is the row player's score out of the games "
          "that pair played.")
    print()
    print(" " * (width + 2) + "".join(f"{label(names[p]):>10s}" for p in order)
          + f"  {'total':>12s}")
    for row in order:
        cells = ""
        won = total = 0.0
        for column in order:
            if row == column:
                cells += f"{'--':>10s}"
                continue
            if (row, column) not in scores:
                cells += f"{'.':>10s}"
                continue
            cells += f"{scores[(row, column)]:>10.1f}"
            won += scores[(row, column)]
            total += played[(row, column)]
        share = f"{won:.0f}/{total:.0f} {won / total * 100:4.0f}%" if total else ""
        print(f"  {label(names[row]):{width}s}" + cells + f"  {share:>12s}")

    print()
    print(f"Elo relative to {label(names[anchor])}; +/- is the error on the gap")
    print()
    for p in order:
        if p == anchor:
            print(f"  {label(names[p]):{width}s}        0  (reference)")
            continue
        value, error = difference(p, anchor, ratings, covariance, index)
        flag = "  *" if abs(value) > 1.96 * error else ""
        print(f"  {label(names[p]):{width}s} {value:>+8.0f} +/- {error:3.0f}{flag}")
    # One flag per player, all against the same reference, all uncorrected. In a
    # ten-player field that is nine comparisons, so a lone star is roughly what
    # chance produces -- the likelihood-ratio test below is what says whether
    # the field has any real spread, and it should be read first.
    print("  (* = differs from the reference at 95%, uncorrected for the "
          f"{len(order) - 1} comparisons)")

    # Is there any real strength spread at all? Fitted with no prior so the
    # comparison is against the data alone.
    free = fit_ratings(kept, anchor=anchor, prior_games=0.0)
    theta = {p: v / ELO_SCALE for p, v in free.items()}
    fitted = null = 0.0
    for m in kept:
        wins, losses = m["a_wins"], m["b_wins"]
        p = 1 / (1 + math.exp(-(theta[m["a"]] - theta[m["b"]])))
        p = min(max(p, 1e-12), 1 - 1e-12)
        fitted += wins * math.log(p) + losses * math.log(1 - p)
        null += (wins + losses) * math.log(0.5)
    ratio, degrees = 2 * (fitted - null), len(free) - 1
    print()
    print(f"spread    : {max(ratings.values()) - min(ratings.values()):.0f} Elo "
          f"across {len(ratings)} players")
    print(f"all equal?: likelihood ratio {ratio:.1f} on {degrees} df, "
          f"p = {chi2_survival(ratio, degrees):.3f}")

    # Colour saturation check. Pairings are colour-swapped, so if colour decided
    # every game each would come back exactly even and the spread would be zero.
    sizes = {played[(m['a'], m['b'])] for m in kept}
    if len(sizes) == 1:
        per = sizes.pop()
        observed = statistics.pstdev([scores[(m["a"], m["b"])] for m in kept])
        print(f"pairings  : sd {observed:.2f} of {per} games "
              f"(fair coin {math.sqrt(per * 0.25):.2f}, colour-decides 0.00)")


if __name__ == "__main__":
    main()
