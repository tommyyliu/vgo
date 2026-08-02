#!/usr/bin/env python3
"""Where does komi actually balance the game?

Fits P(Black wins) = sigmoid(a + b*komi) over played-out games and reports the
crossing, which is the komi that makes the game even.

Only games that were *played* count. A resigned game's winner was assigned by
the resignation rule, and under that rule the mover always concedes -- so the
winner is fixed by ply parity and carries no information about balance. Pooling
those in produced an apparent 86% Black when the played-out subset said 62%.

Two readings are printed because they can disagree, and the disagreement is
informative:

  * the winrate crossing -- the komi at which Black wins half the time
  * the median signed area margin -- how far ahead Black is on the board

A model that plays one seat better than the other shifts the first without
shifting the second, so a gap between them is the model's seat asymmetry rather
than the game's balance.

    scripts/komi-balance-fit.py artifacts/ddrnet-noresign
"""

import argparse
import glob
import json
import math
import pathlib
import statistics
import sys


def load(root: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / "replay" / "shard-*" / "games.jsonl"))):
        rows += [json.loads(line) for line in open(path) if line.strip()]
    return rows


def logistic(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Newton-Raphson fit of a two-parameter logistic.

    Hand-rolled rather than via scipy, which is not a dependency of this repo.
    The ridge on the Hessian diagonal keeps it from blowing up when the data are
    separable -- which happens whenever every game in the sample went one way.
    """
    a = b = 0.0
    for _ in range(200):
        gradient = [0.0, 0.0]
        hessian = [[1e-9, 0.0], [0.0, 1e-9]]
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(a + b * x)))
            weight = p * (1.0 - p)
            gradient[0] += y - p
            gradient[1] += (y - p) * x
            hessian[0][0] += weight
            hessian[0][1] += weight * x
            hessian[1][0] += weight * x
            hessian[1][1] += weight * x * x
        determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0]
        if abs(determinant) < 1e-12:
            return None
        step_a = (hessian[1][1] * gradient[0] - hessian[0][1] * gradient[1]) / determinant
        step_b = (-hessian[1][0] * gradient[0] + hessian[0][0] * gradient[1]) / determinant
        a += step_a
        b += step_b
        if abs(step_a) + abs(step_b) < 1e-10:
            break
    return a, b


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return (0.0, 1.0)
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def bootstrap(xs: list[float], ys: list[float], draws: int = 400) -> tuple[float, float] | None:
    """Percentile interval on the crossing, by resampling games.

    The crossing is a ratio of two fitted parameters, so its uncertainty is not
    something the fit reports directly -- and on a few hundred games it is wide
    enough that quoting a point estimate alone would overstate what is known.
    """
    import random

    rng = random.Random(0)
    crossings = []
    count = len(xs)
    for _ in range(draws):
        picks = [rng.randrange(count) for _ in range(count)]
        fit = logistic([xs[i] for i in picks], [ys[i] for i in picks])
        if fit and fit[1] < -1e-6:
            crossings.append(-fit[0] / fit[1])
    if len(crossings) < draws // 4:
        return None
    crossings.sort()
    return (crossings[len(crossings) // 40], crossings[-len(crossings) // 40])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--buckets", type=int, default=6)
    parser.add_argument("--komi-low", type=float, default=-0.1)
    parser.add_argument("--komi-high", type=float, default=0.2)
    arguments = parser.parse_args()

    rows = load(arguments.root)
    if not rows:
        print(f"no games.jsonl under {arguments.root}", file=sys.stderr)
        return 1

    played = [row for row in rows if not row["resigned"]]
    resigned = len(rows) - len(played)
    black_all = sum(1 for row in rows if row["black_utility"] > 0)
    black_played = sum(1 for row in played if row["black_utility"] > 0)

    print(f"{len(rows)} games: {len(played)} played out, {resigned} resigned")
    print(f"  Black over everything:   {100 * black_all / len(rows):.1f}%")
    if not played:
        print("  no played-out games; nothing to fit")
        return 1
    low, high = wilson(black_played, len(played))
    print(
        f"  Black over played out:   {100 * black_played / len(played):.1f}%"
        f"   95% CI [{100 * low:.1f}%, {100 * high:.1f}%]"
    )

    width = (arguments.komi_high - arguments.komi_low) / arguments.buckets
    print(f"\n  {'komi':>18} {'n':>5} {'Black':>7}  {'margin':>8}")
    for index in range(arguments.buckets):
        start = arguments.komi_low + width * index
        end = start + width
        inside = [
            row for row in played
            if start <= row["komi"] < end
            or (index + 1 == arguments.buckets and abs(row["komi"] - end) < 1e-9)
        ]
        if not inside:
            continue
        black = sum(1 for row in inside if row["black_utility"] > 0)
        margins = [
            row["margin"] if row["black_utility"] > 0 else -row["margin"]
            for row in inside if row["margin"] > 0
        ]
        median = statistics.median(margins) if margins else float("nan")
        print(
            f"  [{start:+.3f},{end:+.3f}) {len(inside):>5} "
            f"{100 * black / len(inside):>6.1f}%  {median:>+8.3f}"
        )

    xs = [row["komi"] for row in played]
    ys = [1.0 if row["black_utility"] > 0 else 0.0 for row in played]
    fit = logistic(xs, ys)
    print()
    if not fit or fit[1] >= -1e-6:
        print("  komi does not move the outcome in this sample; no crossing")
        return 0
    crossing = -fit[0] / fit[1]
    print(f"  P(Black) = sigmoid({fit[0]:+.3f} {fit[1]:+.3f} * komi)")
    interval = bootstrap(xs, ys)
    if interval:
        print(f"  balance point: {crossing:+.4f}   95% CI [{interval[0]:+.4f}, {interval[1]:+.4f}]")
    else:
        print(f"  balance point: {crossing:+.4f}   (interval unavailable)")
    inside = arguments.komi_low <= crossing <= arguments.komi_high
    position = (crossing - arguments.komi_low) / (arguments.komi_high - arguments.komi_low)
    print(
        f"  range [{arguments.komi_low:+.3f},{arguments.komi_high:+.3f}]: "
        f"{'contains it' if inside else 'DOES NOT contain it'}"
        + (f", at {100 * position:.0f}% of the way up" if inside else "")
    )
    if inside and not 0.3 <= position <= 0.7:
        half = (arguments.komi_high - arguments.komi_low) / 2
        print(
            f"  -> off-centre; centring on it would give "
            f"[{crossing - half:+.3f}, {crossing + half:+.3f}]"
        )

    margins = [
        row["margin"] if row["black_utility"] > 0 else -row["margin"]
        for row in played if row["margin"] > 0
    ]
    if margins:
        print(f"\n  median signed area margin: {statistics.median(margins):+.4f} toward Black")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
