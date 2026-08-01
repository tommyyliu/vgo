#!/usr/bin/env python3
"""Where does the game balance, and does the balance point move as the net learns?

Reads the per-game sidecars a generation run writes (`games.jsonl`, one row per
game) and produces two views:

  1. Black's winrate against komi, pooled over a window of shards, with Wilson
     intervals and the fitted 50% crossing. This is the balance point at one
     moment in training.

  2. That crossing plotted against shard index -- the balance point over
     training time. This is the view that answers whether the game has a stable
     balance point or whether it drifts as the model strengthens, which a single
     curve cannot show.

Pooling is not optional. A shard holds ~85 games; across eight buckets that is
~10 games each, and the binomial interval on 5/10 spans roughly [19%, 81%].
A sixteen-shard window gives ~170 per bucket and an interval near +/-7%.

A caveat the plots label rather than hide: winrate-against-komi mixes the game's
balance with the model's own asymmetry between the two seats. If the net simply
plays Black better, the crossing shifts even for a perfectly fair game. The
margin curve is the control -- it should cross zero at the same komi the winrate
crosses 50%, and a gap between the two crossings measures the seat asymmetry.

Usage:
    scripts/komi-balance.py artifacts/ddrnet-komi [--window 16] [--output plot.png]
"""

import argparse
import bisect
import json
import math
import pathlib
import sys


def load_shards(root: pathlib.Path) -> list[tuple[str, list[dict]]]:
    """Every shard under `root` that has a sidecar, ordered by shard name."""
    shards = []
    for sidecar in sorted(root.glob("**/games.jsonl")):
        rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
        if rows:
            shards.append((sidecar.parent.name, rows))
    return shards


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval.

    Wilson rather than the normal approximation because the buckets at the ends
    of the komi range run to extreme rates on few games, exactly where the
    normal interval misbehaves and can leave [0, 1].
    """
    if total == 0:
        return (0.5, 0.0, 1.0)
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (rate, max(0.0, centre - spread), min(1.0, centre + spread))


def bucket(rows: list[dict], low: float, high: float, count: int) -> list[dict]:
    """Black's results over fixed komi buckets.

    Edges come from the range rather than the data so buckets line up across
    windows and stay comparable between runs.
    """
    width = (high - low) / count
    out = []
    for index in range(count):
        start = low + width * index
        end = high if index + 1 == count else start + width
        inside = [
            row for row in rows
            if row["komi"] >= start and (row["komi"] < end or (index + 1 == count and row["komi"] <= end))
        ]
        black = sum(1 for row in inside if row["black_utility"] > 0)
        margins = sorted(
            row["margin"] if row["black_utility"] >= 0 else -row["margin"] for row in inside
        )
        median = 0.0
        if margins:
            middle = len(margins) // 2
            median = (
                margins[middle]
                if len(margins) % 2
                else (margins[middle - 1] + margins[middle]) / 2
            )
        rate, lower, upper = wilson(black, len(inside))
        out.append({
            "centre": (start + end) / 2,
            "low": start, "high": end,
            "games": len(inside), "black_wins": black,
            "rate": rate, "ci": (lower, upper),
            "margin_median": median,
        })
    return out


def crossing(buckets: list[dict], level: float, key: str) -> float | None:
    """Where the curve crosses `level`, by linear interpolation between buckets.

    A logistic fit would be the better estimator, but it needs scipy and it
    hides how thin the data is. Interpolation between adjacent buckets is honest
    about resting on two points, and it returns None when the curve never
    crosses -- which is itself the finding, meaning the range does not bracket
    the balance point.
    """
    usable = [b for b in buckets if b["games"] > 0]
    for left, right in zip(usable, usable[1:]):
        a, b = left[key], right[key]
        if (a - level) * (b - level) <= 0 and a != b:
            span = (level - a) / (b - a)
            return left["centre"] + span * (right["centre"] - left["centre"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=pathlib.Path, help="run directory holding replay shards")
    parser.add_argument("--window", type=int, default=16, help="shards pooled per point (default 16)")
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--komi-low", type=float, default=-0.1)
    parser.add_argument("--komi-high", type=float, default=0.2)
    parser.add_argument("--output", type=pathlib.Path, help="write a PNG here instead of text only")
    arguments = parser.parse_args()

    shards = load_shards(arguments.root)
    if not shards:
        print(f"no games.jsonl under {arguments.root}", file=sys.stderr)
        print("Only runs generated after the sidecar landed carry one; it cannot be backfilled.", file=sys.stderr)
        return 1

    every = [row for _, rows in shards for row in rows]
    print(f"{len(shards)} shards, {len(every)} games\n")

    # Panel 1: the pooled curve over the most recent window.
    recent = [row for _, rows in shards[-arguments.window:] for row in rows]
    latest = bucket(recent, arguments.komi_low, arguments.komi_high, arguments.buckets)
    print(f"Latest {min(arguments.window, len(shards))} shards, {len(recent)} games")
    print(f"  {'komi':>16}  {'games':>5}  {'black':>13}  {'95% CI':>16}  {'margin':>8}")
    for entry in latest:
        low, high = entry["ci"]
        print(
            f"  [{entry['low']:+.4f},{entry['high']:+.4f}) {entry['games']:>5}  "
            f"{entry['black_wins']:>3}/{entry['games']:<3} {100*entry['rate']:>4.0f}%  "
            f"[{100*low:>4.0f}%,{100*high:>4.0f}%]  {entry['margin_median']:>+8.3f}"
        )
    by_rate = crossing(latest, 0.5, "rate")
    by_margin = crossing(latest, 0.0, "margin_median")
    print(f"\n  balance point by winrate: {by_rate:+.4f}" if by_rate is not None
          else "\n  balance point by winrate: not bracketed by this komi range")
    print(f"  balance point by margin:  {by_margin:+.4f}" if by_margin is not None
          else "  balance point by margin:  not bracketed by this komi range")
    if by_rate is not None and by_margin is not None:
        print(f"  seat asymmetry (gap):     {by_rate - by_margin:+.4f}")

    # Panel 2: the crossing over training time, one point per window position.
    history = []
    for end in range(arguments.window, len(shards) + 1):
        pooled = [row for _, rows in shards[end - arguments.window:end] for row in rows]
        point = crossing(bucket(pooled, arguments.komi_low, arguments.komi_high, arguments.buckets), 0.5, "rate")
        if point is not None:
            history.append((end - 1, point))
    if history:
        print(f"\nBalance point over training ({len(history)} windows)")
        for index, point in history[:: max(1, len(history) // 12)]:
            print(f"  shard {index:>3}  {point:+.4f}")
    elif len(shards) < arguments.window:
        print(f"\nBalance point over training: needs {arguments.window} shards, have {len(shards)}")

    if arguments.output:
        try:
            plot(latest, history, arguments)
        except ModuleNotFoundError:
            # matplotlib ships in the `plots` extra, not the base dependencies.
            # The numbers above are the result; the plot is a convenience, so a
            # missing plotting stack must not fail the analysis.
            print(f"\nmatplotlib unavailable in {sys.executable}; skipped {arguments.output}",
                  file=sys.stderr)
            print("Install it with: uv sync --extra plots --extra tensorrt", file=sys.stderr)
            return 0
        print(f"\nwrote {arguments.output}")
    return 0


def plot(latest, history, arguments) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    drawn = [entry for entry in latest if entry["games"] > 0]
    centres = [entry["centre"] for entry in drawn]
    rates = [entry["rate"] for entry in drawn]
    lower = [entry["rate"] - entry["ci"][0] for entry in drawn]
    upper = [entry["ci"][1] - entry["rate"] for entry in drawn]
    left.errorbar(centres, rates, yerr=[lower, upper], marker="o", capsize=3, label="Black winrate")
    left.axhline(0.5, linestyle="--", linewidth=1, color="grey")
    point = crossing(latest, 0.5, "rate")
    if point is not None:
        left.axvline(point, linestyle=":", color="crimson", label=f"balance {point:+.3f}")
    left.set_xlabel("komi")
    left.set_ylabel("Black winrate")
    left.set_ylim(0, 1)
    left.set_title(f"Winrate vs komi (last {arguments.window} shards)")
    left.legend()
    left.grid(alpha=0.3)

    # The margin curve shares the axis: it is the control on seat asymmetry.
    twin = left.twinx()
    twin.plot(centres, [entry["margin_median"] for entry in drawn], marker="s",
              linewidth=1, alpha=0.55, color="seagreen", label="median margin")
    twin.axhline(0.0, linewidth=0.6, color="seagreen", alpha=0.35)
    twin.set_ylabel("median margin (toward Black)", color="seagreen")

    if history:
        right.plot([index for index, _ in history], [point for _, point in history], marker="o")
        right.axhline(0.0, linestyle="--", linewidth=1, color="grey")
        right.set_xlabel("shard index (window end)")
        right.set_ylabel("balance point (komi)")
        right.set_title("Balance point over training")
        right.grid(alpha=0.3)
    else:
        right.text(0.5, 0.5, f"needs {arguments.window} shards", ha="center", va="center")
        right.set_axis_off()

    figure.tight_layout()
    figure.savefig(arguments.output, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
