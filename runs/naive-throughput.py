#!/usr/bin/env python3
"""Paired full vgo-canary playout benchmark; takes two already-built binaries.

CSV samples go to stdout; medians go to stderr. Games can reach the ply cap,
so plies/second is the primary metric, not completed-games/second.
"""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--after-arg", action="append", default=[],
                        help="Extra argument for the after variant; use --after-arg=--flag")
    args = parser.parse_args()
    assert args.samples > 0
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "radius", "workers", "sample", "variant", "games", "completed",
        "plies", "simulations", "nodes", "seconds", "plies_per_second",
        "nodes_per_second", "result_sha256",
    ])
    for radius, workers in [(0.1, 1), (1 / 18, 1), (1 / 18, 2), (1 / 38, 1)]:
        timings = {"before": [], "after": []}
        signature = None
        command = [
            "--pairs", "2", "--first", "64", "--second", "64",
            "--max-plies", "64", "--radius", str(radius),
            "--threads", str(workers), "--seed", "41",
        ]
        # One untimed warmup per binary and fixture, then alternating order.
        for sample in range(-1, args.samples):
            variants = ["before", "after"] if sample % 2 == 0 else ["after", "before"]
            for variant in variants:
                result = json.loads(subprocess.check_output(
                    [getattr(args, variant), *command,
                     *(args.after_arg if variant == "after" else [])], text=True,
                ))
                stable = {k: v for k, v in result.items() if k not in {
                    "wall_seconds", "summed_game_seconds", "nodes_per_wall_second",
                }}
                digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
                if signature is None:
                    signature = digest
                assert signature == digest, "game/search aggregate results changed"
                if sample < 0:
                    continue
                seconds = result["wall_seconds"]
                plies = round(result["average_plies"] * result["games"])
                timings[variant].append(seconds)
                writer.writerow([
                    radius, workers, sample, variant, result["games"],
                    result["completed"], plies, result["search_simulations"],
                    result["expanded_nodes"], seconds, plies / seconds,
                    result["expanded_nodes"] / seconds, digest,
                ])
                sys.stdout.flush()
        before = statistics.median(timings["before"])
        after = statistics.median(timings["after"])
        print(
            f"radius={radius:.6f} workers={workers}: "
            f"{plies/before:.2f} -> {plies/after:.2f} plies/s; "
            f"{before/after:.4f}x; "
            f"before range={min(timings['before']):.4f}..{max(timings['before']):.4f}s; "
            f"after range={min(timings['after']):.4f}..{max(timings['after']):.4f}s; "
            f"completed={result['completed']}/{result['games']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
