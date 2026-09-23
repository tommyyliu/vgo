#!/usr/bin/env python3
"""Paired dense-policy CPU search benchmark, without neural inference.

Uses the RAM probe's synchronized actors and full SearchResult fingerprints.
Reports external process wall time (including teardown) and held-tree build time.
"""
import argparse
import csv
import json
import resource
import statistics
import subprocess
import sys
import time


def limit_probe():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--shard", help="Use v8 replay samples 80, 160, 240 with one actor instead of grid boards")
    args = parser.parse_args()
    assert args.samples > 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["fixture", "actors", "simulations", "sample", "variant",
                     "nodes", "wall_seconds", "build_seconds", "peak_kib"])
    cases = [(n, a, 1400) for n in [20, 80, 160] for a in [1, 8, 32]]
    if args.verify:
        cases = [(n, 2, 64) for n in [20, 80, 160]]
    if args.shard:
        cases = [(n, 1, 64 if args.verify else 1400) for n in [80, 160, 240]]
    for stones, actors, simulations in cases:
        expected = None
        timings = {"baseline": [], "supports": []}
        for sample in range(-1, args.samples):
            order = ["baseline", "supports"] if sample % 2 == 0 else ["supports", "baseline"]
            for variant in order:
                command = [args.binary, str(actors), str(simulations), "--stones", str(stones)]
                if args.shard:
                    command += ["--shard", args.shard, "--sample", str(stones)]
                if variant == "supports":
                    command.append("--supports")
                    if args.verify:
                        command.append("--verify-supports")
                start = time.perf_counter()
                result = json.loads(subprocess.check_output(
                    command, text=True, preexec_fn=limit_probe, timeout=180))
                seconds = time.perf_counter() - start
                identity = (result["nodes"], result["signatures"])
                if expected is None:
                    expected = identity
                assert identity == expected, "full SearchResult fingerprints changed"
                if sample < 0:
                    continue
                timings[variant].append(seconds)
                writer.writerow([stones, actors, simulations, sample, variant,
                                 result["nodes"], seconds, result["build_seconds"],
                                 result["peak_kib"]])
                sys.stdout.flush()
        before, after = [statistics.median(timings[v]) for v in ["baseline", "supports"]]
        print(f"fixture={stones} actors={actors} sims={simulations}: "
              f"{before:.6f}s -> {after:.6f}s, {before/after:.4f}x; "
              f"ranges={min(timings['baseline']):.6f}..{max(timings['baseline']):.6f}/"
              f"{min(timings['supports']):.6f}..{max(timings['supports']):.6f}; fingerprints match",
              file=sys.stderr)


if __name__ == "__main__":
    main()
