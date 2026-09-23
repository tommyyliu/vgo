#!/usr/bin/env python3
"""Fresh-process RAM scaling, bounded address space, exact paired search gate."""
import argparse
import csv
import json
import resource
import subprocess
import sys


def limit_probe():
    # Stop allocation inside the diagnostic instead of risking an unbounded
    # machine-wide experiment. Virtual address-space limit, not an RSS target.
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--large-only", action="store_true",
                        help="Extend the initial sweep to 16 and 32 actors")
    args = parser.parse_args()
    assert args.samples > 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["actors", "simulations", "sample", "variant", "nodes",
                     "tracked_bytes", "resident_kib", "peak_kib", "build_seconds"])
    cases = [(a, s) for s in [350, 700, 1400] for a in [1, 2, 4, 8]]
    cases += [(a, 2800) for a in [1, 2, 4]]
    if args.large_only:
        cases = [(a, s) for s in [350, 700, 1400] for a in [16, 32]]
    for actors, simulations in cases:
        expected = None
        for sample in range(args.samples):
            variants = ["before", "after"] if sample % 2 == 0 else ["after", "before"]
            for variant in variants:
                output = subprocess.check_output(
                    [getattr(args, variant), str(actors), str(simulations)],
                    text=True, preexec_fn=limit_probe, timeout=180,
                )
                result = json.loads(output)
                identity = (result["nodes"], result["signatures"])
                if expected is None:
                    expected = identity
                assert identity == expected, "full search signatures changed"
                writer.writerow([actors, simulations, sample, variant,
                                 result["nodes"], result["tracked_bytes"],
                                 result["resident_kib"], result["peak_kib"],
                                 result["build_seconds"]])
                sys.stdout.flush()
        print(f"verified actors={actors} simulations={simulations}", file=sys.stderr)


if __name__ == "__main__":
    main()
