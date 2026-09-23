#!/usr/bin/env python3
"""Time the large CPU paths, keep the history, and flag regressions.

Builds and runs `vgo-bench` (crates/vgo-selfplay/src/bench.rs), which times
each path single-threaded on the fixed positions in benchmarks/positions.txt,
then compares against the most recent record from this machine and appends the
new one to benchmarks/history.jsonl. The history is committed, so the cost of
every path can be read across the project's life.

Only records from the same host are compared: the numbers are wall-clock and
mean nothing across machines. A change counts as a regression when it is slower
by more than --threshold *and* by more than the combined run-to-run spread of
the two measurements, so noise on a cheap path does not raise an alarm.

    scripts/bench.py              # run, compare, record
    scripts/bench.py --no-record  # run and compare only, e.g. on a dirty tree
    scripts/bench.py --check      # exit 1 on any regression
    scripts/bench.py --show       # print the history for this host, no run

A busy machine inflates every number, so this refuses to run while the load
average is above --maximum-load (a generator running is the usual cause).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "benchmarks" / "history.jsonl"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def history() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def compare(previous: dict, current: dict, threshold: float) -> list[str]:
    """Print a before/after table; return the names that regressed."""
    regressed = []
    print(f"\n  against {previous['commit'][:9]} ({previous['timestamp'][:16]})")
    print(f"  {'path':<22} {'before':>11} {'after':>11} {'change':>8}")
    for name, now in current["results"].items():
        before = previous["results"].get(name)
        if before is None:
            print(f"  {name:<22} {'-':>11} {now['us']:>9.1f}us {'new':>8}")
            continue
        change = now["us"] / before["us"] - 1.0
        noise = before.get("spread", 0.0) + now.get("spread", 0.0)
        flag = ""
        if change > threshold and change > noise:
            flag = "  REGRESSED"
            regressed.append(name)
        elif change < -threshold and -change > noise:
            flag = "  faster"
        print(
            f"  {name:<22} {before['us']:>9.1f}us {now['us']:>9.1f}us "
            f"{100 * change:>+7.1f}%{flag}"
        )
    return regressed


def show(host: str) -> None:
    records = [r for r in history() if r["host"] == host]
    if not records:
        print(f"no history for {host}")
        return
    names = list(records[-1]["results"])
    print(f"{'commit':<10} {'date':<11}" + "".join(f" {n[-14:]:>14}" for n in names))
    for record in records:
        cells = "".join(
            f" {record['results'][n]['us']:>14.1f}" if n in record["results"] else f" {'-':>14}"
            for n in names
        )
        print(f"{record['commit'][:9]:<10} {record['timestamp'][:10]:<11}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--maximum-load", type=float, default=2.0)
    parser.add_argument("--note", default="", help="free text stored with the record")
    arguments = parser.parse_args()
    host = socket.gethostname()

    if arguments.show:
        show(host)
        return

    load = os.getloadavg()[0]
    if load > arguments.maximum_load:
        sys.exit(
            f"load average is {load:.1f}; timings would be inflated. Stop the "
            "generator or pass --maximum-load to override."
        )

    subprocess.run(
        ["cargo", "build", "--release", "-p", "vgo-selfplay", "--bin", "vgo-bench"],
        cwd=ROOT, check=True,
    )
    completed = subprocess.run(
        [str(ROOT / "target/release/vgo-bench"), "--seconds", str(arguments.seconds)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    sys.stderr.write(completed.stderr)
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    record = {
        "schema": "vgo.bench.v1",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "dirty": dirty,
        "host": host,
        "cpu": cpu_model(),
        "load": round(load, 2),
        "note": arguments.note,
        "results": json.loads(completed.stdout),
    }

    earlier = [r for r in history() if r["host"] == host]
    regressed = compare(earlier[-1], record, arguments.threshold) if earlier else []
    if not earlier:
        print("\n  first record for this host; nothing to compare against")

    if arguments.no_record:
        print("\n  not recorded (--no-record)")
    elif dirty:
        print("\n  not recorded: the tree has uncommitted changes, so the record "
              "would name a commit that is not what ran. Commit, or use --no-record.")
    else:
        HISTORY.parent.mkdir(exist_ok=True)
        with HISTORY.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"\n  recorded in {HISTORY.relative_to(ROOT)}")

    if regressed:
        print(f"\n  {len(regressed)} regression(s): {', '.join(regressed)}")
        if arguments.check:
            sys.exit(1)


if __name__ == "__main__":
    main()
