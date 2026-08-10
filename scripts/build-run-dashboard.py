#!/usr/bin/env python3
"""Render the Adam-vs-Muon run dashboard to a self-contained HTML page.

Reads the cached validation curves and head-to-head results, plus live
tournament progress from the journal, and writes one file with the data inlined
-- the Artifact CSP blocks every external host, so nothing may be fetched at
view time.

Re-run it to refresh the page; publishing the same path redeploys in place.

    scripts/build-run-dashboard.py curves.json h2h.json --output dashboard.html
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "run-dashboard.template.html"


def tournament_status() -> dict:
    """Live progress from the systemd journal, or an empty dict if absent."""
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "vgo-joint", "--no-pager"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    marks = [(int(a), int(b)) for a, b in
             re.findall(r"(\d+)/528 games, (\d+)s", out)]
    if not marks:
        return {}
    done, elapsed = marks[-1]
    # Rate from the fifth game on: nothing can finish before the first game
    # completes, so including the startup window understates it badly.
    base = next((m for m in marks if m[0] >= 5), marks[0])
    span, games = elapsed - base[1], done - base[0]
    rate = games / span if span > 0 else 0.0
    return {
        "done": done, "total": 528, "elapsed": elapsed,
        "rate": rate,
        "remaining_s": int((528 - done) / rate) if rate > 0 else None,
        "marks": marks,
    }


def memory() -> dict:
    try:
        rss = subprocess.run(["ps", "-o", "rss=", "-C", "vgo-tournament"],
                             capture_output=True, text=True).stdout.split()
        free = Path("/proc/meminfo").read_text(encoding="utf-8")
        available = int(re.search(r"MemAvailable:\s+(\d+)", free).group(1))
        return {"rss_gb": int(rss[0]) / 1048576 if rss else None,
                "available_gb": available / 1048576}
    except (OSError, ValueError, AttributeError, IndexError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curves", type=Path)
    parser.add_argument("h2h", type=Path)
    parser.add_argument("--elo", type=Path, default=None,
                        help="elo.json from h2h-elo.py")
    parser.add_argument("--ratings", type=Path, default=None,
                        help="ratings.json from rate-tournament.py, once it exists")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    payload = {
        "curves": json.loads(arguments.curves.read_text(encoding="utf-8")),
        "h2h": json.loads(arguments.h2h.read_text(encoding="utf-8")),
        "elo": (json.loads(arguments.elo.read_text(encoding="utf-8"))
                if arguments.elo and arguments.elo.exists() else None),
        "tournament": tournament_status(),
        "memory": memory(),
        "ratings": (json.loads(arguments.ratings.read_text(encoding="utf-8"))
                    if arguments.ratings and arguments.ratings.exists() else None),
        "generated": datetime.now().strftime("%H:%M"),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null/*DATA*/", json.dumps(payload))
    arguments.output.write_text(html, encoding="utf-8")
    t = payload["tournament"]
    print(f"-> {arguments.output}"
          + (f"  ({t['done']}/528 games, {t['rate']:.3f} g/s)" if t else ""))


if __name__ == "__main__":
    main()
