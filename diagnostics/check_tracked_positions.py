"""Run every position in tracked-positions.jsonl against one model.

Reports where the policy ranked the move actually played, so a tactical
pattern (e.g. a capture the policy missed) can be watched across checkpoints
without re-copying the move list by hand each time.

    python3 diagnostics/check_tracked_positions.py <candidate.onnx>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/examples/inspect_policy"
POSITIONS = ROOT / "diagnostics/tracked-positions.jsonl"

sys.path.insert(0, str(ROOT / "training"))
from vgo_training.pipeline import runtime_environment  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <candidate.onnx>")
    model = Path(sys.argv[1]).resolve()
    if not model.exists():
        raise SystemExit(f"no such model: {model}")
    if not BINARY.exists():
        raise SystemExit(
            f"{BINARY} not built; run: cargo build --release -p vgo-inference "
            "--example inspect_policy"
        )

    environment = runtime_environment()
    entries = [
        json.loads(line)
        for line in POSITIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"model: {model}\n")
    for entry in entries:
        command = [
            str(BINARY),
            "--model", str(model),
            "--radius", str(entry["radius"]),
            "--before-ply", str(entry["before_ply"]),
            "--moves", entry["moves"],
            "--top", "1",
        ]
        completed = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=60
        )
        if completed.returncode != 0:
            print(f"[{entry['name']}] FAILED: {completed.stderr.strip().splitlines()[-3:]}")
            continue
        match = re.search(
            r"probability: ([\d.]+)\s+\(([\d.]+)%\)\s+rank (\d+) of (\d+)",
            completed.stdout,
        )
        value_match = re.search(r"value estimate for \w+: (-?[\d.]+)", completed.stdout)
        if not match:
            print(f"[{entry['name']}] no rank line found (position past its ply?)")
            continue
        probability, percent, rank, total = match.groups()
        value = value_match.group(1) if value_match else "?"
        print(
            f"[{entry['name']}] rank {rank}/{total}  prob {percent}%  "
            f"position value {value}"
        )
        if "note" in entry:
            print(f"    {entry['note']}")


if __name__ == "__main__":
    main()
