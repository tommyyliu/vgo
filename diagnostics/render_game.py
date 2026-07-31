"""Render a game from a v4 shard as SVG, the way the web client draws it.

Pairs with `vgo-render-board`: this reads the shard and emits positions as
JSON, the Rust binary draws them from the same `Analysis` the engine uses. The
alternative -- reimplementing Voronoi cells, the group partition, and the
settled-region test in Python -- would be a second geometry implementation to
keep in step, which is exactly the drift the v4 format was meant to remove.

    python3 diagnostics/render_game.py <shard.vgo> [--game N] [--frames 6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from vgo_training.dataset import HEADER, _v4_record_dtype  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/vgo-render-board"


def positions(shard: Path, game: int | None, frames: int):
    blob = shard.read_bytes()
    fields = HEADER.unpack_from(blob, 0)
    version, samples, policy_size = fields[1], fields[2], fields[6]
    if version != 4:
        raise SystemExit(f"{shard} is replay version {version}; this reads v4")
    records = np.frombuffer(
        blob, dtype=_v4_record_dtype(policy_size), count=samples, offset=HEADER.size
    )
    if game is None:
        identifiers, counts = np.unique(records["game"], return_counts=True)
        game = int(identifiers[counts.argmax()])
    rows = np.flatnonzero(records["game"] == game)
    rows = rows[np.argsort(records["ply"][rows])]
    if not len(rows):
        raise SystemExit(f"game {game} is not in {shard}")
    picks = rows[np.linspace(0, len(rows) - 1, min(frames, len(rows))).astype(int)]
    return game, records, picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--game", type=int, default=None)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--settled", action="store_true",
                        help="shade groups whose territory can no longer change")
    parser.add_argument("--stone-ids", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/rasters")
    arguments = parser.parse_args()

    if not BINARY.exists():
        raise SystemExit(
            f"{BINARY} not built; run:\n"
            "  cargo build --release -p vgo-selfplay --bin vgo-render-board"
        )

    game, records, picks = positions(arguments.shard, arguments.game, arguments.frames)
    lines = []
    for row in picks:
        record = records[row]
        count = int(record["stone_count"])
        stones = record["stones"][:count]
        lines.append(json.dumps({
            "radius": float(record["radius"]),
            "to_move": "B" if int(record["to_move"]) == 0 else "W",
            "stones": [
                {"x": float(s["x"]), "y": float(s["y"]),
                 "color": "B" if int(s["color"]) == 0 else "W"}
                for s in stones
            ],
        }))

    command = [
        str(BINARY),
        "--output", str(arguments.output),
        "--prefix", f"game{game}",
        "--size", str(arguments.size),
    ]
    if arguments.settled:
        command.append("--settled")
    if arguments.stone_ids:
        command.append("--stone-ids")

    result = subprocess.run(
        command, input="\n".join(lines), text=True, capture_output=True, check=False
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)

    plies = [int(records[row]["ply"]) for row in picks]
    counts = [int(records[row]["stone_count"]) for row in picks]
    print(f"  game {game}: plies {plies}")
    print(f"  stones     : {counts}")
    print("  blue = Black, orange = White (absolute, unlike the raster's "
          "side-to-move colours)")
    print("  white edges = boundary between groups; grey = within a group")


if __name__ == "__main__":
    main()
