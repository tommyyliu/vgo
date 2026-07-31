"""Render a game the way the web UI draws it, for reading a position by eye.

The semantic channels a model reads are legible one at a time but hard to hold
as a game. This draws the same board `rasterize_rgb_into` produces -- identical
palette, identical draw order (territory, then legal tint, then stones) -- at a
resolution meant for looking at rather than for convolving over, and lays a
game out as a filmstrip.

Reads stone positions straight from the v4 record, so nothing here depends on
the training rasterizer and the two can be compared against each other.

    python3 diagnostics/render_board.py <shard.vgo> [--game N] [--frames 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from vgo_training.dataset import (  # noqa: E402
    HEADER,
    _v4_record_dtype,
)

# crates/vgo-raster/src/lib.rs. Relative to the side to move, as every channel
# in that crate is: CURRENT_* is whoever is about to play.
CURRENT_STONE = np.array([90.0, 162.0, 236.0])
CURRENT_REGION = np.array([34.0, 64.0, 92.0])
OPPONENT_STONE = np.array([240.0, 151.0, 90.0])
OPPONENT_REGION = np.array([104.0, 57.0, 26.0])
BOARD_BACKGROUND = np.array([14.0, 17.0, 22.0])
LEGAL_TINT = np.array([205.0, 214.0, 232.0])
LEGAL_TINT_ALPHA = 42.0 / 255.0


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def render(record, size: int, supersample: int = 2) -> np.ndarray:
    """One position, drawn as the UI draws it.

    Supersampled because stone edges and the legal-mask boundary are the two
    things worth seeing clearly, and both are curves.
    """
    extent = size * supersample
    radius = float(record["radius"])
    count = int(record["stone_count"])
    stones = record["stones"][:count]

    axis = (np.arange(extent) + 0.5) / extent
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")

    to_move = int(record["to_move"])
    is_current = stones["color"] == to_move

    def nearest_square(subset) -> np.ndarray:
        if not len(subset):
            return np.full((extent, extent), np.inf)
        dx = grid_x[None, :, :] - subset["x"][:, None, None]
        dy = grid_y[None, :, :] - subset["y"][:, None, None]
        return (dx * dx + dy * dy).min(axis=0)

    current_square = nearest_square(stones[is_current])
    opponent_square = nearest_square(stones[~is_current])

    color = np.broadcast_to(BOARD_BACKGROUND, (extent, extent, 3)).astype(np.float64).copy()

    # Territory: whichever side is strictly nearer owns the pixel; ties and
    # empty boards leave the background showing.
    current_owns = current_square < opponent_square
    opponent_owns = opponent_square < current_square
    color[current_owns] = CURRENT_REGION
    color[opponent_owns] = OPPONENT_REGION

    # The legal overlay: a placement is legal when it clears the board edge and
    # sits at least two radii from every stone.
    board_clearance = np.minimum.reduce([
        grid_x - radius, 1.0 - radius - grid_x,
        grid_y - radius, 1.0 - radius - grid_y,
    ])
    nearest = np.minimum(current_square, opponent_square)
    stone_clearance = np.where(np.isfinite(nearest), np.sqrt(nearest) - 2.0 * radius, np.inf)
    legal = np.minimum(board_clearance, stone_clearance) > 0.0
    color[legal] = color[legal] * (1.0 - LEGAL_TINT_ALPHA) + LEGAL_TINT * LEGAL_TINT_ALPHA

    color[current_square <= radius * radius] = CURRENT_STONE
    color[opponent_square <= radius * radius] = OPPONENT_STONE

    # Box-filter back down to the requested size.
    reduced = color.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    return np.clip(reduced, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--game", type=int, default=None,
                        help="game id; default is the longest in the shard")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "rasters")
    arguments = parser.parse_args()

    blob = arguments.shard.read_bytes()
    fields = HEADER.unpack_from(blob, 0)
    version, samples, _channels, _height, _width, policy_size = (
        fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
    )
    if version != 4:
        raise SystemExit(f"{arguments.shard} is replay version {version}; this reads v4")
    records = np.frombuffer(
        blob, dtype=_v4_record_dtype(policy_size), count=samples, offset=HEADER.size
    )

    games = records["game"]
    plies = records["ply"]
    if arguments.game is None:
        identifiers, counts = np.unique(games, return_counts=True)
        game = int(identifiers[counts.argmax()])
    else:
        game = arguments.game
    rows = np.flatnonzero(games == game)
    rows = rows[np.argsort(plies[rows])]
    if not len(rows):
        raise SystemExit(f"game {game} is not in this shard")
    picks = rows[np.linspace(0, len(rows) - 1, min(arguments.frames, len(rows))).astype(int)]

    size, pad = arguments.size, 10
    columns = min(3, len(picks))
    kept = -(-len(picks) // columns)
    sheet = np.zeros(
        (kept * size + (kept + 1) * pad, columns * size + (columns + 1) * pad, 3),
        np.uint8,
    ) + 10
    for index, row in enumerate(picks):
        tile = render(records[row], size)
        y = pad + (index // columns) * (size + pad)
        x = pad + (index % columns) * (size + pad)
        sheet[y:y + size, x:x + size] = tile

    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / f"board-game{game}.png"
    write_png(destination, sheet)
    stones = [int(records[row]["stone_count"]) for row in picks]
    print(f"wrote {destination}")
    print(f"  game {game}: plies {[int(plies[row]) for row in picks]}")
    print(f"  stones on board: {stones}")
    print(f"  blue = side to move, orange = opponent, pale tint = legal placements")


if __name__ == "__main__":
    main()
