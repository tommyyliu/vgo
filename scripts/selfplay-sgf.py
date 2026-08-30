#!/usr/bin/env python3
"""Reconstruct an SGF from a self-play game's position shard.

Generation stores positions, not moves -- a record is the board before the ply
its `to_move` is about to play (see docs/POSITION_SHARDS.md). Only the arena
writes SGF, and only for games it played itself, so a self-play game has to be
recovered by differencing: the stone present at ply k+1 and absent at ply k is
the move played at ply k. Captures remove stones, which is why this looks for
the addition rather than any difference.

The last recorded ply has no successor to difference against, so its move is not
recoverable and the game is written one move short of what was played.

Dialect matches `game_sgf` in crates/vgo-selfplay/src/arena.rs, which is what
reference/src/io/vgo-sgf.js reads.

    scripts/selfplay-sgf.py <game-directory> [output.sgf]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import numpy as np
from vgo_training.dataset import _read_header, _v4_record_dtype, header_size


def moves_from(records):
    """(colour, (x, y) or None) per ply, by differencing consecutive boards."""
    out = []
    for index in range(len(records) - 1):
        here, then = records[index], records[index + 1]
        colour = "B" if int(here["to_move"]) == 0 else "W"
        before = {
            (float(s["x"]), float(s["y"]), int(s["color"]))
            for s in here["stones"][: int(here["stone_count"])]
        }
        after = [
            (float(s["x"]), float(s["y"]), int(s["color"]))
            for s in then["stones"][: int(then["stone_count"])]
        ]
        added = [s for s in after if s not in before]
        out.append((colour, (added[0][0], added[0][1]) if len(added) == 1 else None))
    return out


def main():
    game = Path(sys.argv[1])
    shard = game / "dataset.vgo"
    meta = json.loads((game / "games.jsonl").read_text().splitlines()[0])
    magic, version, samples, channels, height, width, policy, cap, stones = _read_header(shard)
    records = np.memmap(
        shard,
        dtype=_v4_record_dtype(policy, version, cap, stones),
        mode="r",
        offset=header_size(version),
        shape=(samples,),
    )
    text = [f'(;FF[4]GM[VGO]SZ[1]RA[{meta["radius"]}]KM[{meta["komi"]}]PL[B]']
    for colour, point in moves_from(records):
        text.append(f";{colour}[{point[0]},{point[1]}]" if point else f";{colour}[]")
    text.append(")")
    sgf = "".join(text)

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else game / "game.sgf"
    out.write_text(sgf)
    winner = "Black" if meta["black_utility"] > 0 else "White"
    print(
        f"{out}\n  game {meta['game']}  radius {meta['radius']:.5f} "
        f"(1/{round(1 / meta['radius'])})  komi {meta['komi']:.4f}\n"
        f"  {meta['plies']} plies, {winner} won, "
        f"{'hit the ply cap' if meta['reached_ply_cap'] else 'ended naturally'}\n"
        f"  {len(records) - 1} moves written (the final ply cannot be differenced)"
    )


if __name__ == "__main__":
    main()
