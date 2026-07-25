"""Render what the network actually receives: all 10 raster channels per position.

Generation writes BMPs for a couple of sample positions, but the interesting
question is what a *range* of positions looks like, so this reads the replay
tensor directly and renders channels straight from the data the net is fed.
"""

import sys
from pathlib import Path

ROOT = Path("/home/tommy/PycharmProjects/vgo")
sys.path.insert(0, str(ROOT / "training"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from vgo_training.dataset import load_dataset

OUT = Path(
    "/tmp/claude-1000/-home-tommy-PycharmProjects-vgo/57b46eeb-7b26-4811-a630-70ee959d90ee/scratchpad"
)

NAMES = [
    "0 current_stones",
    "1 opponent_stones",
    "2 current_voronoi",
    "3 opponent_voronoi",
    "4 current_distance",
    "5 opponent_distance",
    "6 voronoi_ridge",
    "7 legal_clearance",
    "8 radius",
    "9 previous_pass",
]


def main() -> None:
    dataset = load_dataset(
        ROOT / "artifacts/decoupled-run2/iteration-001/replay/dataset.vgo"
    )
    # Spread across the game: opening, midgame, endgame.
    plies = dataset.plies
    picks = []
    for target in (0, 10, 30, 60):
        candidates = (plies == target).nonzero().flatten()
        if len(candidates):
            picks.append((target, int(candidates[0])))
    print("showing plies:", [p for p, _ in picks])

    rows = len(picks)
    figure, axes = plt.subplots(rows, 10, figsize=(22, 2.4 * rows))
    for row, (ply, index) in enumerate(picks):
        state = dataset.states[index]
        for channel in range(10):
            plane = state[channel].numpy()
            axis = axes[row, channel]
            # Channel 7 is signed; the rest are unit-range.
            if channel == 7:
                axis.imshow(plane, cmap="RdBu", vmin=-1, vmax=1)
            else:
                axis.imshow(plane, cmap="viridis", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(NAMES[channel], fontsize=8)
            if channel == 0:
                axis.set_ylabel(f"ply {ply}", fontsize=9)
    figure.suptitle(
        "Network input: 10 semantic channels at 96x96, radius 1/18 (9-across board)",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(OUT / "input_channels.png", dpi=110)
    print("wrote input_channels.png")

    # A compact composite: the channels that carry the geometry, side by side
    # with the policy target, for one midgame position.
    _, index = picks[min(2, len(picks) - 1)]
    state = dataset.states[index]
    cells = dataset.policies.shape[1] - 1
    side = int(round(cells**0.5))
    target = dataset.policies[index, :cells].reshape(side, side)

    figure, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    axes[0].imshow(state[0] - state[1], cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("stones (current - opponent)")
    axes[1].imshow(state[2] - state[3], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Voronoi ownership")
    axes[2].imshow(state[6], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("ridge (channel 6)")
    axes[3].imshow(state[7], cmap="RdBu", vmin=-1, vmax=1)
    axes[3].set_title("legal clearance (channel 7)")
    axes[4].imshow(target, cmap="magma")
    axes[4].set_title(f"policy target ({side}x{side})")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("What the net sees (96x96) vs what it must predict (32x32)")
    figure.tight_layout()
    figure.savefig(OUT / "input_vs_target.png", dpi=120)
    print("wrote input_vs_target.png")


if __name__ == "__main__":
    main()
