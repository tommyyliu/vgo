"""Auxiliary value heads over short horizons.

The value target is the game's terminal outcome: one label shared by every
position in the game, arriving up to seventy plies after the position it
supervises. That is a high-variance signal -- a position can be winning and lose
anyway -- and it is the reason value MAE is the metric that moves most in this
project.

KataGo trains additional heads on exponentially averaged MCTS value over ~6, 16
and 50 turns. Those targets are biased (search is not truth) but far
lower-variance, and the classic bias-variance trade means the main head learns
faster with them alongside. They report it as one of the changes that sped up
training.

## This branch is not runnable yet

The targets do not exist in our shards. `generate_demo.rs` records only the
terminal utility per sample, and its own comment says the rest is unrecoverable:

    The root values are discarded after a playout, so this cannot be
    recovered later; it has to be measured here.

So finishing this needs Rust work first: record the root search value per ply,
carry it through `replay_stream`, and bump the shard version. Everything below
is the training half, written and tested against synthetic values so it is ready
when the data is -- and so the shape of the required data is unambiguous.
"""

from __future__ import annotations

import torch
from torch import nn

# Turn horizons KataGo uses. Short enough to be much lower variance than the
# outcome, long enough to carry more than the current position's evaluation.
DEFAULT_HORIZONS = (6, 16, 50)


def exponential_horizon_targets(
    root_values: torch.Tensor,
    game_ids: torch.Tensor,
    plies: torch.Tensor,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> torch.Tensor:
    """Exponentially averaged future search value, per horizon.

    For each sample, an average of the root values at later plies of the same
    game, weighted by `exp(-delta / horizon)`. Returns `[samples, len(horizons)]`.

    A game's tail has fewer future plies to average, so its targets lean on the
    outcome-adjacent values that remain. That is correct -- near the end, the
    search value *is* nearly the outcome.
    """
    if root_values.ndim != 1:
        raise ValueError("root values must be one per sample")
    if not (root_values.shape == game_ids.shape == plies.shape):
        raise ValueError("root values, game ids and plies must align")

    targets = torch.zeros(
        (root_values.numel(), len(horizons)), dtype=root_values.dtype
    )
    for game in torch.unique(game_ids):
        rows = (game_ids == game).nonzero(as_tuple=True)[0]
        order = rows[torch.argsort(plies[rows])]
        values = root_values[order]
        ply = plies[order].to(root_values.dtype)
        # Pairwise future deltas; the mask keeps each row to its own future.
        delta = ply.unsqueeze(0) - ply.unsqueeze(1)
        future = delta > 0
        for column, horizon in enumerate(horizons):
            weight = torch.where(
                future, torch.exp(-delta / float(horizon)), torch.zeros_like(delta)
            )
            total = weight.sum(dim=1)
            averaged = (weight * values.unsqueeze(0)).sum(dim=1) / total.clamp_min(
                1e-12
            )
            # The final ply has no future: fall back to its own value.
            targets[order, column] = torch.where(total > 0, averaged, values)
    return targets


class ShortTermValueHead(nn.Module):
    """Predicts the horizon-averaged search values from pooled features."""

    def __init__(self, channels: int, horizons: int = len(DEFAULT_HORIZONS)) -> None:
        super().__init__()
        self.horizons = horizons
        self.layers = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, horizons),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # Bounded like the main utility, since these are utilities too.
        return torch.tanh(self.layers(pooled))


def short_term_value_loss(
    predicted: torch.Tensor, targets: torch.Tensor, weight: float = 1.0
) -> torch.Tensor:
    """Mean absolute error against the horizon targets.

    MAE rather than MSE: these targets are noisy estimates, and squaring makes
    the loss chase whichever horizon happens to be most extreme.
    """
    return weight * (predicted - targets).abs().mean()
