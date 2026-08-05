"""Sample recent shards more often than old ones.

A replay window trades two things against each other. A long window holds more
distinct games, which is diversity the model needs; but it also holds games
played by models several generations stale, and training on those pulls the
network back toward what it already was.

Uniform sampling forces a choice between them. Weighting by age does not: hold a
long window for diversity and draw from its recent end more often, so the
gradient follows current play while the tail still contributes.

The weight is exponential in shard age, `decay ** age`, normalized to average
1.0 so switching it on redistributes the sampling budget without changing its
size. `decay=1.0` is uniform, which is the default -- this changes what the
model trains on, so it is opt-in.

Age is measured in shards back from the newest in the window, not in wall time:
what matters is how many model generations separate the data from the learner,
and one shard is one generation.
"""

from __future__ import annotations

import torch


def shard_weights(
    shard_ages: torch.Tensor, decay: float = 1.0, *, floor: float = 0.005
) -> torch.Tensor:
    """Per-shard sampling weight from age, averaging 1.0.

    `shard_ages` is 0 for the newest shard, 1 for the one before it, and so on.
    `decay` is the per-shard factor: 0.9 makes each older shard 10% less likely
    than its successor, 0.5 halves it.

    `floor` keeps the oldest shard contributing something. Without it a long
    window with aggressive decay silently becomes a short one, which is the
    failure this is meant to avoid -- the point is to keep diversity, not to
    discard it more cheaply.

    The floor binds sooner than it looks. It clamps the raw weight *before*
    normalization, so it caps the newest-to-oldest ratio at `1/floor`: at 0.05
    every decay past ~0.7 gives the same 20x spread over a 12-shard window, and
    dialing decay down further does nothing. 0.005 keeps the tail alive while
    leaving the useful range actually reachable.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    if shard_ages.numel() == 0:
        return shard_ages.new_zeros(0)

    raw = torch.pow(
        torch.tensor(decay, dtype=torch.float64), shard_ages.to(torch.float64)
    )
    raw = raw.clamp_min(floor)
    return (raw / raw.mean()).to(torch.float32)


def row_weights(
    shard_sizes: list[int], decay: float = 1.0, *, floor: float = 0.005
) -> torch.Tensor:
    """Per-row weights for a window, newest shard first.

    `shard_sizes` gives the row count of each shard in the order the view
    concatenates them. Returns one weight per row, averaging 1.0, so it can be
    handed straight to `ReplayView.batches(weights=...)`.

    Shards are equal-sized in practice, but they are not required to be: the
    per-shard weight is a property of age, and every row in a shard shares it.
    """
    if decay == 1.0:
        return torch.ones(sum(shard_sizes))
    ages = torch.arange(len(shard_sizes), dtype=torch.float32)
    per_shard = shard_weights(ages, decay, floor=floor)
    weights = torch.repeat_interleave(
        per_shard, torch.tensor(shard_sizes, dtype=torch.long)
    )
    # Renormalize over rows: unequal shard sizes would otherwise let a large
    # old shard outweigh a small new one despite its lower per-row weight.
    return weights / weights.mean()


def effective_window(shard_sizes: list[int], decay: float, *, floor: float = 0.005) -> float:
    """How many shards the weighted window behaves like.

    Reports `1 / sum(p^2)` over the normalized per-shard sampling distribution,
    the standard participation ratio: a uniform window over N shards gives N, and
    a window that draws almost everything from one shard gives ~1. Use it to see
    what a decay actually costs in diversity before running with it.
    """
    weights = row_weights(shard_sizes, decay, floor=floor)
    total = torch.tensor(shard_sizes, dtype=torch.float32)
    per_shard = torch.stack(
        [
            weights[int(offset) : int(offset + size)].sum()
            for offset, size in zip(
                torch.cat([torch.zeros(1), total.cumsum(0)[:-1]]), total
            )
        ]
    )
    share = per_shard / per_shard.sum()
    return float(1.0 / (share * share).sum())
