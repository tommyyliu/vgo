"""Sample positions in proportion to how much search disagreed with the prior.

Uniform sampling spends the same gradient on a position where MCTS confirmed the
network's opinion as on one where it overturned it. The second kind is where the
learnable signal is: it is the only place the search found something the network
did not already know.

Measured on `ddrnet-packed` 2026-08-04, updates 12, 17 and 22 are pairwise
indistinguishable over 64 arena games each, while the same model searching at
1024 simulations beats itself at 512 by 5-1. Search is finding real improvements
that training is averaging away. This is the direct answer to that.

Following KataGo (`docs/KataGoMethods.md`): half the total weight is spread
uniformly, giving every position a baseline, and half is distributed in
proportion to the KL divergence from the policy prior to the search target.
KataGo applies no importance correction -- a surprising position is simply
sampled more often at full weight -- so the gradient is deliberately biased
toward surprise rather than reweighted to stay unbiased.

The prior here is the network's policy at the time the shard was generated,
which is not stored. `beta` is: the coarse-to-fine sampling probability per
cell, which *is* the behaviour policy those proposals were drawn from. So the
KL is computable from the shard as written, with no format change.
"""

from __future__ import annotations

import torch


def policy_kl_from_prior(
    targets: torch.Tensor,
    priors: torch.Tensor,
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Per-sample KL(target || prior) over legal actions.

    Both distributions are renormalized over the mask first, so a prior that
    put weight on illegal actions does not inflate the divergence.
    """
    live = masks.to(targets.dtype)
    target = targets * live
    prior = priors.to(targets.dtype) * live

    target = target / target.sum(dim=1, keepdim=True).clamp_min(epsilon)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(epsilon)

    ratio = (target.clamp_min(epsilon) / prior.clamp_min(epsilon)).log()
    # Contribute only where the target has mass: 0 * log(0) is 0 by convention
    # but NaN in floating point.
    return torch.where(target > 0, target * ratio, torch.zeros_like(target)).sum(dim=1)


def surprise_weights(
    divergence: torch.Tensor,
    *,
    uniform_share: float = 0.5,
    maximum: float = 8.0,
) -> torch.Tensor:
    """Frequency weights averaging 1.0, from per-sample divergences.

    `uniform_share` of the total weight is spread evenly and the rest follows
    the divergence, so at the default every position keeps a baseline of 0.5
    and no position is ever dropped entirely. `maximum` caps the tail: a single
    pathological KL should not consume a whole batch.
    """
    if not 0.0 <= uniform_share <= 1.0:
        raise ValueError("uniform share must be in [0, 1]")
    if maximum <= 0.0:
        raise ValueError("maximum weight must be positive")

    count = divergence.numel()
    if count == 0:
        return divergence.new_zeros(0)

    total = divergence.sum()
    if total <= 0:
        # No disagreement anywhere: fall back to uniform rather than dividing
        # by zero. Happens on a shard whose priors already match the targets.
        return torch.ones_like(divergence)

    shaped = divergence / total * count * (1.0 - uniform_share)
    return (uniform_share + shaped).clamp(max=maximum)


def sample_counts(
    weights: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Turn fractional weights into integer repeat counts.

    A weight of 2.3 becomes two copies plus a third with probability 0.3, which
    is unbiased in expectation. KataGo writes its shards this way rather than
    scaling the loss.
    """
    floor = weights.floor()
    remainder = weights - floor
    draw = torch.rand(weights.shape, generator=generator, device=weights.device)
    return (floor + (draw < remainder).to(weights.dtype)).to(torch.long)


def weighted_row_index(
    weights: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Row indices repeated by `sample_counts`, for building an epoch order."""
    counts = sample_counts(weights, generator)
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
