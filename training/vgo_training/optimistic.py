"""A second policy head, weighted toward positions that beat expectations.

The main policy learns the average move. That is the right target for
generation, but at *search* time it makes MCTS reluctant to look at lines the
network currently undervalues -- which is exactly where a mistake in its
evaluation is hiding.

The optimistic head trains on the same targets with a per-sample weight that
rises when the outcome was better than the network predicted. Guiding search
with it makes MCTS spend visits on lines that have historically surprised the
net upward. KataGo measures **40 to 90 Elo** for this, "depending on other
settings" (`docs/KataGoMethods.md`).

It is used at test time only, never in self-play. KataGo is explicit that the
feedback effects of generating data under an optimistic policy are not
understood, and the same caution applies here: an optimistic prior that shapes
the data it is then trained on could drift somewhere unhelpful. Keeping it out
of generation makes this a change that cannot destabilise a run -- it is a
strictly inference-side lever, which is why it is worth trying early.
"""

from __future__ import annotations

import torch


def outcome_surprise(
    predicted_utility: torch.Tensor, outcome: torch.Tensor
) -> torch.Tensor:
    """How much better the result was than the prediction, in [0, 2].

    Both are mover-relative utilities in [-1, 1]. Only upward surprise counts:
    a position that went worse than expected is already handled by the main
    policy, and weighting it here would just duplicate that signal.
    """
    return (outcome - predicted_utility).clamp_min(0.0)


def optimistic_weights(
    surprise: torch.Tensor,
    *,
    baseline: float = 0.25,
    scale: float = 1.0,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Per-sample weights for the optimistic policy loss, averaging about 1.

    `baseline` keeps unsurprising positions contributing something, so the head
    stays a policy rather than collapsing onto the handful of lucky games.
    """
    if baseline < 0.0:
        raise ValueError("baseline must be nonnegative")
    if maximum <= 0.0:
        raise ValueError("maximum weight must be positive")
    weights = (baseline + scale * surprise).clamp(max=maximum)
    mean = weights.mean()
    if mean <= 0:
        return torch.ones_like(weights)
    # Renormalize so switching this on does not change the effective loss
    # scale, only its distribution across samples.
    return weights / mean


def optimistic_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross-entropy against the same targets the main policy uses."""
    log_probabilities = torch.log_softmax(
        logits.masked_fill(masks == 0, float("-inf")), dim=1
    )
    # Illegal actions carry no target mass, and 0 * -inf is NaN rather than 0.
    contributions = torch.where(
        targets > 0, targets * log_probabilities, torch.zeros_like(targets)
    )
    per_sample = -contributions.sum(dim=1)
    return (per_sample * weights).mean()
