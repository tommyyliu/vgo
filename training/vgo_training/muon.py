"""Muon: orthogonalized momentum for the 2D weights, Adam for everything else.

Muon replaces the raw gradient with its (approximate) orthogonalization, so
every direction in a weight matrix gets a comparable-sized step instead of the
update being dominated by a few large singular directions. It only makes sense
for matrices, so the usual arrangement -- and the one measured to work on this
project -- is hybrid: Muon on the 2D+ trunk weights, Adam on biases, norms,
embeddings, and the output heads.

Prior result on this codebase (w96/b8 conv net, 1.35M params): Muon 0.01 on the
tower reached all-Adam's *final* loss by epoch 4-5 and ended ~11% lower on
value MSE. **Learning rate 0.01 is the operating point; 0.02 destabilised the
value head and 0.04 diverged.**

The scale after Newton-Schulz is the part that is easy to get wrong, and there
are two valid conventions:

- `original` -- multiply by the aspect ratio `max(1, rows/cols)**0.5`, which is
  ~1 for squarish matrices. This is Muon's own convention and wants lr ~0.01.
- `match_rms_adamw` -- multiply by `0.2 * max(rows, cols)**0.5`, which rescales
  the update to AdamW's RMS so it can run at AdamW-like learning rates.

They are not interchangeable: using the second scale with the first learning
rate makes the step ~30x too hot. This module implements `original`, matching
the configuration already validated here.

Newton-Schulz uses the standard quintic coefficients (3.4445, -4.7750, 2.0315)
in bfloat16 for five steps -- these are tuned to maximise slope at zero rather
than to converge exactly, which is fine because only the direction matters.
"""

from __future__ import annotations

import torch
from torch import nn


def orthogonalize(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal polar factor of `matrix` by Newton-Schulz."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    # The iteration needs spectral norm <= 1 to stay in its basin.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    return x.to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Muon over 2D+ parameters. Pair it with Adam over the rest."""

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(
            list(parameters),
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(gradient)
                buffer = state["momentum"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = (
                    gradient.lerp(buffer, group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                # Conv filters orthogonalize as [out, in*kh*kw].
                original_shape = update.shape
                if update.ndim > 2:
                    update = update.reshape(len(update), -1)
                update = orthogonalize(update, group["ns_steps"])
                # Muon's own scale: ~1 for squarish matrices. Not
                # `0.2 * max(rows, cols)**0.5`, which belongs with AdamW-like
                # learning rates and is ~30x hotter at lr 0.01.
                update = update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                update = update.reshape(original_shape)
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


class HybridMuon(torch.optim.Optimizer):
    """One optimizer, Muon on groups flagged `use_muon` and Adam on the rest.

    The point runs use two separate optimizer objects, which is simpler. This
    exists for `PersistentLearner`, which owns exactly one `self.optimizer` and
    drives it through `zero_grad`/`step`/`state_dict` -- so the hybrid has to
    live inside a single object to be droppable there.
    """

    def __init__(
        self,
        groups,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(
            groups,
            dict(
                lr=1e-3,
                use_muon=False,
                momentum=momentum,
                nesterov=nesterov,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if group.get("use_muon"):
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(gradient)
                    buffer = state["momentum"]
                    buffer.lerp_(gradient, 1.0 - group["momentum"])
                    update = (
                        gradient.lerp(buffer, group["momentum"])
                        if group["nesterov"]
                        else buffer
                    )
                    shape = update.shape
                    if update.ndim > 2:
                        update = update.reshape(len(update), -1)
                    update = orthogonalize(update, group["ns_steps"])
                    update = (
                        update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                    )
                    update = update.reshape(shape)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                    continue

                # Plain Adam for heads, norms and biases.
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                denominator = (exp_avg_sq / bias2).sqrt().add_(group["eps"])
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(exp_avg / bias1, denominator, value=-group["lr"])
        return loss


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split into (Muon parameters, Adam parameters).

    Muon takes 2D+ weights in the trunk. Everything else goes to Adam: biases
    and norms (1D, no matrix structure to orthogonalize), embeddings (rows are
    looked up independently, so orthogonalizing across them is meaningless), and
    the output heads, whose matrices are rank-degenerate -- a `Linear(d, 1)` is
    a single row and orthogonalization would just renormalise it.
    """
    muon: list[nn.Parameter] = []
    adam: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith(
            ("value_head", "pass_head", "candidate_logit", "policy_decoder", "bias.")
        )
        if parameter.ndim >= 2 and not head and "embedding" not in name:
            muon.append(parameter)
        else:
            adam.append(parameter)
    return muon, adam
