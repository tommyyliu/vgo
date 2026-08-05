"""Global pooling that gives the heads more than an average.

The value and pass heads read one number per channel, produced by a mean over
the board. A mean answers "how much of this feature is there on average" and
nothing else: it cannot say whether a feature is concentrated in one corner or
spread thin, and those are different board states.

KataGo's `KataGPool` concatenates three statistics instead of one -- mean, a
board-size-scaled mean, and max -- tripling the channels reaching each head. The
max is what a mean cannot express: the strongest single response anywhere, which
is how "there exists a threatened group" looks to a convolutional feature.

The scaled mean matters less here than for KataGo. Theirs handles boards from
9x9 to 19x19 in one net, so a head needs to know which it is looking at; ours
runs one resolution per model. It is kept because it costs a multiply and
because a model trained at one raster may later be run at another.
"""

from __future__ import annotations

import torch
from torch import nn

# KataGo centres the size term on a 19x19 board (sqrt(361) = 19) and divides by
# 10 to keep it near unit scale. Ours is a continuous board rendered to a
# square raster, so the reference point is that raster's own side length.
_SIZE_SCALE = 10.0


def global_pool(
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    size_reference: float | None = None,
) -> torch.Tensor:
    """Pool `[N, C, H, W]` to `[N, 3C]`: mean, scaled mean, max.

    `mask` is `[N, 1, H, W]` with 1 on live cells; None treats every cell as
    live, which is the usual case for a fully rendered raster.
    """
    batch, channels, height, width = features.shape
    if mask is None:
        live = float(height * width)
        mean = features.mean(dim=(2, 3))
        peak = features.amax(dim=(2, 3))
        live_per_sample = features.new_full((batch, 1), live)
    else:
        live_per_sample = mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean = (features * mask).sum(dim=(2, 3)) / live_per_sample
        # Off-board cells must not win the max. Every activation reaching here
        # is post-ReLU and so at least 0, and this makes masked cells -1.
        peak = (features + (mask - 1.0)).amax(dim=(2, 3))

    reference = size_reference if size_reference is not None else float(height)
    scale = (live_per_sample.sqrt() - reference) / _SIZE_SCALE
    return torch.cat((mean, mean * scale, peak), dim=1)


class GlobalPool(nn.Module):
    """`global_pool` as a module, so a head can declare its input width.

    A head reading pooled features needs `3 * channels` inputs rather than
    `channels`; `output_width` exists so that is stated once at construction
    instead of being open-coded at every call site.
    """

    def __init__(self, channels: int, size_reference: float | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.size_reference = size_reference

    @property
    def output_width(self) -> int:
        return 3 * self.channels

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {features.shape[1]}"
            )
        return global_pool(features, mask, size_reference=self.size_reference)
