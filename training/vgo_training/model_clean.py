"""The policy-value network.

    input   states        [B, C, H, W]
    output  policy_logits [B, P*P + 1]   P placements, then one pass logit
            values        [B, 2] logits while training, [B] utility at inference

One architecture: a DDRNet-style dual-resolution net. A flat residual tower and
a U-Net encoder/decoder lived here too, but nothing has trained either since the
dual-resolution net landed -- every pipeline config and every recent checkpoint
is `ddrnet`. They are in git history if a comparison is ever wanted again.

Two conventions apply throughout:

- **Value is categorical.** The heads emit two logits, P(mover wins) and
  P(mover loses). `value_utility` collapses them to the [-1, 1] scalar the
  search consumes. Training keeps the logits so the loss sees two classes;
  inference collapses. A tanh scalar was tried and abandoned: its
  `(1 - v^2)` gradient factor vanished exactly on confidently-wrong positions
  (median damping 0.0004), so the cases most needing correction learned slowest.
- **Placement may be coarser than the raster.** `policy_resolution` pools the
  policy features to a P x P grid while the tower still reads full resolution,
  concentrating the coarse->fine proposal budget over fewer cells. None keeps
  the two equal.
"""

from __future__ import annotations

import torch
from torch import nn


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


def value_utility(logits: torch.Tensor) -> torch.Tensor:
    """Mover-relative utility in [-1, 1] from win/loss logits: P(win) - P(loss)."""
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 0] - probabilities[:, 1]


class ResidualBlock(nn.Module):
    def __init__(self, width: int, groups: int = 8) -> None:
        super().__init__()
        divisor = next(g for g in range(min(groups, width), 0, -1) if width % g == 0)
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(divisor, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.layers(inputs))


def residual_stack(width: int, blocks: int, groups: int = 8) -> list[ResidualBlock]:
    """`blocks` residual blocks at `width`."""
    return [ResidualBlock(width, groups) for _ in range(blocks)]


def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return nn.functional.interpolate(
        inputs, size=size, mode="bilinear", align_corners=False
    )


def _resize_to_policy(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pool down and/or interpolate up so a map lands exactly on the policy grid."""
    pooled_size = (min(size[0], inputs.shape[-2]), min(size[1], inputs.shape[-1]))
    if pooled_size != inputs.shape[-2:]:
        inputs = nn.functional.adaptive_avg_pool2d(inputs, pooled_size)
    if pooled_size != size:
        inputs = _resize(inputs, size)
    return inputs


# --------------------------------------------------------------------------
# ddrnet
# --------------------------------------------------------------------------


class _Down(nn.Module):
    """Stride-2 conv downsample, then residual blocks at the smaller scale."""

    def __init__(
        self, channels_in: int, channels_out: int, blocks: int, groups: int = 8
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            *residual_stack(channels_out, blocks, groups)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(self.reduce(inputs))


class _DDRContext(nn.Module):
    """Compact DAPPM-style context module for the low-resolution branch.

    DDRNet's five pooling scales target megapixel road scenes; a VGO raster
    leaves only a 6x6 or 8x8 semantic map, so native, half, and global are the
    scales that carry distinct information. Each coarser scale is added to and
    processed from the preceding one before all three are compressed together.
    """

    def __init__(
        self, channels_in: int, branch_channels: int, channels_out: int
    ) -> None:
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.Conv2d(channels_in, branch_channels, kernel_size=1), nn.ReLU()
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels_in, branch_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_in, branch_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.process1 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.compression = nn.Conv2d(
            branch_channels * 3, channels_out, kernel_size=1
        )
        self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        native = self.scale0(inputs)
        half = self.process1(_resize(self.scale1(inputs), size) + native)
        global_context = self.process2(
            _resize(self.scale2(inputs), size) + half
        )
        combined = torch.cat((native, half, global_context), dim=1)
        return torch.relu(self.compression(combined) + self.shortcut(inputs))


class _PredictionHeads(nn.Module):
    """One complete set of output heads reading (semantic, fused) features.

    The net instantiates two of these. `plain` reads raw trunk features and is
    what inference and the exported graph use. `normed` reads batch-normalized
    features and carries most of the training loss.

    Why both: with no normalization anywhere, nothing penalizes weight
    magnitude, so training inflates it until the trunk overflows fp16 and the
    value head saturates. A norm in front of a head removes the incentive --
    trunk scale becomes a no-op on the normalized output. Keeping a second,
    unnormalized copy for inference means no BatchNorm running statistics reach
    export and there is no train/inference divergence.
    """

    def __init__(self, context_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Conv2d(context_channels, detail_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(detail_channels, 1, kernel_size=1),
        )
        self.pass_head = nn.Linear(context_channels, 1)
        self.value = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.ReLU(),
            # Two logits, P(win) and P(loss); `value_utility` collapses them.
            nn.Linear(context_channels, 2),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        fused: torch.Tensor,
        policy_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        placement = _resize_to_policy(self.policy(fused), policy_size).flatten(
            start_dim=1
        )
        pooled = semantic.mean(dim=(-2, -1))
        return (
            torch.cat((placement, self.pass_head(pooled)), dim=1),
            self.value(pooled),
        )


class DDRNetPolicyValueNet(nn.Module):
    """Dual-resolution net: a detail branch that keeps placement geometry and a
    context branch that gathers global information, exchanged by two bilateral
    fusions.

    Resolutions, for a 128px raster at the default `stem_stride=4`:

        stem      128 -> 32
        detail     32          (three stages, all at 32x32)
        context    16 -> 8     (downsampled once per stage)

    DDRNet-23-slim runs detail at stride 8 and context to stride 64, which is
    too coarse for a game raster; this shifts both branches one octave higher.

    `blocks` is shared checkpoint metadata across architectures. Here it sets
    the blocks per stage in groups of four: 1-4 -> one, 5-8 -> two, and so on.

    Reference: Hong et al., "Deep Dual-resolution Networks for Real-time and
    Accurate Semantic Segmentation of Road Scenes", arXiv:2101.06085.
    """

    def __init__(
        self,
        channels: int,
        width: int = 64,
        blocks: int = 8,
        policy_resolution: int | None = None,
        stem_stride: int = 4,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if stem_stride not in (1, 2, 4):
            raise ValueError("stem stride must be 1, 2, or 4")
        self.policy_resolution = policy_resolution
        self.stem_stride = stem_stride
        self.norm_groups = norm_groups

        stem_channels = max(8, width // 2)
        detail_channels = width
        context_channels = width * 2
        deep_channels = width * 4
        stage_blocks = max(1, (blocks + 3) // 4)

        # --- stem: sets the resolution the whole tower reasons at ------------
        # Lowering stem_stride trades compute for spatial fidelity. At the
        # default 4, a stone of radius 1/18 spans 3.6 cells and the context
        # branch's 8x8 fusion sees 0.89 cells per stone, so configurations
        # differing only by a sub-cell clearance gap can be the same tensor.
        first = 2 if stem_stride >= 2 else 1
        second = 2 if stem_stride >= 4 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(channels, stem_channels, kernel_size=3, stride=first, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                stem_channels, detail_channels, kernel_size=3, stride=second, padding=1
            ),
            nn.ReLU(),
        )

        # --- trunk: detail stays at one scale, context steps down ------------
        self.detail_entry = nn.Sequential(
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )
        self.detail_stage1 = nn.Sequential(
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )
        self.detail_stage2 = nn.Sequential(
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )
        self.context_stage1 = _Down(
            detail_channels, context_channels, stage_blocks, norm_groups
        )
        self.context_stage2 = _Down(
            context_channels, deep_channels, stage_blocks, norm_groups
        )

        # --- bilateral fusion: one pair of projections per exchange ----------
        self.context_to_detail1 = nn.Conv2d(
            context_channels, detail_channels, kernel_size=1
        )
        self.detail_to_context1 = nn.Conv2d(
            detail_channels, context_channels, kernel_size=3, stride=2, padding=1
        )
        self.context_to_detail2 = nn.Conv2d(
            deep_channels, detail_channels, kernel_size=1
        )
        self.detail_to_context2 = nn.Sequential(
            nn.Conv2d(
                detail_channels, context_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                context_channels, deep_channels, kernel_size=3, stride=2, padding=1
            ),
        )

        # --- context module and the map the heads read -----------------------
        self.context = _DDRContext(
            deep_channels, max(8, width // 2), context_channels
        )
        self.detail_tail = nn.Sequential(
            nn.Conv2d(detail_channels, context_channels, kernel_size=1),
            nn.ReLU(),
            *residual_stack(context_channels, 1, norm_groups),
        )

        # --- heads: see _PredictionHeads for why there are two sets ----------
        # Heads attach at two places -- value and pass read the pooled semantic
        # map, policy reads the fusion -- so covering both takes two norms
        # rather than DDRNet's single trunk norm.
        self.semantic_norm = nn.BatchNorm2d(context_channels)
        self.fused_norm = nn.BatchNorm2d(context_channels)
        self.heads = _PredictionHeads(context_channels, detail_channels)
        self.heads_normed = _PredictionHeads(context_channels, detail_channels)

        # No He re-initialization. torch's Conv2d default is Kaiming-uniform
        # with a=sqrt(5), about 2.4x below He scale for ReLU, and the previous
        # code overrode it. Measured at w96/b16: with GroupNorm the override
        # raised fresh policy logit std from 0.12 to 8.77 and peak activation
        # 1.68x, but the norm rescales every block so that washes out within a
        # few hundred steps. Dropped on that basis; git history has it.

    def _trunk(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stem, both branches, and both fusions. Returns (semantic, fused)."""
        detail = self.detail_entry(self.stem(states))

        # Both directions read the pre-fusion values. This simultaneous
        # exchange is what distinguishes DDRNet from a one-way decoder.
        detail_before = self.detail_stage1(detail)
        context_before = self.context_stage1(detail)
        detail = torch.relu(
            detail_before
            + _resize(
                self.context_to_detail1(context_before), detail_before.shape[-2:]
            )
        )
        context = torch.relu(context_before + self.detail_to_context1(detail_before))

        detail_before = self.detail_stage2(detail)
        context_before = self.context_stage2(context)
        detail = torch.relu(
            detail_before
            + _resize(
                self.context_to_detail2(context_before), detail_before.shape[-2:]
            )
        )
        context = torch.relu(context_before + self.detail_to_context2(detail_before))

        semantic = self.context(context)
        fused = torch.relu(
            self.detail_tail(detail) + _resize(semantic, detail.shape[-2:])
        )
        return semantic, fused

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Inference returns (policy_logits, values). Training additionally
        returns the normalized heads' outputs too, which carry most of the
        training loss:

            (policy, values, normed_policy, normed_values)
        """
        semantic, fused = self._trunk(states)
        policy_size = (
            (self.policy_resolution, self.policy_resolution)
            if self.policy_resolution is not None
            else states.shape[-2:]
        )

        policy_logits, values = self.heads(semantic, fused, policy_size)
        if not self.training:
            # Values collapse to the scalar utility, which is the ONNX contract
            # every Rust caller already expects.
            return policy_logits, value_utility(values)

        normed_logits, normed_values = self.heads_normed(
            self.semantic_norm(semantic), self.fused_norm(fused), policy_size
        )
        return policy_logits, values, normed_logits, normed_values


# --------------------------------------------------------------------------


def build_model(
    architecture: str,
    channels: int,
    width: int,
    blocks: int,
    policy_resolution: int | None = None,
    stem_stride: int = 4,
    norm_groups: int = 8,
) -> nn.Module:
    """Construct the policy-value net.

    `norm_groups` changes the function the network computes, so it is recorded
    in the checkpoint and passed back when rebuilding for export -- rebuilding
    without it silently drops the normalization layers.
    """
    if architecture != "ddrnet":
        raise ValueError(
            f"unknown model architecture: {architecture!r}; only 'ddrnet' "
            "remains -- see git history for the flat and unet towers"
        )
    return DDRNetPolicyValueNet(
        channels=channels,
        width=width,
        blocks=blocks,
        policy_resolution=policy_resolution,
        stem_stride=stem_stride,
        norm_groups=norm_groups,
    )
