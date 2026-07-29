from __future__ import annotations

import torch
from torch import nn


MODEL_ARCHITECTURES = ("flat", "unet", "ddrnet")


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.layers(inputs))


class RasterPolicyValueNet(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int = 32,
        blocks: int = 3,
        policy_resolution: int | None = None,
    ) -> None:
        super().__init__()
        # See UNetPolicyValueNet: the placement grid may be coarser than the
        # raster the tower reads. None keeps them equal.
        self.policy_resolution = policy_resolution
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.policy_map = nn.Conv2d(width, 1, kernel_size=1)
        self.pass_head = nn.Linear(width, 1)
        self.value_head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(self.stem(states))
        pooled = features.mean(dim=(-2, -1))
        if self.policy_resolution is not None:
            features = nn.functional.adaptive_avg_pool2d(
                features, (self.policy_resolution, self.policy_resolution)
            )
        placement_logits = self.policy_map(features).flatten(start_dim=1)
        pass_logit = self.pass_head(pooled)
        policy_logits = torch.cat((placement_logits, pass_logit), dim=1)
        values = self.value_head(pooled).squeeze(1)
        return policy_logits, values


class _Down(nn.Module):
    """Stride-2 conv downsample followed by residual blocks at the smaller scale."""

    def __init__(self, channels_in: int, channels_out: int, blocks: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels_out) for _ in range(blocks)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(self.reduce(inputs))


class _Up(nn.Module):
    """Bilinear upsample, concatenate the skip connection, fuse, then residual blocks."""

    def __init__(self, channels_in: int, channels_skip: int, channels_out: int, blocks: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels_in + channels_skip, channels_out, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels_out) for _ in range(blocks)))

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = torch.cat((self.up(inputs), skip), dim=1)
        return self.body(self.fuse(upsampled))


class UNetPolicyValueNet(nn.Module):
    """Encoder/bottleneck/decoder policy-value net with the same contract as
    RasterPolicyValueNet: input [B, C, H, W] -> policy_logits [B, H*W + 1], values [B].

    Full-resolution stages stay thin (placement detail only); nearly all residual
    blocks and channels live at the 4x-downsampled bottleneck, where convolution is
    ~16x cheaper per block. The policy map reads the full-resolution decoder output
    (placement precision preserved through skip connections); the pass and value
    heads read the bottleneck for global context.
    """

    def __init__(
        self,
        channels: int,
        width: int = 64,
        blocks: int = 8,
        policy_resolution: int | None = None,
    ) -> None:
        super().__init__()
        # The policy head may emit a coarser placement grid than the raster it
        # reads. The encoder/decoder still runs at full resolution, so the
        # Voronoi boundary channels keep their detail; only the placement output
        # is coarsened, which concentrates the coarse->fine proposal budget over
        # far fewer cells. None keeps the policy grid equal to the input raster.
        self.policy_resolution = policy_resolution
        shallow = max(16, width // 2)
        middle = width
        bottleneck = width * 2
        self.stem = nn.Sequential(
            nn.Conv2d(channels, shallow, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.enc0 = nn.Sequential(ResidualBlock(shallow))
        self.down1 = _Down(shallow, middle, 1)
        self.down2 = _Down(middle, bottleneck, blocks)
        self.up1 = _Up(bottleneck, middle, middle, 1)
        self.up2 = _Up(middle, shallow, shallow, 1)
        self.policy_map = nn.Conv2d(shallow, 1, kernel_size=1)
        self.pass_head = nn.Linear(bottleneck, 1)
        self.value_head = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip_full = self.enc0(self.stem(states))
        skip_mid = self.down1(skip_full)
        bottleneck = self.down2(skip_mid)
        decoded = self.up2(self.up1(bottleneck, skip_mid), skip_full)
        if self.policy_resolution is not None:
            # Pool the features, not the logits: averaging feature channels
            # before the 1x1 projection keeps more signal than averaging the
            # scalar logits it would otherwise produce.
            decoded = nn.functional.adaptive_avg_pool2d(
                decoded, (self.policy_resolution, self.policy_resolution)
            )
        placement_logits = self.policy_map(decoded).flatten(start_dim=1)
        pooled = bottleneck.mean(dim=(-2, -1))
        pass_logit = self.pass_head(pooled)
        policy_logits = torch.cat((placement_logits, pass_logit), dim=1)
        values = self.value_head(pooled).squeeze(1)
        return policy_logits, values


class _DDRContext(nn.Module):
    """A compact DAPPM-style context module for the small low-resolution branch.

    DDRNet's original five fixed pooling scales target 1024x2048 road scenes.
    VGO rasters leave only a 6x6 or 8x8 semantic map, so native, half, and global
    scales carry the distinct context that is available without redundant 1x1
    branches. As in DAPPM, each coarser scale is added to and processed from the
    preceding scale before all scales are compressed together.
    """

    def __init__(
        self, channels_in: int, branch_channels: int, channels_out: int
    ) -> None:
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.Conv2d(channels_in, branch_channels, kernel_size=1),
            nn.ReLU(),
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
        self.compression = nn.Conv2d(branch_channels * 3, channels_out, kernel_size=1)
        self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=1)

    @staticmethod
    def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return nn.functional.interpolate(
            inputs, size=size, mode="bilinear", align_corners=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        native = self.scale0(inputs)
        half = self.process1(self._resize(self.scale1(inputs), size) + native)
        global_context = self.process2(
            self._resize(self.scale2(inputs), size) + half
        )
        combined = torch.cat((native, half, global_context), dim=1)
        return torch.relu(self.compression(combined) + self.shortcut(inputs))


class DDRNetPolicyValueNet(nn.Module):
    """DDRNet-inspired dual-resolution policy/value network.

    The official DDRNet-23-slim keeps its detail branch at output stride 8 and
    drives the context branch down to stride 64. That schedule is efficient for
    megapixel road scenes but too coarse for a 96-128px game raster. This
    adaptation shifts the two branches one octave higher: policy detail remains
    at stride 4 while semantic context runs at strides 8 and 16. Two bilateral
    fusions repeatedly exchange precise placement geometry and global context.

    ``blocks`` remains checkpoint metadata shared by every architecture. Here it
    controls the number of residual blocks in each DDRNet stage in groups of
    four: 1-4 -> one block, 5-8 -> two blocks, and so on. Thus the common
    ``width=64, blocks=8`` setting corresponds to the two-block stages of
    DDRNet-23-slim without copying its scene-specific stride schedule.

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
    ) -> None:
        super().__init__()
        if stem_stride not in (1, 2, 4):
            raise ValueError("stem stride must be 1, 2, or 4")
        self.policy_resolution = policy_resolution
        self.stem_stride = stem_stride
        stem_channels = max(8, width // 2)
        detail_channels = width
        context_channels = width * 2
        deep_channels = width * 4
        stage_blocks = max(1, (blocks + 3) // 4)

        # The stem sets the resolution the whole tower reasons at, which the
        # default 4 puts at raster/4 -- 32x32 from a 128 input, where a stone of
        # radius 1/18 spans 3.6 cells and the context branch's 8x8 fusion sees
        # 0.89 cells per stone. Legal placement turns on a 2r clearance that is
        # sub-cell at those strides, so configurations that differ by whether a
        # gap is playable can be the same tensor to the model.
        #
        # Lowering it trades compute for spatial fidelity: stride 2 doubles the
        # detail branch's resolution, stride 1 keeps the raster's.
        first = 2 if stem_stride >= 2 else 1
        second = 2 if stem_stride >= 4 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(
                channels, stem_channels, kernel_size=3, stride=first, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                stem_channels,
                detail_channels,
                kernel_size=3,
                stride=second,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.detail_entry = nn.Sequential(
            *(ResidualBlock(detail_channels) for _ in range(stage_blocks))
        )

        self.detail_stage1 = nn.Sequential(
            *(ResidualBlock(detail_channels) for _ in range(stage_blocks))
        )
        self.context_stage1 = _Down(
            detail_channels, context_channels, stage_blocks
        )
        self.context_to_detail1 = nn.Conv2d(
            context_channels, detail_channels, kernel_size=1
        )
        self.detail_to_context1 = nn.Conv2d(
            detail_channels,
            context_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.detail_stage2 = nn.Sequential(
            *(ResidualBlock(detail_channels) for _ in range(stage_blocks))
        )
        self.context_stage2 = _Down(
            context_channels, deep_channels, stage_blocks
        )
        self.context_to_detail2 = nn.Conv2d(
            deep_channels, detail_channels, kernel_size=1
        )
        self.detail_to_context2 = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                context_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                context_channels,
                deep_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        context_branch = max(8, width // 2)
        self.context = _DDRContext(
            deep_channels, context_branch, context_channels
        )
        self.detail_tail = nn.Sequential(
            nn.Conv2d(detail_channels, context_channels, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(context_channels),
        )
        self.policy_features = nn.Sequential(
            nn.Conv2d(
                context_channels, detail_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.policy_map = nn.Conv2d(detail_channels, 1, kernel_size=1)
        self.pass_head = nn.Linear(context_channels, 1)
        self.value_head = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.ReLU(),
            nn.Linear(context_channels, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return nn.functional.interpolate(
            inputs, size=size, mode="bilinear", align_corners=False
        )

    @classmethod
    def _resize_policy(
        cls, inputs: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        pooled_size = (
            min(size[0], inputs.shape[-2]),
            min(size[1], inputs.shape[-1]),
        )
        if pooled_size != inputs.shape[-2:]:
            inputs = nn.functional.adaptive_avg_pool2d(inputs, pooled_size)
        if pooled_size != size:
            inputs = cls._resize(inputs, size)
        return inputs

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.detail_entry(self.stem(states))

        # Both directions consume the pre-fusion branch values. This is the
        # bilateral exchange that distinguishes DDRNet from a one-way decoder.
        detail_before = self.detail_stage1(detail)
        context_before = self.context_stage1(detail)
        detail = torch.relu(
            detail_before
            + self._resize(
                self.context_to_detail1(context_before),
                detail_before.shape[-2:],
            )
        )
        context = torch.relu(
            context_before + self.detail_to_context1(detail_before)
        )

        detail_before = self.detail_stage2(detail)
        context_before = self.context_stage2(context)
        detail = torch.relu(
            detail_before
            + self._resize(
                self.context_to_detail2(context_before),
                detail_before.shape[-2:],
            )
        )
        context = torch.relu(
            context_before + self.detail_to_context2(detail_before)
        )

        semantic = self.context(context)
        fused = torch.relu(
            self.detail_tail(detail)
            + self._resize(semantic, detail.shape[-2:])
        )
        placement_logits = self.policy_map(self.policy_features(fused))
        target_size = (
            (self.policy_resolution, self.policy_resolution)
            if self.policy_resolution is not None
            else states.shape[-2:]
        )
        placement_logits = self._resize_policy(
            placement_logits, target_size
        ).flatten(start_dim=1)

        pooled = semantic.mean(dim=(-2, -1))
        pass_logit = self.pass_head(pooled)
        policy_logits = torch.cat((placement_logits, pass_logit), dim=1)
        values = self.value_head(pooled).squeeze(1)
        return policy_logits, values


def build_model(
    architecture: str,
    channels: int,
    width: int,
    blocks: int,
    policy_resolution: int | None = None,
    stem_stride: int = 4,
) -> nn.Module:
    """Construct a policy-value net by architecture name. Older checkpoints without
    an architecture field are the flat residual tower.

    `policy_resolution` coarsens the placement grid the policy head emits while
    leaving the input raster untouched; None keeps them equal."""
    if architecture in ("", "flat", "raster"):
        return RasterPolicyValueNet(
            channels=channels, width=width, blocks=blocks, policy_resolution=policy_resolution
        )
    if architecture == "unet":
        return UNetPolicyValueNet(
            channels=channels, width=width, blocks=blocks, policy_resolution=policy_resolution
        )
    if architecture == "ddrnet":
        return DDRNetPolicyValueNet(
            channels=channels,
            width=width,
            blocks=blocks,
            policy_resolution=policy_resolution,
            stem_stride=stem_stride,
        )
    raise ValueError(f"unknown model architecture: {architecture!r}")
