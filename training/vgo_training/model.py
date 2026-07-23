from __future__ import annotations

import torch
from torch import nn


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
    def __init__(self, channels: int, width: int = 32, blocks: int = 3) -> None:
        super().__init__()
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

    def __init__(self, channels: int, width: int = 64, blocks: int = 8) -> None:
        super().__init__()
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
        placement_logits = self.policy_map(decoded).flatten(start_dim=1)
        pooled = bottleneck.mean(dim=(-2, -1))
        pass_logit = self.pass_head(pooled)
        policy_logits = torch.cat((placement_logits, pass_logit), dim=1)
        values = self.value_head(pooled).squeeze(1)
        return policy_logits, values


def build_model(architecture: str, channels: int, width: int, blocks: int) -> nn.Module:
    """Construct a policy-value net by architecture name. Older checkpoints without
    an architecture field are the flat residual tower."""
    if architecture in ("", "flat", "raster"):
        return RasterPolicyValueNet(channels=channels, width=width, blocks=blocks)
    if architecture == "unet":
        return UNetPolicyValueNet(channels=channels, width=width, blocks=blocks)
    raise ValueError(f"unknown model architecture: {architecture!r}")

