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

