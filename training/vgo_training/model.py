from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .attention import BoardTransformerBlock



def _group_count(width: int, preferred: int) -> int:
    """Largest divisor of ``width`` not exceeding ``preferred``.

    Channel counts vary through the network -- 96 in the detail branch, 192 and
    384 in the context branch -- and GroupNorm requires the group count to
    divide the channels exactly. Falling back to the nearest divisor keeps one
    setting valid at every width instead of constraining the widths themselves.
    """
    for candidate in range(min(preferred, width), 0, -1):
        if width % candidate == 0:
            return candidate
    return 1


def residual_stack(width: int, blocks: int, groups: int | None) -> list["ResidualBlock"]:
    return [ResidualBlock(width, groups=groups) for _ in range(blocks)]


def value_head(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(channels, channels),
        nn.ReLU(),
        # Two logits -- P(mover wins), P(mover loses) -- rather than a scalar
        # through tanh. tanh + MSE has gradient 2*(v - target)*(1 - v^2), and
        # that last factor is what killed learning: measured on 512 real
        # positions, the median damping was 0.0004 -- a 2500x weaker gradient.
        # Softmax cross-entropy has gradient (p - target) in logit space, so
        # being wrong and certain is exactly the case that learns fastest.
        #
        # Two classes rather than KataGo's three: it carries a no-result class
        # for ko and timeout, and our ties need black - white - komi inside
        # f64::EPSILON on continuous areas. Zero ties in 1400 games.
        nn.Linear(channels, 2),
    )


def apply_he_initialization(module: nn.Module) -> None:
    """Re-initialize convolutions to He/Kaiming scale for ReLU fan-in.

    ``nn.Conv2d`` defaults to Kaiming-uniform with ``a=sqrt(5)``, which is about
    2.4x below He scale for ReLU. Measured on a fresh DDRNet that leaves every
    block contractive -- the residual branch carries a fifth of the variance the
    skip does -- so training has to inflate weights merely to propagate signal.
    """
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(
                child.weight, mode="fan_in", nonlinearity="relu"
            )
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def value_utility(logits: torch.Tensor) -> torch.Tensor:
    """Mover-relative utility in [-1, 1] from win/loss logits.

    P(win) - P(loss) over the two-class softmax, which for two classes is
    tanh(z_win - z_loss) / 1 -- bounded by construction rather than by a
    squashing layer, so the bound costs no gradient. This is what the search
    consumes and what the exported graph emits, so the ONNX contract and every
    Rust caller are unchanged by the head becoming categorical.
    """
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 0] - probabilities[:, 1]


class ResidualBlock(nn.Module):
    """Residual block, optionally with a GroupNorm after each convolution.

    Normalization divides weight drift out at every block. Without it, peak
    activation compounds with the product of per-layer gains: scaling every
    conv weight by 1.5 took an unnormalized model from 308 to 665088, past
    fp16's 65504, and the normalized one from 9.7 to 15.3. Cost is +4.9%
    forward in eager PyTorch, and it lowers to ONNX ``InstanceNormalization``.
    """

    def __init__(self, width: int, groups: int | None = None) -> None:
        super().__init__()
        if groups is None:
            self.layers = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
            )
        else:
            divisor = _group_count(width, groups)
            self.layers = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(divisor, width),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(divisor, width),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.layers(inputs))


class _Down(nn.Module):
    """Stride-2 conv downsample followed by residual blocks at the smaller scale.

    `attention_blocks` replaces that many trailing residual blocks with
    transformer blocks, which relate every cell to every other one directly
    rather than through stacked local receptive fields. It needs `board` -- the
    (height, width) reaching this stage -- because rotary position tables are
    precomputed per resolution.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        blocks: int,
        groups: int | None = None,
        attention_blocks: int = 0,
        attention_heads: int = 8,
        board: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if attention_blocks and board is None:
            raise ValueError("attention blocks need the board size at this stage")
        if attention_blocks > blocks:
            raise ValueError(
                f"cannot replace {attention_blocks} of {blocks} residual blocks"
            )
        self.reduce = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        kept = blocks - attention_blocks
        self.body = nn.Sequential(
            *residual_stack(channels_out, kept, groups)
        )
        # Held apart from `body` because a transformer block takes an optional
        # mask that nn.Sequential cannot thread through.
        self.attention = nn.ModuleList(
            BoardTransformerBlock(
                channels_out, attention_heads, board[0], board[1],
                rope_theta=max(100.0, 4.0 * max(board)),
            )
            for _ in range(attention_blocks)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.body(self.reduce(inputs))
        for block in self.attention:
            features = block(features)
        return features


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

    ``blocks`` controls the number of residual blocks in each DDRNet stage in groups of
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
        norm_groups: int | None = None,
        context_attention_blocks: int = 0,
        attention_heads: int = 8,
        raster_resolution: int | None = None,
    ) -> None:
        super().__init__()
        if stem_stride not in (1, 2, 4):
            raise ValueError("stem stride must be 1, 2, or 4")
        # Attention is the only part of this net that is not resolution-agnostic:
        # rotary position tables are built per board size, so a model with
        # attention is fixed to the raster it was constructed for.
        if context_attention_blocks and raster_resolution is None:
            raise ValueError(
                "context attention needs raster_resolution to size its position tables"
            )
        self.policy_resolution = policy_resolution
        self.stem_stride = stem_stride
        self.context_attention_blocks = context_attention_blocks
        self.attention_heads = attention_heads
        self.raster_resolution = raster_resolution
        # Stem divides by stem_stride; each context stage halves again.
        trunk = None if raster_resolution is None else raster_resolution // stem_stride
        context1_board = None if trunk is None else (trunk // 2, trunk // 2)
        context2_board = None if trunk is None else (trunk // 4, trunk // 4)
        self.norm_groups = norm_groups
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
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )

        self.detail_stage1 = nn.Sequential(
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )
        self.context_stage1 = _Down(
            detail_channels,
            context_channels,
            stage_blocks,
            groups=norm_groups,
            attention_blocks=context_attention_blocks,
            attention_heads=attention_heads,
            board=context1_board,
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
            *residual_stack(detail_channels, stage_blocks, norm_groups)
        )
        self.context_stage2 = _Down(
            context_channels,
            deep_channels,
            stage_blocks,
            groups=norm_groups,
            attention_blocks=context_attention_blocks,
            attention_heads=attention_heads,
            board=context2_board,
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

        # One batch norm, after KataGo's method. Without normalization anywhere,
        # nothing penalizes weight magnitude: scaling a conv up costs nothing, so
        # training inflates it. Measured on ddrnet-fp32 update 2, the residual
        # weights drift to 7.6x He scale, each conv multiplies std by ~24x, the
        # trunk peaks at 68824 against fp16's 65504 limit, and the value head's
        # tanh saturates on 67% of positions. A freshly initialized net peaks at
        # 1.5 with 0% saturation, so this is drift, not the topology.
        #
        # A norm in front of the heads removes the incentive: trunk weight scale
        # becomes a no-op on the normalized output, so there is nothing to gain
        # by growing it. Heads attach at two places here -- value and pass read
        # the pooled 8x8 semantic map, policy reads the 32x32 fusion -- so
        # covering all three takes two norms rather than DDRNet's single trunk.
        #
        # Each norm feeds a *training* head that carries most of the loss. A
        # second copy of each head reads the unnormalized features and carries
        # the rest; that copy is what inference uses, so no running statistics
        # are needed at export and there is no train/inference divergence.
        self.semantic_norm = nn.BatchNorm2d(context_channels)
        self.fused_norm = nn.BatchNorm2d(context_channels)
        self.detail_tail = nn.Sequential(
            nn.Conv2d(detail_channels, context_channels, kernel_size=1),
            nn.ReLU(),
            *residual_stack(context_channels, 1, norm_groups),
        )
        self.policy_features = nn.Sequential(
            nn.Conv2d(
                context_channels, detail_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.policy_map = nn.Conv2d(detail_channels, 1, kernel_size=1)
        # Ownership: who holds each cell when the game ends, in [-1, 1] from the
        # mover's view. Spatial rather than scalar on purpose -- a game's ~58
        # positions all share one value label, which a net of this capacity
        # memorises by trajectory (training MAE 0.040 against validation 0.467).
        # Ownership varies within a game, so the same trajectory cannot collapse
        # to one number.
        self.ownership_features = nn.Sequential(
            nn.Conv2d(context_channels, detail_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ownership_map = nn.Conv2d(detail_channels, 1, kernel_size=1)
        self.pass_head = nn.Linear(context_channels, 1)
        self.value_head = value_head(context_channels)

        # The normalized twins. These see batch-normalized features and take the
        # bulk of the loss, so they drive optimization; the heads above learn the
        # same predictions from unnormalized features and are used at inference.
        self.policy_features_normed = nn.Sequential(
            nn.Conv2d(
                context_channels, detail_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.policy_map_normed = nn.Conv2d(detail_channels, 1, kernel_size=1)
        self.ownership_features_normed = nn.Sequential(
            nn.Conv2d(context_channels, detail_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ownership_map_normed = nn.Conv2d(detail_channels, 1, kernel_size=1)
        self.pass_head_normed = nn.Linear(context_channels, 1)
        self.value_head_normed = value_head(context_channels)

        # A normalized block wants unit-variance convolutions.
        if norm_groups is not None:
            apply_he_initialization(self)

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
        target_size = (
            (self.policy_resolution, self.policy_resolution)
            if self.policy_resolution is not None
            else states.shape[-2:]
        )

        def heads(
            semantic_features: torch.Tensor,
            fused_features: torch.Tensor,
            policy_features: nn.Module,
            policy_map: nn.Module,
            pass_head: nn.Module,
            value_head: nn.Module,
            ownership_features: nn.Module | None = None,
            ownership_map: nn.Module | None = None,
        ) -> tuple[torch.Tensor, ...]:
            placement = policy_map(policy_features(fused_features))
            placement = self._resize_policy(placement, target_size).flatten(
                start_dim=1
            )
            pooled = semantic_features.mean(dim=(-2, -1))
            logits = torch.cat((placement, pass_head(pooled)), dim=1)
            values = value_head(pooled)
            if ownership_features is None:
                return logits, values
            # Same resize as the policy so the map lands on the policy grid,
            # which is the resolution the target is rendered at.
            # Raw, not through tanh. A tanh here saturates at initialisation --
            # measured -1.0000 to -0.9993 on a fresh model, with gradients of
            # 1e-6 -- because the (1 - v^2) factor vanishes exactly where the
            # output is pinned. That is the same trap the scalar value head was
            # in before it became categorical. MSE against +/-1 targets does not
            # need a bounded output; the head learns the bound from the data.
            ownership = self._resize_policy(
                ownership_map(ownership_features(fused_features)), target_size
            ).flatten(start_dim=1)
            return logits, values, ownership

        policy_logits, values = heads(
            semantic,
            fused,
            self.policy_features,
            self.policy_map,
            self.pass_head,
            self.value_head,
        )
        if not self.training:
            # Ownership is an auxiliary target, not something the search reads,
            # so it stays out of the exported graph entirely. The value logits
            # collapse to the scalar utility here, so the exported graph keeps
            # emitting exactly what the search always consumed.
            return policy_logits, value_utility(values)

        # Training only. The normalized heads carry most of the loss and so are
        # what actually shapes the trunk; returning them lets the learner add
        # their loss without the exported graph ever seeing a BatchNorm.
        ownership = heads(
            semantic,
            fused,
            self.policy_features,
            self.policy_map,
            self.pass_head,
            self.value_head,
            self.ownership_features,
            self.ownership_map,
        )[2]
        normed_logits, normed_values, normed_ownership = heads(
            self.semantic_norm(semantic),
            self.fused_norm(fused),
            self.policy_features_normed,
            self.policy_map_normed,
            self.pass_head_normed,
            self.value_head_normed,
            self.ownership_features_normed,
            self.ownership_map_normed,
        )
        return (
            policy_logits,
            values,
            normed_logits,
            normed_values,
            ownership,
            normed_ownership,
        )


def build_model(
    channels: int,
    width: int,
    blocks: int,
    policy_resolution: int | None = None,
    stem_stride: int = 4,
    norm_groups: int | None = None,
    context_attention_blocks: int = 0,
    attention_heads: int = 8,
    raster_resolution: int | None = None,
) -> DDRNetPolicyValueNet:
    """`policy_resolution` coarsens the placement grid the policy head emits
    while leaving the input raster untouched; None keeps them equal.

    `norm_groups` changes the function the network computes, so it must be
    recorded in the checkpoint and passed back when rebuilding for export."""
    return DDRNetPolicyValueNet(
        channels=channels,
        width=width,
        blocks=blocks,
        policy_resolution=policy_resolution,
        stem_stride=stem_stride,
        norm_groups=norm_groups,
        context_attention_blocks=context_attention_blocks,
        attention_heads=attention_heads,
        raster_resolution=raster_resolution,
    )


def load_model(checkpoint_path: Path) -> tuple[DDRNetPolicyValueNet, dict[str, object]]:
    """Rebuild a checkpoint's network in eval mode, returning it and the raw checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    architecture = checkpoint.get("architecture")
    if architecture != "ddrnet" or checkpoint.get("variance_scaled"):
        raise ValueError(
            f"{checkpoint_path}: only normalized ddrnet checkpoints load here "
            f"(architecture={architecture!r}); older models need the "
            "archive/pre-prune branch"
        )
    height = int(checkpoint["height"])
    stored_policy = checkpoint.get("policy_resolution")
    model = build_model(
        channels=int(checkpoint["channels"]),
        width=int(checkpoint["model_width"]),
        blocks=int(checkpoint["blocks"]),
        policy_resolution=(
            int(stored_policy)
            if stored_policy is not None and int(stored_policy) != height
            else None
        ),
        norm_groups=checkpoint.get("norm_groups"),
        context_attention_blocks=int(checkpoint.get("context_attention_blocks", 0)),
        attention_heads=int(checkpoint.get("attention_heads", 8)),
        raster_resolution=height,
    )
    # The batch-normalized twin heads and the ownership head exist only while
    # training, so an exported model lacks them. Everything inference reads
    # still has to be present.
    missing, _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
    required = [
        name
        for name in missing
        if not ("_normed" in name or "_norm." in name or name.startswith("ownership_"))
    ]
    if required:
        raise RuntimeError(
            f"checkpoint is missing {len(required)} inference weight(s), "
            f"starting with {required[0]}"
        )
    model.eval()
    return model, checkpoint
