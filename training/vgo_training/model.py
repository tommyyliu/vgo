from __future__ import annotations

import torch
from torch import nn


MODEL_ARCHITECTURES = ("flat", "unet", "ddrnet")


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


def residual_stack(
    width: int,
    blocks: int,
    variance_scaled: bool,
    start: int = 1,
    groups: int | None = None,
) -> list["ResidualBlock"]:
    """Blocks for one stack, scaled by depth when variance scaling is on.

    ``start`` is the trunk variance already accumulated when the stack begins,
    so a stack that continues an existing trunk keeps counting rather than
    restarting at 1. It is unused under normalization, which needs no notion of
    accumulated variance.
    """
    return [
        ResidualBlock(
            width,
            residual_scale=(
                (index**-0.5) if (variance_scaled and groups is None) else None
            ),
            groups=groups,
        )
        for index in range(start, start + blocks)
    ]


def apply_he_initialization(module: nn.Module) -> None:
    """Re-initialize convolutions to He/Kaiming scale for ReLU fan-in.

    ``nn.Conv2d`` defaults to Kaiming-uniform with ``a=sqrt(5)``, which is about
    2.4x below He scale for ReLU. Measured on a fresh DDRNet that leaves every
    block contractive -- the residual branch carries a fifth of the variance the
    skip does -- so training has to inflate weights merely to propagate signal,
    and then overshoots. Fixed-variance scaling assumes variance-preserving
    convolutions, so the two changes only make sense together: He alone raises
    the fresh peak to 4514, and scaling alone corrects growth that is not
    happening yet.
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
    """Residual block, optionally variance-scaled or normalized.

    ``residual_scale`` is KataGo's fixed-variance initialization: a constant
    where a normalization layer would otherwise sit, chosen so the idealized
    variance leaving the block is 1. Treating each conv-activation pair as
    variance-preserving and the skip sum as adding variances, a trunk whose
    blocks each contribute variance 1 reaches variance ``n`` at the nth block,
    so that block scales its residual branch by ``1/sqrt(n)``.

    ``groups`` instead puts a GroupNorm after each convolution, which is what
    the reference DDRNet does and what the scaling was standing in for.

    The two differ in what they can promise. A fixed constant is chosen once,
    from the weights' scale at initialization, and cannot respond when training
    moves them: measured on ddrnet-vs, weight scale grows sublinearly -- the
    context branch's He ratio fits sqrt(updates) with r=0.99 and its slope
    decays from 0.23 to 0.04 per update -- yet peak activation still compounds
    at 1.07x per update after update 30, because peak follows the *product* of
    per-layer gains and eight sublinear factors still multiply. Scaling every
    conv weight by 1.5 takes the scaled model from 308 to 665088, past fp16's
    65504; the same perturbation takes the normalized model from 9.7 to 15.3.
    Normalization divides the drift out at every block, so growth is polynomial
    rather than exponential.

    Cost is small: +4.9% forward in eager PyTorch, backward unchanged, +0.01M
    parameters, and it lowers to ONNX ``InstanceNormalization``.
    """

    def __init__(
        self,
        width: int,
        residual_scale: float | None = None,
        groups: int | None = None,
    ) -> None:
        super().__init__()
        if groups is not None and residual_scale is not None:
            raise ValueError(
                "a normalized block does not also take a residual scale"
            )
        self.residual_scale = residual_scale
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
        residual = self.layers(inputs)
        if self.residual_scale is not None:
            residual = residual * self.residual_scale
        return torch.relu(inputs + residual)


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
            # Two logits -- P(mover wins), P(mover loses) -- rather than a
            # scalar through tanh. The outcome is categorical, so a
            # distribution over categories is what the data actually is.
            #
            # tanh + MSE has gradient 2*(v - target)*(1 - v^2), and that last
            # factor is what killed learning: measured on 512 real positions
            # from update 11, the median damping was 0.0004 -- a 2500x weaker
            # gradient -- with 65% of positions under 0.01. A confidently wrong
            # evaluation produced almost no signal to correct it. Softmax
            # cross-entropy has gradient (p - target) in logit space, with no
            # such factor, so being wrong and certain is exactly the case that
            # learns fastest.
            #
            # Two classes rather than KataGo's three: it carries a no-result
            # class for ko and timeout, and our ties need black - white - komi
            # inside f64::EPSILON on continuous areas. Zero ties in 1400 games.
            nn.Linear(width, 2),
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
        values = value_utility(self.value_head(pooled))
        return policy_logits, values


class _Down(nn.Module):
    """Stride-2 conv downsample followed by residual blocks at the smaller scale."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        blocks: int,
        variance_scaled: bool = False,
        start: int = 1,
        groups: int | None = None,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            *residual_stack(channels_out, blocks, variance_scaled, start, groups)
        )

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
            # Two logits -- P(mover wins), P(mover loses) -- rather than a
            # scalar through tanh. The outcome is categorical, so a
            # distribution over categories is what the data actually is.
            #
            # tanh + MSE has gradient 2*(v - target)*(1 - v^2), and that last
            # factor is what killed learning: measured on 512 real positions
            # from update 11, the median damping was 0.0004 -- a 2500x weaker
            # gradient -- with 65% of positions under 0.01. A confidently wrong
            # evaluation produced almost no signal to correct it. Softmax
            # cross-entropy has gradient (p - target) in logit space, with no
            # such factor, so being wrong and certain is exactly the case that
            # learns fastest.
            #
            # Two classes rather than KataGo's three: it carries a no-result
            # class for ko and timeout, and our ties need black - white - komi
            # inside f64::EPSILON on continuous areas. Zero ties in 1400 games.
            nn.Linear(bottleneck, 2),
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
        values = value_utility(self.value_head(pooled))
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
        variance_scaled: bool = False,
        norm_groups: int | None = None,
    ) -> None:
        super().__init__()
        if stem_stride not in (1, 2, 4):
            raise ValueError("stem stride must be 1, 2, or 4")
        self.policy_resolution = policy_resolution
        self.stem_stride = stem_stride
        # Normalization supersedes the fixed scaling: both stand where a norm
        # would go, and a block takes one or the other.
        self.variance_scaled = variance_scaled and norm_groups is None
        self.norm_groups = norm_groups
        variance_scaled = self.variance_scaled
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
        # The detail branch is one continuous trunk across its three stacks, so
        # the variance count carries over rather than restarting per stack.
        self.detail_entry = nn.Sequential(
            *residual_stack(detail_channels, stage_blocks, variance_scaled, 1, norm_groups)
        )

        self.detail_stage1 = nn.Sequential(
            *residual_stack(
                detail_channels,
                stage_blocks,
                variance_scaled,
                1 + stage_blocks,
                norm_groups,
            )
        )
        self.context_stage1 = _Down(
            detail_channels,
            context_channels,
            stage_blocks,
            variance_scaled,
            groups=norm_groups,
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
            *residual_stack(
                detail_channels,
                stage_blocks,
                variance_scaled,
                1 + 2 * stage_blocks,
                norm_groups,
            )
        )
        self.context_stage2 = _Down(
            context_channels,
            deep_channels,
            stage_blocks,
            variance_scaled,
            start=1 + stage_blocks,
            groups=norm_groups,
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
            *residual_stack(context_channels, 1, variance_scaled, 1, norm_groups),
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
        self.value_head = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.ReLU(),
            # Two logits -- P(mover wins), P(mover loses) -- rather than a
            # scalar through tanh. The outcome is categorical, so a
            # distribution over categories is what the data actually is.
            #
            # tanh + MSE has gradient 2*(v - target)*(1 - v^2), and that last
            # factor is what killed learning: measured on 512 real positions
            # from update 11, the median damping was 0.0004 -- a 2500x weaker
            # gradient -- with 65% of positions under 0.01. A confidently wrong
            # evaluation produced almost no signal to correct it. Softmax
            # cross-entropy has gradient (p - target) in logit space, with no
            # such factor, so being wrong and certain is exactly the case that
            # learns fastest.
            #
            # Two classes rather than KataGo's three: it carries a no-result
            # class for ko and timeout, and our ties need black - white - komi
            # inside f64::EPSILON on continuous areas. Zero ties in 1400 games.
            nn.Linear(context_channels, 2),
        )

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
        self.value_head_normed = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.ReLU(),
            # Two logits -- P(mover wins), P(mover loses) -- rather than a
            # scalar through tanh. The outcome is categorical, so a
            # distribution over categories is what the data actually is.
            #
            # tanh + MSE has gradient 2*(v - target)*(1 - v^2), and that last
            # factor is what killed learning: measured on 512 real positions
            # from update 11, the median damping was 0.0004 -- a 2500x weaker
            # gradient -- with 65% of positions under 0.01. A confidently wrong
            # evaluation produced almost no signal to correct it. Softmax
            # cross-entropy has gradient (p - target) in logit space, with no
            # such factor, so being wrong and certain is exactly the case that
            # learns fastest.
            #
            # Two classes rather than KataGo's three: it carries a no-result
            # class for ko and timeout, and our ties need black - white - komi
            # inside f64::EPSILON on continuous areas. Zero ties in 1400 games.
            nn.Linear(context_channels, 2),
        )

        # He scale is what both schemes assume: the fixed constants are derived
        # from it, and a normalized block wants unit-variance convolutions too.
        if variance_scaled or norm_groups is not None:
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
            ownership = ownership_map(ownership_features(fused_features))
            ownership = torch.tanh(
                self._resize_policy(ownership, target_size)
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
    architecture: str,
    channels: int,
    width: int,
    blocks: int,
    policy_resolution: int | None = None,
    stem_stride: int = 4,
    variance_scaled: bool = False,
    norm_groups: int | None = None,
) -> nn.Module:
    """Construct a policy-value net by architecture name. Older checkpoints without
    an architecture field are the flat residual tower.

    `policy_resolution` coarsens the placement grid the policy head emits while
    leaving the input raster untouched; None keeps them equal.

    `variance_scaled` applies fixed-variance initialization and He-scale convs
    (ddrnet only). `norm_groups` instead puts GroupNorm in every residual block
    and supersedes it. Both change the function the network computes, so both
    must be recorded in the checkpoint and passed back when rebuilding for
    export -- rebuilding without them silently drops the constants or the
    normalization layers."""
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
            variance_scaled=variance_scaled,
            norm_groups=norm_groups,
        )
    raise ValueError(f"unknown model architecture: {architecture!r}")
