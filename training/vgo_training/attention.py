"""Transformer blocks for a board, in the smallest form that still works.

The board is treated as a sequence of `height * width` tokens, one per cell.
Attention then relates any two cells directly, where a convolution needs depth
to do the same -- which is the reason to want it here: territory and group
connectivity are global relationships over a Voronoi board.

Three ideas carry the whole file:

**Pre-norm residual.** Each block returns only what to *add* to the trunk, and
normalizes its input rather than its output. Gradients reach early layers
through the untouched residual path, which is what makes depth trainable.

**Rotary position.** Attention is permutation-invariant, so position has to be
supplied. Rather than adding a position embedding, RoPE *rotates* each query and
key by an angle derived from its coordinates. The dot product between two
tokens then depends only on their relative offset, which is the property worth
having on a board. Half the head dimension encodes the row, half the column.

**SwiGLU.** The feed-forward half computes `silu(W1 x) * (W3 x)` instead of
`relu(W1 x)`: one branch decides *how much* of the other to let through. It
costs a third projection and reliably beats the plain form.

This is a from-scratch reading of KataGo's `model_pytorch.py`. The faithful port
lives in `katago_transformer.py`; the two agree numerically (see
`tests/test_attention.py`). Dropped here as board-specific or infrastructural:
geometric/topological attention biases, register tokens, grouped-query
attention, and the fp16 logit-penalty instrumentation.
"""

from __future__ import annotations

import torch
from torch import nn


def rope_tables(head_dim: int, height: int, width: int, theta: float = 100.0):
    """Cos/sin rotation tables for every cell, shaped `(height * width, head_dim)`.

    `theta` sets the longest wavelength. It must comfortably exceed the board
    size, or distant cells wrap around to the same angle and become
    indistinguishable.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")

    # Half the dimension per axis, and each axis pairs up its components, so a
    # quarter of head_dim gives the frequencies for one axis.
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))

    rows = torch.arange(height, dtype=torch.float32)
    cols = torch.arange(width, dtype=torch.float32)
    grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")

    angles = torch.cat(
        [grid_row.unsqueeze(-1) * freqs, grid_col.unsqueeze(-1) * freqs], dim=-1
    )
    # Flatten the board to a sequence, then duplicate each angle so it applies
    # to both halves of the (x, y) pair it rotates.
    angles = angles.flatten(0, 1).repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent pair of components of `x` by the given angles.

    Treating components (a, b) as the complex number a + bi, this is
    multiplication by e^(i*angle). `x` is (batch, seq, heads, dim).
    """
    paired = x.reshape(*x.shape[:-1], -1, 2)
    a, b = paired.unbind(dim=-1)
    swapped = torch.stack([-b, a], dim=-1).flatten(-2)

    cos = cos.view(1, x.shape[1], 1, x.shape[-1])
    sin = sin.view(1, x.shape[1], 1, x.shape[-1])
    return x * cos + swapped * sin


def board_attention_mask(mask: torch.Tensor | None, batch: int, seq: int, dtype):
    """Additive mask that sends attention to off-board cells to zero.

    `mask` is N1HW with 1 on the board. Returns an additive bias of 0 or -inf,
    or None when every cell is live.
    """
    if mask is None:
        return None
    flat = mask.reshape(batch, 1, 1, seq)
    bias = torch.zeros_like(flat, dtype=dtype)
    bias.masked_fill_(flat == 0, float("-inf"))
    return bias


class BoardSelfAttention(nn.Module):
    """Multi-head self-attention over the board, returning a residual.

    `qk_norm` normalizes queries and keys before their dot product. That bounds
    the attention logits, which is what keeps the softmax stable in fp16 -- the
    usual failure is a few large logits saturating to inf.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        height: int,
        width: int,
        *,
        head_dim: int | None = None,
        rope_theta: float = 100.0,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim or channels // heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.RMSNorm(channels, eps=1e-6)
        self.to_q = nn.Linear(channels, heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(channels, heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(channels, heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(heads * self.head_dim, channels, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        if rope_theta <= 2.0 * max(height, width):
            raise ValueError(
                f"rope_theta {rope_theta} is too small for a {height}x{width} board"
            )
        cos, sin = rope_tables(self.head_dim, height, width, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: NCHW -> NCHW residual."""
        batch, channels, height, width = x.shape
        seq = height * width

        tokens = self.norm(x.flatten(2).transpose(1, 2))

        def heads_of(projection: nn.Linear) -> torch.Tensor:
            return projection(tokens).view(batch, seq, self.heads, self.head_dim)

        q = rotate(heads_of(self.to_q), self.rope_cos, self.rope_sin).transpose(1, 2)
        k = rotate(heads_of(self.to_k), self.rope_cos, self.rope_sin).transpose(1, 2)
        v = heads_of(self.to_v).transpose(1, 2)

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        attended = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=board_attention_mask(mask, batch, seq, q.dtype),
            scale=self.scale,
        )

        merged = attended.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.to_out(merged).transpose(1, 2).view(batch, channels, height, width)


class BoardFeedForward(nn.Module):
    """Per-cell feed-forward, returning a residual.

    `depthwise` adds a 3x3 convolution inside the hidden layer. Attention and a
    pointwise FFN are both position-agnostic once RoPE has been applied, so this
    is a cheap way to restore a little local structure.
    """

    def __init__(
        self,
        channels: int,
        hidden: int,
        *,
        swiglu: bool = True,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.swiglu = swiglu
        self.norm = nn.RMSNorm(channels, eps=1e-6)
        self.up = nn.Linear(channels, hidden, bias=False)
        self.gate = nn.Linear(channels, hidden, bias=False) if swiglu else None
        self.activation = nn.SiLU() if swiglu else nn.ReLU()
        self.conv = (
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
            if depthwise
            else None
        )
        self.down = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: NCHW -> NCHW residual."""
        batch, channels, height, width = x.shape

        tokens = self.norm(x.flatten(2).transpose(1, 2))
        hidden = self.activation(self.up(tokens))
        if self.gate is not None:
            hidden = hidden * self.gate(tokens)

        if self.conv is not None:
            spatial = hidden.transpose(1, 2).view(batch, -1, height, width)
            spatial = self.conv(spatial)
            if mask is not None:
                spatial = spatial * mask
            hidden = spatial.flatten(2).transpose(1, 2)

        return self.down(hidden).transpose(1, 2).view(batch, channels, height, width)


class BoardTransformerBlock(nn.Module):
    """Attention then feed-forward, each with its own residual connection."""

    def __init__(
        self,
        channels: int,
        heads: int,
        height: int,
        width: int,
        *,
        ffn_hidden: int | None = None,
        rope_theta: float = 100.0,
        qk_norm: bool = True,
        swiglu: bool = True,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.attention = BoardSelfAttention(
            channels, heads, height, width,
            rope_theta=rope_theta, qk_norm=qk_norm,
        )
        self.feed_forward = BoardFeedForward(
            channels, ffn_hidden or 4 * channels,
            swiglu=swiglu, depthwise=depthwise,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(x, mask)
        return x + self.feed_forward(x, mask)
