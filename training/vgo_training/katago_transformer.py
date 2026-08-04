"""Faithful reproduction of KataGo's transformer blocks.

Ported from `python/katago/train/model_pytorch.py` in lightvector/KataGo
(fetched 2026-08-04). This keeps KataGo's structure, naming, and numerics so the
two can be compared directly; the stripped-down version in
`vgo_training/attention.py` is the one to read for understanding.

Scope: `TransformerAttentionBlock`, `TransformerFFNBlock`, `RMSNormMask`, and
the 2D RoPE helpers. Deliberately omitted, because they are Go-board specific
or serve KataGo's own tooling rather than the architecture:

  * GAB / TAB attention biases (complex equivariant convs over board topology)
  * inline register tokens and grouped-query attention
  * flex-attention block masks, attention-weight capture, logit-penalty hooks
  * brenorm / fixup normalization variants

What is kept is the part that matters: pre-norm RMSNorm, 2D rotary position
embeddings, masked attention over the board as a sequence, and a SwiGLU FFN
with an optional depthwise conv. Both halves return *residuals only* -- the
caller adds them to the trunk, which is KataGo's convention.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def precompute_freqs_cos_sin_2d(
    dim: int, pos_len: int, theta: float = 100.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for 2D RoPE, interleaved real layout.

    Half the head dimension encodes the row, half the column, so a token's
    position on the board is carried by rotation rather than by an added
    embedding. Returns `(pos_len * pos_len, dim)` each.
    """
    assert dim % 4 == 0
    dim_half = dim // 2

    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))

    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing="ij")

    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs

    emb = torch.cat([emb_h, emb_w], dim=-1)
    emb = emb.flatten(0, 1)
    emb = emb.repeat_interleave(2, dim=-1)

    return emb.cos(), emb.sin()


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K by their positions. Both are (B, S, H, D); cos/sin are (S, D)."""

    def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x.unbind(dim=-1)
        return torch.stack([-x1, x0], dim=-1).flatten(-2)

    cos = cos.view(1, xq.shape[1], 1, xq.shape[-1])
    sin = sin.view(1, xq.shape[1], 1, xq.shape[-1])

    xq_out = xq * cos + rotate_every_two(xq) * sin
    xk_out = xk * cos + rotate_every_two(xk) * sin
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNormMask(nn.Module):
    """RMSNorm that ignores off-board positions.

    Two modes, matching KataGo. `spatial=False` normalizes each token over its
    channels, which is ordinary RMSNorm applied in NHWC. `spatial=True`
    normalizes over channels *and* space jointly, so the statistic is taken only
    over live positions -- `mask_sum_hw` is the per-sample count of them. The
    output is re-masked either way so padding stays exactly zero.
    """

    def __init__(self, c_in: int, spatial: bool, cgroup_size: int | None = None) -> None:
        super().__init__()
        self.c_in = c_in
        self.spatial = spatial
        self.cgroup_size = cgroup_size
        self.eps = 1e-6
        if cgroup_size is not None:
            assert spatial, "cgroup_size requires spatial=True"
            assert c_in % cgroup_size == 0
            self.num_groups = c_in // cgroup_size
        if not spatial:
            self.norm = nn.RMSNorm(c_in, eps=self.eps)
        else:
            self.norm = None
            self.gamma = nn.Parameter(torch.ones(c_in))
        self.beta = nn.Parameter(torch.zeros(c_in))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, mask_sum_hw: torch.Tensor
    ) -> torch.Tensor:
        """x: NCHW, mask: N1HW, mask_sum_hw: N111 -> NCHW."""
        if not self.spatial:
            out = x.permute(0, 2, 3, 1)
            out = self.norm(out)
            out = out.permute(0, 3, 1, 2)
            return (out + self.beta.view(1, -1, 1, 1)) * mask

        if self.cgroup_size is not None:
            n, c, h, w = x.shape
            x_grouped = x.view(n, self.num_groups, self.cgroup_size, h, w)
            mask_grouped = mask.view(n, 1, 1, h, w)
            mean_sq = torch.sum(
                x_grouped * x_grouped * mask_grouped, dim=(2, 3, 4), keepdim=True
            ) / (self.cgroup_size * mask_sum_hw.unsqueeze(2) + self.eps)
            out = (x_grouped / torch.sqrt(mean_sq + self.eps)).view(n, c, h, w)
        else:
            mean_sq = torch.sum(x * x * mask, dim=(1, 2, 3), keepdim=True) / (
                self.c_in * mask_sum_hw + self.eps
            )
            out = x / torch.sqrt(mean_sq + self.eps)
        return (out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)) * mask


class TransformerAttentionBlock(nn.Module):
    """Attention half of a transformer block. Returns the residual only.

    Pre-norm RMSNorm, then multi-head self-attention over the board flattened to
    a sequence, with 2D rotary embeddings supplying position. Off-board tokens
    are masked with an additive -inf bias before the softmax.
    """

    def __init__(
        self,
        name: str,
        c_main: int,
        config: dict[str, Any],
        pos_len: int,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.c_main = c_main
        self.use_rope = use_rope

        self.num_heads = config["transformer_heads"]
        self.q_head_dim = config.get(
            "attention_query_head_dim", c_main // self.num_heads
        )
        self.v_head_dim = config.get(
            "attention_value_head_dim", c_main // self.num_heads
        )

        if self.use_rope:
            assert self.q_head_dim % 4 == 0, "2D RoPE needs a head dim divisible by 4"

        self.q_proj = nn.Linear(c_main, self.num_heads * self.q_head_dim, bias=False)
        self.k_proj = nn.Linear(c_main, self.num_heads * self.q_head_dim, bias=False)
        self.v_proj = nn.Linear(c_main, self.num_heads * self.v_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, c_main, bias=False)

        # Normalizing q and k before the dot product bounds the logits, which is
        # what keeps attention numerically safe in fp16.
        self.use_qk_norm = config.get("attention_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.q_head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.q_head_dim, eps=1e-6)

        if self.use_rope:
            self.rope_theta = config.get("rope_theta", 100.0)
            assert self.rope_theta > pos_len * 2.0, (
                f"rope theta {self.rope_theta} is too small for pos_len {pos_len}"
            )
            cos_cached, sin_cached = precompute_freqs_cos_sin_2d(
                self.q_head_dim, pos_len, self.rope_theta
            )
            self.register_buffer("cos_cached", cos_cached, persistent=False)
            self.register_buffer("sin_cached", sin_cached, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        self.norm1 = nn.RMSNorm(c_main, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: NCHW -> NCHW residual. mask: N1HW, or None for a full board."""
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        x_norm = self.norm1(x_in)

        q = self.q_proj(x_norm).view(
            batch_size, seq_len, self.num_heads, self.q_head_dim
        )
        k = self.k_proj(x_norm).view(
            batch_size, seq_len, self.num_heads, self.q_head_dim
        )
        v = self.v_proj(x_norm).view(
            batch_size, seq_len, self.num_heads, self.v_head_dim
        )

        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = None
        if mask is not None:
            mask_flat = mask.reshape(batch_size, 1, 1, seq_len)
            attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_mask.masked_fill_(mask_flat == 0, float("-inf"))

        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0,
            scale=1.0 / math.sqrt(self.q_head_dim),
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_heads * self.v_head_dim
        )
        attn_output = self.out_proj(attn_output)
        return attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)


class TransformerFFNBlock(nn.Module):
    """Feed-forward half of a transformer block. Returns the residual only.

    RMSNorm -> FFN (optionally SwiGLU) -> optional depthwise conv. The depthwise
    conv is KataGo's way of putting a little locality back into an otherwise
    position-agnostic FFN.
    """

    def __init__(
        self,
        name: str,
        c_main: int,
        config: dict[str, Any],
        activation: type[nn.Module] = nn.ReLU,
        use_swiglu: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.c_main = c_main
        self.ffn_dim = config["transformer_ffn_channels"]
        self.use_swiglu = use_swiglu
        self.use_depthwise_conv = config.get("transformer_ffn_depthwise_conv", False)

        self.ffn_linear1 = nn.Linear(c_main, self.ffn_dim, bias=False)
        if self.use_swiglu:
            self.ffn_linear_gate = nn.Linear(c_main, self.ffn_dim, bias=False)
            self.ffn_act = nn.SiLU(inplace=False)
        else:
            self.ffn_act = activation()
        if self.use_depthwise_conv:
            self.ffn_dwconv = nn.Conv2d(
                self.ffn_dim, self.ffn_dim, kernel_size=3, padding=1,
                groups=self.ffn_dim, bias=False,
            )
        self.ffn_linear2 = nn.Linear(self.ffn_dim, c_main, bias=False)
        self.norm = nn.RMSNorm(c_main, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: NCHW -> NCHW residual."""
        batch_size, channels, height, width = x.shape
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        xn = self.norm(x_in)

        x1 = self.ffn_act(self.ffn_linear1(xn))
        if self.use_swiglu:
            x1 = x1 * self.ffn_linear_gate(xn)
        if self.use_depthwise_conv:
            x1_spatial = x1.permute(0, 2, 1).view(
                batch_size, self.ffn_dim, height, width
            )
            if mask is not None:
                x1_spatial = self.ffn_dwconv(x1_spatial) * mask
            else:
                x1_spatial = self.ffn_dwconv(x1_spatial)
            x1 = x1_spatial.view(batch_size, self.ffn_dim, -1).permute(0, 2, 1)
        x1 = self.ffn_linear2(x1)

        return x1.permute(0, 2, 1).view(batch_size, channels, height, width)
