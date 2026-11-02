#!/usr/bin/env python3
"""The model shapes timed on 2026-11-01, as exportable inference paths.

Timing only: there are no training heads here. `Spec()` is the loop's model, its
module names match model.py, and `--check` proves the two compute the same thing.
Delete this once the shapes worth training are options in model.py.

    training/.venv/bin/python performance/model_variants.py --check
    training/.venv/bin/python performance/model_variants.py --list
    training/.venv/bin/python performance/model_variants.py d222 --output scratch/
    training/.venv/bin/python performance/model_timing.py scratch/d222.onnx

Results are in docs/reference/MODEL_ARCHITECTURE.md.
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from vgo_training.attention import BoardSelfAttention, BoardTransformerBlock
from vgo_training.model import DDRNetPolicyValueNet, ResidualBlock, _DDRContext, _group_count, build_model, value_head, value_utility
from vgo_training.packed_input import PackedInputModel
from vgo_training.packed_states import _LAYOUTS


@dataclass(frozen=True)
class Spec:
    name: str = "base"
    group: str = "base"
    raster: int = 128
    policy: int = 128
    stem: int = 32
    detail: int = 64
    context: int = 128
    deep: int = 256
    entry: int = 4
    stage1: int = 4
    stage2: int = 4
    ctx1: int = 4          # blocks in the stage, attention included
    ctx2: int = 4
    attn1: int = 1
    attn2: int = 1
    ffn: int = 4
    block: str = "res"     # detail branch
    ctx_block: str = "res"
    norm: str = "gn2"      # gn2, gn1, none
    fast_gn: bool = False
    groups: int = 8
    tail: str = "res"      # res, bt, nbt, res64, conv1
    fused: int = 0         # 0 means `context`
    head: int = 0          # 0 means 64
    head_kernel: int = 3
    up: str = "bilinear"   # bilinear, ps, ps2c
    dappm: bool = True
    static: bool = False


def conv(i, o, k=3, s=1, g=1, d=1, bias=False):
    return nn.Conv2d(i, o, k, stride=s, padding=d * (k - 1) // 2, groups=g, dilation=d, bias=bias)


class Res(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class PartialConv(nn.Module):
    """FasterNet: a 3x3 over a quarter of the channels, the rest passed through."""
    def __init__(self, w):
        super().__init__()
        self.part = w // 4
        self.conv = conv(self.part, self.part)

    def forward(self, x):
        head, rest = x[:, : self.part], x[:, self.part :]
        return torch.cat((self.conv(head), rest), dim=1)


class FastGroupNorm(nn.Module):
    """nn.GroupNorm's arithmetic through GlobalAveragePool, not Reshape+InstanceNorm."""
    def __init__(self, groups, width, eps=1e-5):
        super().__init__()
        self.groups, self.members, self.eps = groups, width // groups, eps
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        g, m = self.groups, self.members
        mean_c = nn.functional.adaptive_avg_pool2d(x, 1).reshape(-1, g, m)
        square_c = nn.functional.adaptive_avg_pool2d(x * x, 1).reshape(-1, g, m)
        mean = mean_c.mean(dim=2, keepdim=True)
        variance = square_c.mean(dim=2, keepdim=True) - mean * mean
        scale = (variance + self.eps).sqrt().reciprocal()
        mean = mean.expand(-1, g, m).reshape(-1, g * m, 1, 1)
        scale = scale.expand(-1, g, m).reshape(-1, g * m, 1, 1)
        return (x - mean) * scale * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class Factory:
    def __init__(self, spec: Spec):
        self.spec = spec

    def n(self, w):
        if self.spec.norm == "none":
            return nn.Identity()
        cls = FastGroupNorm if self.spec.fast_gn else nn.GroupNorm
        return cls(_group_count(w, self.spec.groups), w)

    def block(self, kind: str, w: int) -> nn.Module:
        s, n, r = self.spec, self.n, nn.ReLU
        if kind == "res":
            if s.fast_gn:
                return Res(conv(w, w), n(w), r(), conv(w, w), n(w))
            return ResidualBlock(w, groups=None if s.norm == "none" else s.groups, norms_per_block=1 if s.norm in ("gn1", "half") else 2)
        if kind == "res1":
            return Res(conv(w, w), n(w))
        if kind == "dil":
            return Res(conv(w, w), n(w), r(), conv(w, w, d=2), n(w))
        if kind == "asym":
            return Res(conv_a(w), conv_b(w), n(w), r(), conv_a(w), conv_b(w), n(w))
        if kind == "nbt1":
            m = w // 2
            return Res(conv(w, m, 1), n(m), r(), conv(m, m), n(m), r(), conv(m, m), n(m), r(), conv(m, w, 1), n(w))
        if kind == "nbt2":
            m = w // 2
            return Res(conv(w, m, 1), n(m), r(), self.block("res", m), self.block("res", m), conv(m, w, 1), n(w))
        if kind == "ds":
            return Res(conv(w, w, g=w), conv(w, w, 1), n(w), r(), conv(w, w, g=w), conv(w, w, 1), n(w))
        if kind == "ds5":
            return Res(conv(w, w, 5, g=w), conv(w, w, 1), n(w), r(), conv(w, w, 5, g=w), conv(w, w, 1), n(w))
        if kind == "inv":
            e = 3 * w
            return Res(conv(w, e, 1), n(e), r(), conv(e, e, g=e), r(), conv(e, w, 1), n(w))
        if kind == "cnx":
            return Res(conv(w, w, 7, g=w), n(w), conv(w, 2 * w, 1), r(), conv(2 * w, w, 1))
        if kind == "pconv":
            return Res(PartialConv(w), conv(w, 2 * w, 1), n(2 * w), r(), conv(2 * w, w, 1))
        if kind == "g4":
            return Res(conv(w, w, g=4), n(w), r(), conv(w, w, 1), n(w), r(), conv(w, w, g=4), n(w), r(), conv(w, w, 1), n(w))
        raise ValueError(kind)


def conv_a(w):
    return nn.Conv2d(w, w, (1, 3), padding=(0, 1), bias=False)


def conv_b(w):
    return nn.Conv2d(w, w, (3, 1), padding=(1, 0), bias=False)


class Down(nn.Module):
    def __init__(self, f, i, o, blocks, attention, board, ffn):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.ReLU())
        self.body = nn.Sequential(*[f.block(f.spec.ctx_block, o) for _ in range(blocks - attention)])
        self.attention = nn.ModuleList(
            BoardTransformerBlock(o, 8, board, board, rope_theta=max(100.0, 4.0 * board), ffn_hidden=ffn * o)
            for _ in range(attention)
        )

    def forward(self, x):
        x = self.body(self.reduce(x))
        for block in self.attention:
            x = block(x)
        return x


class Net(nn.Module):
    def __init__(self, s: Spec):
        super().__init__()
        self.spec = s
        f = Factory(s)
        trunk = s.raster // 4
        board1, board2 = (trunk + 1) // 2, ((trunk + 1) // 2 + 1) // 2
        self.stem = nn.Sequential(
            nn.Conv2d(7, s.stem, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(s.stem, s.detail, 3, stride=2, padding=1), nn.ReLU(),
        )
        def stack(count):
            if s.norm == "half":
                plain = Factory(replace(s, norm="none"))
                return nn.Sequential(*[(f if i % 2 else plain).block(s.block, s.detail) for i in range(count)])
            return nn.Sequential(*[f.block(s.block, s.detail) for _ in range(count)])
        self.detail_entry = stack(s.entry)
        self.detail_stage1 = stack(s.stage1)
        self.context_stage1 = Down(f, s.detail, s.context, s.ctx1, s.attn1, board1, s.ffn)
        self.context_to_detail1 = nn.Conv2d(s.context, s.detail, 1)
        self.detail_to_context1 = nn.Conv2d(s.detail, s.context, 3, stride=2, padding=1)
        self.detail_stage2 = stack(s.stage2)
        self.context_stage2 = Down(f, s.context, s.deep, s.ctx2, s.attn2, board2, s.ffn)
        self.context_to_detail2 = nn.Conv2d(s.deep, s.detail, 1)
        self.detail_to_context2 = nn.Sequential(
            nn.Conv2d(s.detail, s.context, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(s.context, s.deep, 3, stride=2, padding=1),
        )
        self.context = _DDRContext(s.deep, 32, s.context) if s.dappm else nn.Conv2d(s.deep, s.context, 1)
        fused = s.fused or s.context
        self.semantic_to_fused = nn.Identity() if fused == s.context else nn.Conv2d(s.context, fused, 1)
        self.detail_tail = self._tail(f, s, fused)
        head = s.head or 64
        self.policy_features = nn.Sequential(nn.Conv2d(fused, head, s.head_kernel, padding=s.head_kernel // 2), nn.ReLU())
        self.up_ratio = s.policy // trunk
        if s.up == "bilinear":
            self.policy_map = nn.Conv2d(head, 1, 1)
        elif s.up == "ps":
            assert self.up_ratio * trunk == s.policy
            self.policy_map = nn.Sequential(nn.Conv2d(head, self.up_ratio**2, 1), nn.PixelShuffle(self.up_ratio))
        elif s.up == "ps2c":
            assert self.up_ratio == 4 and head % 4 == 0
            self.policy_map = nn.Sequential(
                nn.PixelShuffle(2), nn.Conv2d(head // 4, head // 4, 3, padding=1), nn.ReLU(),
                nn.Conv2d(head // 4, 4, 1), nn.PixelShuffle(2),
            )
        else:
            raise ValueError(s.up)
        self.pass_head = nn.Linear(s.context, 1)
        self.value_head = value_head(s.context)

    @staticmethod
    def _tail(f, s, fused):
        w, r = s.detail, nn.ReLU
        if s.tail == "res":
            return nn.Sequential(nn.Conv2d(w, fused, 1), r(), f.block("res", fused))
        if s.tail == "bt":
            m = fused // 2
            return nn.Sequential(nn.Conv2d(w, fused, 1), r(), Res(conv(fused, m, 1), f.n(m), r(), conv(m, m), f.n(m), r(), conv(m, fused, 1), f.n(fused)))
        if s.tail == "nbt":
            return nn.Sequential(nn.Conv2d(w, fused, 1), r(), f.block("nbt1", fused))
        if s.tail == "res64":
            return nn.Sequential(f.block("res", w), nn.Conv2d(w, fused, 1))
        if s.tail == "conv1":
            return nn.Conv2d(w, fused, 1)
        raise ValueError(s.tail)

    @staticmethod
    def _resize(x, size):
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, states):
        detail = self.detail_entry(self.stem(states))
        detail_before = self.detail_stage1(detail)
        context_before = self.context_stage1(detail)
        detail = torch.relu(detail_before + self._resize(self.context_to_detail1(context_before), detail_before.shape[-2:]))
        context = torch.relu(context_before + self.detail_to_context1(detail_before))
        detail_before = self.detail_stage2(detail)
        context_before = self.context_stage2(context)
        detail = torch.relu(detail_before + self._resize(self.context_to_detail2(context_before), detail_before.shape[-2:]))
        context = torch.relu(context_before + self.detail_to_context2(detail_before))
        semantic = self.context(context)
        fused = torch.relu(self.detail_tail(detail) + self._resize(self.semantic_to_fused(semantic), detail.shape[-2:]))
        placement = self.policy_map(self.policy_features(fused))
        if self.spec.up == "bilinear":
            placement = DDRNetPolicyValueNet._resize_policy(placement, (self.spec.policy, self.spec.policy))
        pooled = semantic.mean(dim=(-2, -1))
        return torch.cat((placement.flatten(1), self.pass_head(pooled)), dim=1), value_utility(self.value_head(pooled))


def macs(net: Net) -> dict:
    """Multiplies by top-level module, with attention's dot products counted."""
    totals: dict[str, float] = {}
    handles = []
    def add(name, value):
        totals[name] = totals.get(name, 0.0) + value
    for top, child in net.named_children():
        for module in child.modules():
            if isinstance(module, nn.Conv2d):
                def hook(m, _i, o, top=top):
                    k = m.kernel_size[0] * m.kernel_size[1]
                    add(top, o.shape[-1] * o.shape[-2] * k * m.in_channels * m.out_channels / m.groups)
            elif isinstance(module, nn.Linear):
                def hook(m, i, _o, top=top):
                    tokens = i[0].numel() / i[0].shape[-1]
                    add(top, tokens * m.in_features * m.out_features)
            elif isinstance(module, BoardSelfAttention):
                def hook(m, i, _o, top=top):
                    seq = i[0].shape[-1] * i[0].shape[-2]
                    add(top, 2 * m.heads * seq * seq * m.head_dim)
            else:
                continue
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, 7, net.spec.raster, net.spec.raster))
    for h in handles:
        h.remove()
    return totals


def specs() -> list[Spec]:
    v = lambda name, group, **k: Spec(name=name, group=group, **k)
    d = lambda a, b, c: dict(entry=a, stage1=b, stage2=c)
    a0 = dict(attn1=0, attn2=0)
    return [
        Spec(),
        # Existing knobs, for reference
        v("raster96", "reference", raster=96),
        v("raster192", "reference", raster=192),
        v("raster256", "reference", raster=256),
        v("w48", "reference", stem=24, detail=48, context=96, deep=192),
        v("w56", "reference", stem=28, detail=56, context=128, deep=224),
        v("attn0", "reference", attn1=0, attn2=0),
        v("attn_deep_only", "reference", attn1=0),
        v("attn_none_shorter", "reference", attn1=0, attn2=0, ctx1=3, ctx2=3),
        v("attn_ffn2", "reference", ffn=2),
        v("norm1", "reference", norm="gn1"),
        v("norm0_upper_bound", "reference", norm="none"),
        v("no_dappm", "reference", dappm=False),
        # Tail
        v("tail_bt", "tail", tail="bt"),
        v("tail_nbt", "tail", tail="nbt"),
        v("tail_res64", "tail", tail="res64"),
        v("tail_conv1", "tail", tail="conv1"),
        v("fused64", "tail", fused=64),
        v("fused64_head32", "tail", fused=64, head=32),
        v("fused64_nobt_conv1", "tail", fused=64, tail="conv1"),
        # Policy upsampling
        v("up_ps", "upsample", up="ps"),
        v("up_ps2c", "upsample", up="ps2c"),
        v("head_1x1", "upsample", head_kernel=1),
        # Block kinds in the detail branch
        v("b_res1", "block", block="res1"),
        v("b_nbt1", "block", block="nbt1"),
        v("b_nbt2", "block", block="nbt2"),
        v("b_ds", "block", block="ds"),
        v("b_ds5", "block", block="ds5"),
        v("b_inv", "block", block="inv"),
        v("b_cnx", "block", block="cnx"),
        v("b_pconv", "block", block="pconv"),
        v("b_g4", "block", block="g4"),
        v("b_asym", "block", block="asym"),
        v("b_dil", "block", block="dil"),
        # ... and in both branches
        v("all_nbt1", "block", block="nbt1", ctx_block="nbt1"),
        v("all_nbt2", "block", block="nbt2", ctx_block="nbt2"),
        v("all_ds", "block", block="ds", ctx_block="ds"),
        # Depth and width layout
        v("d333", "layout", entry=3, stage1=3, stage2=3),
        v("d222", "layout", entry=2, stage1=2, stage2=2),
        v("d122", "layout", entry=1, stage1=2, stage2=2),
        v("d112", "layout", entry=1, stage1=1, stage2=2),
        v("d222_c66", "layout", entry=2, stage1=2, stage2=2, ctx1=6, ctx2=6),
        v("ctx96_192", "layout", context=96, deep=192, fused=128),
        v("detail48", "layout", detail=48),
        v("detail56", "layout", detail=56),
        v("detail52", "layout", detail=52),
        v("detail60", "layout", detail=60),
        # Serving only: same weights, no retraining
        v("serve_fast_gn", "serving", fast_gn=True),
        v("serve_static", "serving", static=True),
        v("serve_min32", "serving"),
        # Is the width penalty the channel count, or GroupNorm at that count?
        v("wd32", "width", detail=32),
        v("wd40", "width", detail=40),
        v("wd72", "width", detail=72),
        v("wd80", "width", detail=80),
        v("wd96", "width", detail=96),
        v("wd128", "width", detail=128),
        v("wd32_n0", "width", detail=32, norm="none"),
        v("wd48_n0", "width", detail=48, norm="none"),
        v("wd56_n0", "width", detail=56, norm="none"),
        v("wd60_n0", "width", detail=60, norm="none"),
        v("wd96_n0", "width", detail=96, norm="none"),
        v("wd48_g2", "width", detail=48, groups=2),
        v("wd48_g3", "width", detail=48, groups=3),
        v("wd48_g6", "width", detail=48, groups=6),
        v("wd48_g12", "width", detail=48, groups=12),
        v("wd64_g4", "width", groups=4),
        # Winners stacked
        v("norm_half", "combo", norm="half"),
        v("c_d222_bt", "combo", tail="bt", **d(2, 2, 2)),
        v("c_d222_bt_a0", "combo", tail="bt", **a0, **d(2, 2, 2)),
        v("c_d222_bt_short", "combo", tail="bt", **a0, ctx1=3, ctx2=3, **d(2, 2, 2)),
        v("c_d222_bt_a0_n1", "combo", tail="bt", norm="gn1", **a0, **d(2, 2, 2)),
        v("c_d333_bt_a0_n1", "combo", tail="bt", norm="gn1", **a0, **d(3, 3, 3)),
        v("c_d122_bt_a0_n1", "combo", tail="bt", norm="gn1", **a0, **d(1, 2, 2)),
        v("c_d112_bt_a0_n1", "combo", tail="bt", norm="gn1", **a0, **d(1, 1, 2)),
        v("c_d222_bt_a0_n1_ps", "combo", tail="bt", norm="gn1", up="ps", **a0, **d(2, 2, 2)),
        v("c_d222_dil_bt_a0_n1", "combo", tail="bt", norm="gn1", block="dil", **a0, **d(2, 2, 2)),
        v("c_d222_c55_bt_a0_n1", "combo", tail="bt", norm="gn1", ctx1=5, ctx2=5, **a0, **d(2, 2, 2)),
        v("c_res1_d666_bt_a0_n1", "combo", tail="bt", norm="gn1", block="res1", **a0, **d(6, 6, 6)),
        v("c_d222_bt_a0_n0", "combo", tail="bt", norm="none", **a0, **d(2, 2, 2)),
        v("c_d112_bt_a0_n0", "combo", tail="bt", norm="none", **a0, **d(1, 1, 2)),
        v("c_w32_bt_a0_n1", "combo", detail=32, tail="bt", norm="gn1", **a0),
        v("c_w32_d666_bt_a0_n1", "combo", detail=32, tail="bt", norm="gn1", **a0, **d(6, 6, 6)),
        v("blocks12", "knobs", ctx1=3, ctx2=3, **d(3, 3, 3)),
        v("blocks8", "knobs", ctx1=2, ctx2=2, **d(2, 2, 2)),
        v("w48_g6", "knobs", stem=24, detail=48, context=96, deep=192, groups=6),
        v("w32_all", "knobs", stem=16, detail=32, context=64, deep=128),
        v("c_w48g6_bt_a0_n1", "finalist", stem=24, detail=48, context=96, deep=192, groups=6, tail="bt", norm="gn1", **a0),
        v("c_w32_ctx96_bt_a0_n1", "finalist", detail=32, context=96, deep=192, fused=128, tail="bt", norm="gn1", **a0),
        v("c_d222_short_n1_ps", "finalist", tail="bt", norm="gn1", up="ps", ctx1=3, ctx2=3, **a0, **d(2, 2, 2)),
        v("c_d122_short_n1_ps", "finalist", tail="bt", norm="gn1", up="ps", ctx1=3, ctx2=3, **a0, **d(1, 2, 2)),
        v("c_d222_c22_n1_ps", "finalist", tail="bt", norm="gn1", up="ps", ctx1=2, ctx2=2, **a0, **d(2, 2, 2)),
    ]


def check() -> None:
    """The baseline must be model.py: same weights in, same numbers out."""
    torch.manual_seed(3)
    real = build_model(
        channels=7, width=64, blocks=16, norm_groups=8, context_attention_blocks=1,
        attention_heads=8, raster_resolution=128,
    ).eval()
    mine = Net(Spec()).eval()
    missing, unexpected = mine.load_state_dict(real.state_dict(), strict=False)
    training_only = lambda key: "_normed" in key or "_norm." in key or key.startswith("ownership_")
    assert not missing and all(training_only(key) for key in unexpected), (missing, unexpected)
    states = torch.randn(4, 7, 128, 128)
    with torch.no_grad():
        ours, theirs = mine(states), real(states)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    print("the baseline computes exactly what model.py computes")


def export(spec: Spec, path: Path) -> None:
    torch.manual_seed(1)
    layout, r = _LAYOUTS[7], spec.raster
    wrapped = PackedInputModel(
        Net(spec).eval(), binary=layout.binary, continuous=layout.continuous,
        scalar=layout.scalar, height=r, width=r,
    ).eval()
    n = 32 if spec.static else 2
    example = (
        torch.zeros((n, 3, -(-r * r // 8)), dtype=torch.uint8),
        torch.zeros((n, 1, r, r), dtype=torch.float16),
        torch.zeros((n, 3), dtype=torch.float16),
    )
    batch = torch.export.Dim("batch", min=1, max=32)
    torch.onnx.export(
        wrapped, example, input_names=["bits", "dense", "scalars"],
        output_names=["policy_logits", "values"],
        dynamic_shapes=None if spec.static else ({0: batch}, {0: batch}, {0: batch}),
        dynamo=True, optimize=True, opset_version=20, external_data=False, verbose=False,
    ).save(path, external_data=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check()
    shapes = {spec.name: spec for spec in specs()}
    if arguments.list:
        for spec in shapes.values():
            net = Net(spec).eval()
            print(f"{spec.name:24s} {spec.group:10s} {sum(p.numel() for p in net.parameters()) / 1e6:6.2f}M {sum(macs(net).values()) / 1e9:6.3f}G")
    arguments.output.mkdir(parents=True, exist_ok=True)
    for name in arguments.names:
        export(shapes[name], arguments.output / f"{name}.onnx")


if __name__ == "__main__":
    main()
