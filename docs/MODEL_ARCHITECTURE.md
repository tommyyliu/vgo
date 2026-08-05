# Model architecture

The network the search evaluates. Reference-level detail: exact shapes, where
the parameters and the time go, and what each piece is for. For how it fits the
rest of the system see [`OVERVIEW.md`](OVERVIEW.md) §6; for the decisions and
their evidence see §8 there.

Source: [`training/vgo_training/model.py`](../training/vgo_training/model.py).
Production is `architecture=ddrnet, width=96, blocks=16, norm_groups=8` at a
128x128 compact raster.

## Why this shape

Placement is continuous, so the board is rendered to a raster and the policy is
a spatial map (see [`POLICY_REDESIGN.md`](POLICY_REDESIGN.md)). That leaves two
demands in tension:

- **Placement needs resolution.** A move is a real coordinate, and the policy
  grid has to resolve moves that differ by less than a stone radius.
- **Evaluation needs context.** Who owns a region, whether a group is settled,
  and what the score is are all global properties of the whole board.

A single trunk has to choose. DDRNet runs both: a high-resolution *detail*
branch and a low-resolution *context* branch, exchanging information twice.

Reference: Hong et al., "Deep Dual-resolution Networks for Real-time and
Accurate Semantic Segmentation of Road Scenes", arXiv:2101.06085. The original
targets 1024x2048 road scenes with detail at stride 8 and context to stride 64;
that is far too coarse for a ~128px game raster, so both branches sit one octave
higher here.

## Shapes

Measured at batch 2, 128x128x5 input, `stem_stride=4`:

```
input             (5, 128, 128)
  stem            (96,  32,  32)   128 -> 32, two strided convs
  detail_entry    (96,  32,  32)
                          |
  detail_stage1   (96,  32,  32)   context_stage1  (192, 16, 16)
                          +---- bilateral fusion ----+
  detail_stage2   (96,  32,  32)   context_stage2  (384,  8,  8)
                          +---- bilateral fusion ----+
                          |        context         (192,  8,  8)   DAPPM
  detail_tail    (192,  32,  32)
                          |
  fused          (192,  32,  32)   detail_tail + upsampled context
```

`blocks` sets the depth of each stage in groups of four: `stage_blocks =
max(1, (blocks + 3) // 4)`, so `blocks=16` gives four residual blocks per stage
and 21 in total.

Channel widths derive from `width`: stem `width/2`, detail `width`, context
`2*width`, deep context `4*width`.

## Where the parameters are

18.823M total at w96/b16. The exported graph is ~18.29M — the `_normed` twin
heads (0.37M) and ownership (0.17M) are training-only.

| module | params | note |
|---|---|---|
| `context_stage2` | 11.287M | 60% of the model: 384 channels |
| `context_stage1` | 2.823M | |
| `detail_to_context2` | 0.830M | fusion projection |
| `detail_tail` | 0.683M | |
| `detail_entry` / `stage1` / `stage2` | 0.665M each | 96 channels each |
| `context` (DAPPM) | 0.199M | |

**Parameters concentrate in context; time concentrates in detail.** The context
branch holds 75% of the weights but runs at 16x16 and 8x8, while the detail
branch is only 11% of the parameters and runs at 32x32 — four times the
positions. Measured forward cost per stage at batch 32:

| stage | resolution | ms |
|---|---|---|
| stem + detail_entry | 32x32 | 2.18 |
| detail_stage1 | 32x32 | 1.90 |
| detail_stage2 | 32x32 | 1.92 |
| detail_tail | 32x32 | 1.37 |
| context_stage1 | 16x16 | 1.48 |
| context_stage2 | 8x8 | 1.28 |

Full forward is 14.5 ms at batch 32. Anything aiming to cut inference cost has
to come out of the detail branch or the block count; the policy resize is
0.017 ms and the stem alone 0.29 ms, so neither is worth touching.

## Components

**`ResidualBlock`** — two 3x3 convolutions and a skip. Deep trunks need the
residual branch held near unit variance or activations compound, and there are
two mutually exclusive ways to do it: `groups` puts a GroupNorm after each
convolution, and `residual_scale` applies KataGo's fixed `1/sqrt(n)` constant
where a norm would sit. Prefer `groups`; the constant cannot respond to weight
drift and is kept for older checkpoints.

**`_Down`** — stride-2 conv, then residual blocks at the smaller scale. Both
context stages are one of these.

**Bilateral fusion** — after each stage, detail and context exchange. Context
projects to detail channels through a 1x1 and upsamples; detail projects to
context channels through strided convs. Both directions read the *pre-fusion*
values, which is what distinguishes DDRNet from a one-way decoder.

**`_DDRContext`** — a compact DAPPM. The original's five pooling scales assume
a megapixel scene; an 8x8 semantic map leaves native, half and global as the
scales carrying distinct information. Each coarser scale is added to and
processed from the preceding one, then all three are compressed together.

## Heads

Everything reads one of two feature maps: `semantic` (192, 8, 8) from the
context module, and `fused` (192, 32, 32), the detail tail plus upsampled
context.

| head | reads | output |
|---|---|---|
| policy | `fused` | placement map resized to the policy grid, plus one pass logit |
| value | pooled `semantic` | two logits, P(mover wins) and P(mover loses) |
| ownership | `fused` | per-cell owner, training only |

**Each head exists twice.** A plain set reads raw trunk features; a `_normed`
set reads batch-normalized ones and carries 80% of the loss
(`NORMED_HEAD_WEIGHT`). Without a norm in front of *some* head nothing penalizes
weight magnitude, and the trunk inflates until activations overflow fp16 —
measured at 68824 against fp16's 65504 on `ddrnet-fp32` update 2. Keeping an
unnormalized twin for inference is what keeps BatchNorm running statistics out
of the exported graph, so there is no train/serve divergence.

**Value is categorical**, and ownership uses the same idea with the redundant
logit dropped. Both replaced tanh+MSE, whose `(1 - v^2)` gradient factor
vanished exactly where the model was confidently wrong. See `OVERVIEW.md` §8.

## Contract

Training returns six tensors: policy, value logits, the two `_normed`
equivalents, and both ownership maps. Inference returns two — policy logits and
the collapsed scalar utility — which is exactly what `vgo-inference` reads and
what the ONNX graph exports. `model.py`'s `forward` branches on
`self.training`, so ownership and the normalized twins never reach export.

## Experimental options

Off by default; each is byte-identical to the default when disabled.

| option | effect | status |
|---|---|---|
| `context_attention_blocks` | replaces trailing residual blocks in each context stage with transformer blocks | measured: faster early, converges to a tie |
| `global_pooling` | heads read mean, size-scaled mean and max instead of the mean alone | branch `experiment/global-pooling` |
| `optimistic_policy` | second policy head weighted toward better-than-predicted outcomes, for search to read | branch `experiment/optimistic-policy` |

`context_attention_blocks` carries a constraint the rest of the net does not:
rotary position tables are built per board size, so a model with attention is
fixed to the raster resolution it was constructed for and `raster_resolution`
becomes required.

## What is known about changing it

- **GroupNorm at 8 groups is right, for a reason unrelated to why it was
  chosen.** Grouping buys nothing representationally here — across 42 norm sites
  the spread *across* group means is 1.24x against 3.35x *within* groups — and
  LayerNorm trains equivalently. But 1 group is ~43% slower on TensorRT while
  4/8/12 are within 0.4% of each other.
- **One norm per block instead of two** is +11% inference and -10% training wall
  time with no measured drift: peak validation activation 1.04x, fp16 headroom
  747x. Not adopted because strength was never measured in an arena.
- **Width is the large lever and is not free.** w96 = 14.43 ms / 18.82M,
  w64 = 7.66 ms / 8.37M. A capacity change, not a serving change.
- **The policy branch runs twice per training step.** `forward` calls its head
  helper once for policy and value, then again with the same unnormalized
  modules purely to extract ownership. ~0.9% of a training step; a clarity fix,
  not a performance one.
