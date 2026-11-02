# Model architecture

The network the search evaluates. Reference-level detail: exact shapes, where
the parameters and the time go, and what each piece is for. For how it fits the
rest of the system see [`OVERVIEW.md`](../OVERVIEW.md) §6; for the decisions and
their evidence see §8 there.

Source: [`training/vgo_training/model.py`](../../training/vgo_training/model.py).
The loop trains `width=64, blocks=16, norm_groups=8`, one attention block per
context stage, on a 256x256 `compact-radius` raster (about 8.2M parameters).
The shapes and timings below were measured on an earlier w96 model at 128x128;
the structure is the same.

## Why this shape

Placement is continuous, so the board is rendered to a raster and the policy is
a spatial map (see [`POLICY_REDESIGN.md`](../history/POLICY_REDESIGN.md)). That leaves two
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

Full forward is 14.5 ms at batch 32. For what actually moves throughput, see
[Inference speed](#inference-speed).

## Components

**`ResidualBlock`** — two 3x3 convolutions and a skip. Deep trunks need the
residual branch controlled or activations compound. `groups` enables GroupNorm
after the convolutions; without it the block is unnormalized. Historical
fixed residual scaling is no longer an option in the current model constructor.

`norms_per_block` defaults to two. The one-norm experiment removes only the
normalization after the second convolution, retaining the first normalization,
skip and activation. This setting applies to convolutional residual blocks;
attention and head normalization remain as before. Checkpoint metadata records
the count, and loading rejects weights incompatible with that architecture.

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

## Inference speed

Measured 2026-11-01 on a 300 W RTX 5070 Ti at 128 input, batch 32, two lanes, with
[`performance/model_timing.py`](../../performance/model_timing.py). That is the card's
ceiling. Shapes are defined in
[`performance/model_variants.py`](../../performance/model_variants.py). The loop's net
reads 21.1k positions/s, repeatable to about 1%, and a trained net reads the same as a
random one. Strength was not measured for any of this.

| Change | Arithmetic | Positions/s |
|---|---|---|
| Detail blocks 4/4/4 to 1/1/2 | 70% | +43% |
| Detail blocks 4/4/4 to 2/2/2 | 78% | +28% |
| `--blocks 8` | 63% | +48% |
| `--blocks 12` | 81% | +21% |
| Detail width 64 to 32 | 64% | +36% |
| No trunk norms (a bound, not a design) | 100% | +29% |
| One norm per block | 100% | +15% |
| Attention blocks to residual blocks | 100% | +7% |
| Tail residual block 128-64-128 bottleneck | 88% | +4% to +8% |
| Remove the DAPPM context module | 100% | +2%, noise |
| PixelShuffle policy, 16 channels | 100% | none |
| Input 96 / 192 / 256 | 56% / 228% / 411% | +34% / -65% / -79% |

Stacked, the shapes reach 34k to 41k. What decides the result:

- **Time follows arithmetic for plain blocks**, so cutting them pays one for one. Each
  residual stage, context stages included, is about 15% of the arithmetic, because doubling
  the channels at half the resolution keeps the cost constant.
- **Cheap convolutions are not cheap here.** Depthwise blocks do 61% of the arithmetic for
  +10%, and inverted and grouped blocks run slower than the plain block.
- **GroupNorm is about 22% of inference time**, so trunk norms cost more than their
  arithmetic suggests.
- **Narrowing from 64 saves little until 32.** With norms off, detail widths 48, 56 and 60 all
  read 28k, against 27k at 64, 38k at 32 and 18k at 96. Multiples of 32 look like the efficient
  widths, though 40, 72 and 80 were not measured with norms off.
- **Some group sizes hit a slow path.** Width 48 reads 15k with 8 groups and 22k with 2, 3, 6
  or 12. Groups of 5, 6, 7, 9 and 13 channels were slow, as was 10 at width 60, while 4, 8, 12,
  16 and 24 were not. The cause is not known. `_group_count` picks a slow case at widths such as
  40, 48, 56 and 72, so try other group counts before trusting a timing at those widths.

In the generator the lanes set the rate, not the card. Each spends about 0.3 ms per batch on
the CPU around `session.run`, so a 39k ceiling gave 31k. A third lane gave +15% on these
nets, against +7% on the current one in the same short runs and nothing over whole games. The
CPU then has to keep up: on two lanes, over 140 s early in games, the same nets stopped at
25.7k on 8 physical cores and 22k on 8 threads of 4 cores. Whole games need more CPU than that.

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
