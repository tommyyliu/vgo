# Does a human-visible board teach as well as the semantic raster?

## The question

The network currently sees ten engineered channels. A human sees a picture. If
the picture is sufficient, the engineered channels are doing work the network
could do itself — and we are paying for them in memory, bandwidth, and the risk
that a hand-designed feature encodes an assumption the search later has to fight.

Measured on `ddrnet-pipe/replay/shard-000004` (400 samples):

| channel | distinct values | nonzero | human-visible? |
|---|---|---|---|
| `current_stones` | 2 | 3.7% | yes — the stones |
| `opponent_stones` | 2 | 4.1% | yes — the stones |
| `current_voronoi` | 3 | 41.4% | yes — territory shading |
| `opponent_voronoi` | 3 | 51.9% | yes — territory shading |
| `current_distance` | 276 | 100% | **no** — computed distance field |
| `opponent_distance` | 276 | 100% | **no** — computed distance field |
| `voronoi_ridge` | 100648 | 23.7% | **no** — continuous ridge field |
| `legal_clearance` | 154 | 100% | **no** — signed legality map |
| `radius` | **1** | 100% | constant; see below |
| `previous_pass` | 2 | 10.2% | no — game state, not pixels |

Four channels carry what a person could see. Four are derived fields no player
has access to. `radius` held exactly one value (0.11) across every sample — it is
a per-run constant occupying 2.0 GB of the training working set and 10% of every
host-to-device transfer while carrying no information within a run.

So the real question is narrower than "RGB or not": **can a convnet recover
distance, ridge, and legality structure from stone positions alone?** Convolution
over a 128x128 raster has the receptive field to do it in principle. Whether it
does so at this depth, at this sample budget, is empirical.

## Two experiments, in order

### 1. Ablation (do this first)

Zero out `current_distance`, `opponent_distance`, `voronoi_ridge`, and
`legal_clearance` at load time; keep everything else, including the format.
Train against an existing shard window and compare to the unablated baseline on
the same data.

This needs no format change, no Rust change, and no new generation. It isolates
exactly the question — are the derived channels load-bearing — and costs one
training run.

Add a second arm dropping only `radius`, which should be free and is worth
confirming.

**Read the result as:** if `policy_kl` and `value_mae` land within noise of the
baseline, the derived channels are redundant and the RGB experiment is worth its
cost. If value loss degrades materially, they are carrying real signal, and the
interesting follow-up is *which* one.

Caveat: ablating to zero is not the same as never having trained with the
channel. A model trained from scratch without it may learn a different internal
representation and do better than an ablated-at-load model suggests. Treat a
negative ablation result as a lower bound, not a verdict.

### 2. RGB raster

Three channels at 128x128, rendered as a player sees the board: stones drawn as
filled discs of the current radius, territory as the Voronoi fill, everything
else omitted.

## Rendering: port, do not drive JavaScript

`reference/js-reference/voronoi_go.html` is the cleanest statement of the visual
language, and it is worth reading before writing the rasterizer — the colour
choices and the territory fill are already resolved there.

It is not, however, reusable as a renderer. `renderBoard()` (line 328) and
`draw()` (line 400) build **SVG strings and mutate DOM nodes** — `setAttribute`
on `$ghost`, `$snap`, hover state, `requestAnimationFrame` scheduling. There is
no `render(position) -> pixels` seam. Extracting one means either running a
headless browser inside the generation loop (a per-position browser round trip
against a Rust actor pool that currently issues ~600k inferences per shard — not
viable) or rewriting the drawing anyway.

The geometry it draws from is already in Rust: `crates/vgo-core/src/voronoi.rs`
computes the cells, and `crates/vgo-raster` already rasterizes them for the
`*_voronoi` channels. The port is therefore small — it is a recolouring of
geometry we already have, not new geometry.

**Use the JS as the visual specification, implement in `vgo-raster`.** Match its
palette so the RGB raster and the human view are the same picture.

## Implementation sketch

The blocker is that `CHANNEL_COUNT` is a compile-time constant
(`crates/vgo-raster/src/lib.rs:5`) consumed across `vgo-inference`
(`onnx.rs:144, 198, 263, 331`), including in the ONNX shape profiles and the
model metadata contract. A second channel layout cannot simply be a different
number in the same constant.

Least invasive path:

1. Add `RasterKind { Semantic, Rgb }` to `RasterConfig`, with a `channels()`
   method returning 10 or 3. Keep `CHANNEL_COUNT` as the semantic value so
   nothing existing changes meaning.
2. Replace `CHANNEL_COUNT` uses in `vgo-inference` with `config.raster.channels()`.
   The ONNX profile strings and the `vgo.channels` metadata property then follow
   the config rather than a constant, which is also what makes the exported model
   self-describing.
3. Add `rasterize_rgb_into` beside `rasterize_into`. Same geometry, three
   channels: stone discs and Voronoi fill, using the JS palette.
4. Thread `--raster-kind {semantic,rgb}` through `generate-demo`, `arena`, and
   the pipeline config. It belongs in the run identity, not the operational set —
   two runs with different raster kinds are not comparable and must not resume
   into each other.
5. `build_model` takes `channels` already; the training side needs only the
   dataset header's channel count, which it reads today.

## What to measure

Run at the settings `ddrnet-pipe` is using now — w96/b16, 6144 samples/shard,
512 simulations, replay window 5 — so the comparison is against a known curve.

- **Elo against the semantic baseline's pool.** The only measure that matters.
  Cross-representation arena is valid: both play the same game.
- `value_mae` and `policy_kl` per update, against the baseline at equal update
  count.
- `ply0_candidate_jaccard`. If the RGB model's search collapses, the
  representation is not supporting the coarse-to-fine sampler.

Efficiency, secondary but real: `states` falls from 20.13 GB to ~6 GB of the
training working set, and the per-batch host-to-device transfer from 41.9 MB to
12.6 MB. At batch 64 the whole transfer is currently 1.5 ms against ~17.7 ms of
compute, so expect no measurable speedup from that alone — the win is memory
headroom on a box that has OOM'd at ~24 GB.

## Order of work

1. Ablation arms — no code beyond a load-time mask. Cheapest evidence.
2. Drop `radius` regardless of the outcome; a constant channel is pure cost.
3. RGB raster only if the ablation says the derived channels are not carrying
   the value signal.

Do not start any of this until the current `ddrnet-pipe` run finishes. It is the
matched baseline every arm here is measured against, and it is also the only
thing keeping the box under its memory ceiling.
