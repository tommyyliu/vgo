# A client-side bot

Preliminary plan, 2026-08-16. Nothing here is built yet.

## What this is for

The game has a community site run by the person who invented the game. A bot
that runs **in the player's browser** gives that site a practice opponent and
smooths over matchmaking when nobody else is online, without anyone operating a
GPU. That last part is the constraint that shapes everything below: a hosted
engine is a scaling and cost liability even at today's volume, and it fails at
exactly the moment the game gets popular. A static bundle on a CDN does not.

`vgo-serve-move` already exists and would give full-strength play in days rather
than weeks. It is the right answer to a different question — "let people play the
strong engine now" — and it is worth keeping as an optional backend. This
document is about the always-available one.

## The mistake to avoid

The obvious plan is "export the ONNX, run it under onnxruntime-web with the
WebGPU backend, done". That optimises the cheap half. Generation profiling puts
**inference at 2.3% of lane time against 23% for CPU rasterization**. Moving the
network to the GPU and leaving the raster in JavaScript replaces a fast Rust
rasterizer with a slow JS one and keeps the actual bottleneck.

The second trap is the transfer. One position is `5 x 128 x 128` f32 =
**327 KB**. Rasterizing on the CPU and uploading a tensor per evaluation is
~130 MB of upload for a 400-simulation move. Natively this already shows up as
inference being "host-blocked by a 21 MB staging memcpy".

So the design goal is: **the raster is produced on the GPU and never leaves it.**

## Shape of the thing

    ┌─ WASM (compiled from the existing Rust) ────────────┐
    │  vgo-core    rules, legality, settlement            │
    │  vgo-search  MCTS, coarse/fine selection            │
    │  vgo-raster  settled-region contours only           │
    └──────────────┬──────────────────────────────────────┘
                   │ stone list (x, y, colour), ~28 median
                   v
    ┌─ WGSL compute shader ───────────────────────────────┐
    │  channels 0,1,2,4 straight from the stone buffer    │
    │  channel 3 filled from uploaded contours            │
    │  writes the [5,128,128] f32 tensor into a GPU buffer│
    └──────────────┬──────────────────────────────────────┘
                   │ ort.Tensor.fromGpuBuffer  (zero copy)
                   v
    ┌─ onnxruntime-web, WebGPU EP ────────────────────────┐
    │  policy_logits [16385], values [1]                  │
    └─────────────────────────────────────────────────────┘

### Why compile the Rust instead of rewriting in JS

There are already two implementations of the rules, and `game.rs` carries the
comment *"Must match `place` in reference/src/engine/game.js"*. They have
diverged in practice: the even-trade bug — a placement capturing exactly one
stone read as a pass, ending live games — had to be fixed in both. A third
implementation in JavaScript would make that worse, and search and rules are the
two things where a subtle divergence is least visible and most damaging. WASM
keeps one source of truth.

### The channel split, and why it is favourable

The compact layout (`COMPACT_CHANNELS`, `crates/vgo-raster/src/lib.rs`) is five
planes:

| # | channel | how it is computed | GPU? |
|--:|---|---|---|
| 0 | `current_stones` | pixel centre within `r` of a current stone | trivial |
| 1 | `opponent_stones` | same, opponent | trivial |
| 2 | `voronoi_ridge` | `clamp(1 - (d2-d1)/r, 0, 1)`, nearest two over **all** stones | trivial |
| 3 | `settled` | per-stone region contour, then scanline fill | partly |
| 4 | `komi` | constant plane, mover-relative | trivial |

Channels 0, 1, 2 are the expensive ones and are exactly what a GPU is for:
`128 x 128 x ~52` stone-distance evaluations, about 850K, which is miserable on
a CPU and nothing on a GPU. Every pixel is independent; one thread per pixel
loops the stone list from a uniform buffer.

`settled` is the awkward one, but less awkward than it looks. The cost there is
not per-pixel — it is per-stone geometry (`SettledRegion`, contour extraction
against the legal-vertex set), which is cheap and stays in WASM. What reaches
the GPU is a polygon per stone, and filling polygons is the one thing graphics
hardware has always done. Two options, in increasing order of effort:

1. **Fill on the CPU, upload the mask.** 128 x 128 bytes = 16 KB per position
   against 327 KB for the whole tensor. Keeps the transfer small and the shader
   simple. Start here.
2. **Fill on the GPU** by drawing the contours as triangle fans into the
   channel-3 slice. Removes the last upload. Do it only if step 1 measures as a
   bottleneck.

## Phases

Ordered so that the thing most likely to kill the project is measured first.

**Phase 0 — is it fast enough at all.** Export one checkpoint at fixed batch 1
and fp16, load it in onnxruntime-web on the WebGPU EP, measure milliseconds per
evaluation in a real browser. Everything downstream is a function of this
number: at 5 ms a 400-simulation move takes 2 s, at 30 ms it takes 12 s and the
project needs a different design. **No other work is worth doing before this
number exists.**

Two things to fix while doing it:

- Re-export at **fixed batch**. The dynamic-batch export carries 105 `Reshape`
  and 47 `Shape` nodes out of 604. Dynamic shapes are where ORT-web falls back
  to the CPU and stalls per node; a fixed batch constant-folds most of them.
- Export **fp16**. Halves the 33 MB download, and is already validated on this
  architecture — fp16 agreed with fp32 on 100% of policy argmaxes with value
  differing by 0.004.

**Phase 1 — strength at a browser-sized budget.** Measure Elo against
simulations for one checkpoint, playing it against itself at two budgets
(`vgo-arena --opponent-simulations`). This says what a 100- or 200-simulation
bot actually costs relative to the 800-simulation engine, which decides whether
the client-side bot is a real opponent or a toy. Relevant worry: the policy head
is the known ceiling — the search prior is ~0.6 bits from uniform — so cutting
search leans hardest on the weakest component.

**Phase 2 — the WGSL rasterizer**, validated bit-for-bit against
`rasterize_compact_into` over a corpus of real positions. This is the piece with
the most measurable payoff (23% of lane time) and it can be developed and tested
natively, before any browser work.

**Phase 3 — WASM build** of core/search/raster-contours, and the JS glue that
drives search, calls the shader, and hands the buffer to ORT.

**Phase 4 — packaging**: a single script tag or small module the community site
can embed, a difficulty setting that maps to simulation count, and a graceful
fallback.

## Fallbacks and reach

WebGPU is in Chrome and Edge everywhere, Safari 26+, and Firefox on Windows with
other platforms following. That is good but not universal, and this bot is for a
community site where "it does not work on my machine" is the failure that
matters. Three tiers:

1. **WebGPU** — the design above.
2. **WASM CPU inference** (ORT-web's default backend) with a much lower
   simulation count. Slower, works everywhere, still a real opponent.
3. **Remote** — point at `vgo-serve-move` for full strength, off by default,
   for anyone who wants to play the strong engine and is willing to use a
   server.

The tier should be chosen at runtime from feature detection, not configured.

## Open questions

- **Search tree memory.** A node carries a 336 KB fine grid at 128² policy
  resolution; a 800-simulation game tree is ~256 MB, which is hostile in a
  browser tab. Client-side probably means leaning on the coarse/fine split so
  only expanded nodes carry fine grids, or accepting far fewer simulations.
  Needs a measurement of actual resident size at 100-400 simulations.
- **Model size after fp16.** ~16 MB is acceptable for a game people return to,
  with the model cached. int8 would halve it again but has not been validated
  on this architecture and would need its own strength measurement.
- **Which checkpoint ships**, and how it gets updated as the run improves.
- **Whether the ridge channel needs f32.** If the shader can produce f16
  directly into an f16 model input, the tensor halves to 164 KB and the shader
  writes half as much.
