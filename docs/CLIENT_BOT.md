# A client-side bot

Written 2026-08-16 as a plan; updated the same day, once it was built, to say
what actually happened. **It is built and wired into
`reference/js-reference/voronoi_go.html`** — pick `engine: in browser` in the
model-opponent panel.

    npm --prefix client run build:wasm     # crates/vgo-wasm -> client/vendor
    node client/serve.js                   # serves the repository root
    # http://localhost:8123/reference/js-reference/voronoi_go.html

Two build inputs are fetched or generated rather than tracked, and both must be
present: `client/vendor/vgo_wasm*` from the command above, and
`client/public/model.onnx`, an exported checkpoint.

The headline correction is below in **The mistake to avoid**: the design that
document argued for — rasterize in a compute shader so the tensor never leaves
the GPU — was not built and is not needed. The reasoning was right about where
the cost was and wrong about what to do with it.

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

So the design goal was: **the raster is produced on the GPU and never leaves
it.**

**That is not what was built, and the premise did not survive contact.** Two
things changed underneath it.

The first is that rasterization stopped being expensive. Computing the settled
channel from distance transforms rather than a per-stone geometric solve made it
2.8-7.2x cheaper and, more usefully, flat in the stone count where the old one
quadrupled between 28 and 52 stones. `crates/vgo-raster` compiles to WASM with
everything else, so the raster is produced next to the search that asked for it
and the shader has nothing left to win.

The second is that the 23%-versus-2.3% split is a fact about the *native*
pipeline, where inference runs on a local GPU at batch 32 through TensorRT. In a
browser the ratio inverts: one WebGPU inference costs 10-13 ms almost regardless
of batch size, which dwarfs a rasterization measured in fractions of a
millisecond. Optimising the raster there would be optimising the cheap half —
the same error, pointed the other way.

The transfer worry was real and is handled by batching rather than by sharing
buffers. Per-position upload cost falls with batch size exactly as inference
does, so a leaf batch of 8 pays it once for eight positions.

## Shape of the thing

    ┌─ WASM (compiled from the existing Rust) ────────────┐
    │  vgo-core    rules, legality, settlement            │
    │  vgo-search  MCTS, coarse/fine selection, stepped   │
    │  vgo-raster  the full [5,128,128] tensor            │
    └──────────────┬──────────────────────────────────────┘
                   │ Float32Array, batch * 5 * 128 * 128
                   v
    ┌─ onnxruntime-web, WebGPU EP ────────────────────────┐
    │  policy_logits [batch, 16385], values [batch]       │
    └──────────────┬──────────────────────────────────────┘
                   │ values and policies, same order
                   v
              back into the search, which resumes

The loop is driven from JavaScript because it has to be: `session.run()` returns
a promise and the thread that would block on it is the thread that resolves it.
`SteppedSearch` hands the loop out for that reason, and the side effect is the
one that matters for a browser — the caller can stop at a *deadline* instead of
a simulation count, so the same code is sensible on a desktop and a phone.

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

Ordered so that the thing most likely to kill the project was measured first.

**Phase 0 — is it fast enough at all. Done, and the answer is yes.** Measured in
Chrome on a discrete GPU, WebGPU EP, dynamic-batch fp32 export:

| leaf batch | ms per position | positions in 5 s |
|---|---|---|
| 1 | 13.0 | 385 |
| 8 | 1.56 | 3,208 |

One inference costs roughly the same whether it carries 1 position or 8, so the
batch is nearly free throughput. At batch 8 a 5-second move is ~3,200
simulations, which is *four times* the 800 self-play uses — the budget is not
the constraint anyone expected it to be. Firefox on Linux has no WebGPU outside
nightly today and falls to the WASM backend at 42-47 ms per position.

The two export improvements below were **not** needed to clear the bar and have
not been done. They are still worth doing, mostly for the download:

- Re-export at **fixed batch**. The dynamic-batch export carries 105 `Reshape`
  and 47 `Shape` nodes out of 604. Dynamic shapes are where ORT-web falls back
  to the CPU and stalls per node; a fixed batch constant-folds most of them.
- Export **fp16**. Halves the 33 MB download, and is already validated on this
  architecture — fp16 agreed with fp32 on 100% of policy argmaxes with value
  differing by 0.004.

**Phase 1 — strength at a browser-sized budget. Not done.** Measure Elo against
simulations for one checkpoint, playing it against itself at two budgets
(`vgo-arena --opponent-simulations`). Phase 0 makes this less urgent than it
looked, since 3,200 simulations is above self-play's budget rather than below
it, but it is still the number that says whether the bot is an opponent or a
toy. Relevant worry: the policy head is the known ceiling — the search prior is
~0.6 bits from uniform — so cutting search leans hardest on the weakest
component.

The specific open measurement is **leaf batch 8 against 4**. Eight is what the
bot uses and it is chosen for throughput; MCTS pays for a wide leaf batch in
search quality, because concurrent descents under virtual loss collide and
select less informatively. Self-play uses 4. This is measurable natively with
`vgo-arena` and has not been.

**Phase 2 — the WGSL rasterizer. Abandoned**, for the reasons under *The mistake
to avoid*. `crates/vgo-raster-cuda` is the surviving fragment of this line of
work and is not used either.

**Phase 3 — WASM build. Done.** `crates/vgo-wasm` exposes `Game` and `Search`;
`client/src/bot.js` drives the loop.

**Phase 4 — packaging. Done, apart from difficulty.** `client/dist/vgo-bot.js`
is one 360 KB ES module with the engine's WASM inlined; see *Embedding* below.
The bot is also wired into the reference client behind an engine selector, with
the server backend as the default.

What is left is a difficulty setting, which should map to *time*, not simulation
count, since that is the axis the search is now driven on — with the caveat that
equal time is unequal strength across machines, so a site running a ladder wants
`maxSimulations` instead.

## Embedding

    npm --prefix client run build:wasm    # crates/vgo-wasm -> client/vendor/
    npm --prefix client run build         # vendor/ + src/  -> client/dist/vgo-bot.js

`client/examples/embed.html` is the smallest working integration and the smoke
test for the bundle. The API is `client/src/index.js`, which is also what the
reference client imports — deliberately, so the API a host integrates against is
the one that gets exercised here and cannot rot while still appearing to work.

```js
import { createBot } from './vgo-bot.js';
import * as ort from './ort.all.min.mjs';

const bot = await createBot({ ort, modelUrl: '/vgo/model.onnx' });

const move = await bot.chooseMove({
  radius: 1 / 18, komi: 0.104, toMove: 'B', passes: 0,
  stones: [{ x: 0.5, y: 0.5, c: 'B' }],
}, { thinkMillis: 5000, signal, onProgress });
// { pass: false, x: 0.41, y: 0.62, simulations: 3208, elapsed: 5012 }
```

The bot knows nothing about games — no board, no history, no turn tracking, no
game-over. The host owns all of that and asks one question: given this position
and this much time, what is the move?

Three things a host has to serve, and only one is ours:

| file | size | bundled? |
|---|---|---|
| `dist/vgo-bot.js` | ~360 KB | engine WASM inlined |
| onnxruntime-web | ~810 KB + a backend `.wasm` it fetches | no |
| the model | ~33 MB | no |

The last two stay out on purpose. Both are large, cacheable, and already
artefacts a host deploys deliberately; inlining them would turn a 360 KB module
into a 60 MB one that cannot be cached apart from the code. Our WASM *is*
inlined, because the failure it prevents is a page that loads the module and
then 404s on a `.wasm` nobody remembered to copy.

Two fields are easy to leave out and both change the game:

- **`komi` defaults to 0**, which is honest — a record that does not mention it
  was played at none — but it is a large handicap, not a neutral setting. At
  komi 0 Black wins about nine games in ten. The balanced value measured on this
  engine is 0.104, and it drifts as the models improve.
- **`passes`** is how many consecutive passes precede the position. A search
  that believes nobody has passed does not know that passing now would end the
  game, so it can neither pass to close out a win nor see that passing while
  behind hands over the result. A board full of stones does not carry this, so
  the host must send it. **`vgo-serve-move` has the same blind spot** — its
  protocol has no field for it — so the server backend still plays the endgame
  without knowing whether a pass ends anything.

Errors are all `VgoBotError` with a `code` (`unsupported`, `invalid-position`,
`finished`, `aborted`, `inference`), so a host can tell a bad position from a
missing model without matching on message text.

### One thing the wiring got wrong twice

`SearchConfig::canary` is a test default and gets two fields wrong for playing
use. Both were live in the browser bot until measured, and neither is visible
without measuring: the search returns legal, plausible moves either way.

- `coarse_pool = 0` does not mean "no pooling". It means candidates come from
  the legacy quasi-random sequence rather than the network's policy map — the
  policy head stops choosing where to look at all. `vgo-serve-move` uses 4.
- `leaf_batch = 1` throws away the entire batching win above: 13.0 ms per
  position instead of 1.56.

`Game::search` now takes both explicitly, and refuses to make either implicit.

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

Tiers 1 and 2 are chosen at runtime by ORT, from `executionProviders:
['webgpu', 'wasm']` — no feature detection of our own. Tier 3 is the `engine`
selector in the client, and is the default there because that page is often
opened straight off disk, where ES modules and WASM cannot load at all.

Cross-origin isolation (COOP/COEP) is what `client/serve.js` sets, and it is
worth being precise about who needs it. WebGPU does not. Multi-threaded WASM
inference does, because it needs `SharedArrayBuffer`, and that is the difference
between ~95 ms and ~25 ms per position on the fallback tier. So a host that
cannot set those headers still gets tiers 1 and 3 intact and a slower tier 2.

## Open questions

- **Search tree memory**, and Phase 0 made this *worse* rather than better. A
  node carries a 336 KB fine grid at 128² policy resolution, so an
  800-simulation tree is ~256 MB — and the measured browser budget is 3,200
  simulations in five seconds, not 800. Nothing has measured resident size in a
  real tab, and a tab that dies at second four is a worse failure than a weak
  move. This is the most likely thing to bite next.
- **Model size after fp16.** ~16 MB is acceptable for a game people return to,
  with the model cached. int8 would halve it again but has not been validated
  on this architecture and would need its own strength measurement.
- **Which checkpoint ships**, and how it gets updated as the run improves.
- **Whether the ridge channel needs f32.** If the shader can produce f16
  directly into an f16 model input, the tensor halves to 164 KB and the shader
  writes half as much.
