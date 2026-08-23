# Embedding the Voronoi Go bot

A browser opponent that runs entirely on the player's machine. No server, no
GPU to operate, no per-move cost — the site serves three static files and the
bot plays from there.

The rules, the search and the board rasterization are `crates/vgo-core`,
`crates/vgo-search` and `crates/vgo-raster` compiled to WebAssembly: the same
code self-play runs, not a reimplementation. Only the neural-network evaluation
happens in JavaScript, through onnxruntime-web.

```js
import { createBot } from './vgo-bot.js';
import * as ort from './ort.all.min.mjs';

const bot = await createBot({ ort, modelUrl: '/vgo/model.onnx' });

const move = await bot.chooseMove({
  radius: 1 / 18,
  komi: 0.104,
  toMove: 'W',
  passes: 0,
  stones: [{ x: 0.30, y: 0.35, c: 'B' }, { x: 0.70, y: 0.65, c: 'W' }],
}, { thinkMillis: 5000 });

// { pass: false, x: 0.41, y: 0.62, simulations: 3208, elapsed: 5012 }
```

`examples/embed.html` is that, runnable.

## What this is and is not

It answers exactly one question: **given this position and this much time, what
is the move?**

It has no board, no move history, no turn tracking and no notion of the game
being over. Those belong to the site, which already has them, and a bot that
kept its own copy would be a second source of truth to keep in sync. You hand it
a position; it hands back a move.

It is not a UI. There is no rendering here — `reference/js-reference/voronoi_go.html`
is the reference client if you want to see one, but it is a development tool
rather than something to ship.

## Building

```bash
npm --prefix client run build:wasm   # crates/vgo-wasm -> client/vendor/
npm --prefix client run build        # vendor/ + src/  -> client/dist/vgo-bot.js
```

`build:wasm` needs the Rust toolchain with the `wasm32-unknown-unknown` target
and `wasm-bindgen-cli` at a version matching the `wasm-bindgen` crate. `build`
needs only Node.

## What the site has to serve

| file | size | where it comes from |
|---|---|---|
| `vgo-bot.js` | 360 KB, 139 KB gzipped | `npm run build` |
| onnxruntime-web | ~810 KB + one backend `.wasm` | `node_modules/onnxruntime-web/dist/` |
| the model | ~33 MB | exported from a training run |

The bot's own WebAssembly is inlined into `vgo-bot.js`, so that file is
self-contained. The other two are not bundled on purpose: both are large,
cacheable, and already artefacts you deploy deliberately. Inlining them would
turn a 360 KB module into a 60 MB one that could not be cached apart from the
code.

**onnxruntime-web fetches its own siblings.** `ort.all.min.mjs` loads
`ort-wasm-simd-threaded.*.wasm` from the directory it was served from, so copy
the whole set, not just the `.mjs`. A 404 on those shows up as an execution
provider silently failing over rather than as a missing-file error.

**Cross-origin isolation is optional.** WebGPU does not need it. Multi-threaded
WASM inference does, because it needs `SharedArrayBuffer`, and that is the
difference between ~95 ms and ~25 ms per position on the CPU fallback. A host
that cannot set `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` still gets full-speed WebGPU and a
slower fallback. `serve.js` sets both, for development.

## `createBot(options)`

| option | default | what it does |
|---|---|---|
| `ort` | — | an imported onnxruntime-web module |
| `ortUrl` | — | where to import one from, if you have not already. One of these two is required |
| `modelUrl` / `modelBuffer` | — | the exported `.onnx`, by URL or as bytes. Required |
| `executionProviders` | `['webgpu','wasm']` | ORT takes the first that initialises |
| `leafBatch` | 8 | positions per inference |
| `coarsePool` | 16 | how candidates are drawn from the policy map |
| `rasterKind` | `'compact-pass'` | channel layout the model reads; a property of the export |
| `wasm` | inlined | override for the engine binary; only needed in unusual hosting |

Expensive, and worth doing once when a board loads rather than when a player
asks for a move: it fetches tens of megabytes of model and compiles a WebGPU
pipeline.

**`rasterKind` must match the export.** `compact-pass` -- the five compact planes
plus "the previous move was a pass" -- is what every model since that plane was
added is trained on, and is the default. Older exports are five-channel
`compact`. `createBot` compares the width the layout produces against the
model's own declared input and refuses a mismatch, so the usual failure is a
clear error at load rather than a bad bot. The case it cannot catch is two
layouts of the same width: `compact-pass` and `compact-dead-zone` are both six
planes and differ in the capture predicate, so passing the wrong one there loads
cleanly and plays blind.

**Leave `coarsePool` alone unless you know why you are changing it.** Zero is not
"off" — it makes the search draw candidate moves from a quasi-random sequence
instead of the network's policy map, so the policy head stops guiding where the
search looks at all, and the bot goes on playing legal, plausible-looking, much
weaker moves. Any other value searches the model through a different sampler than
the one its strength was measured with. Sixteen is what every recipe in `runs/`
uses.

`leafBatch` is a throughput choice and is safe to lower on a machine where memory
matters more than speed; see *Known limits*.

## The position

```js
{
  radius: 1 / 18,      // stone radius. Default 1/18 — but see the note below
  komi: 0.104,         // area spotted to White. Default 0 — see below
  toMove: 'B',         // 'B' | 'W' | 'black' | 'white'
  passes: 0,           // consecutive passes before this position. Default 0
  stones: [ { x, y, c } ],
  ply: 12,             // optional; currently unused by the search
}
```

Coordinates are the **unit square**: the board is `[0,1] × [0,1]` however large
it is drawn, so scale your own pixels and nothing here depends on the display.
`voronoigo.com` uses an 18-unit board with radius 1, so divide its coordinates
by `boardSize` and its radius becomes exactly 1/18.

**The models were not trained at 1/18.** Every run so far used `39/700`, 0.286%
larger — a board 17.95 stone-radii wide rather than 18. That is not a game
constant; it is the reference client's radius slider sitting at its default of 39
pixels on a 700-pixel board, which is where the training recipes took it from.
`DEFAULT_RADIUS` is the game's value and `TRAINING_RADIUS` is the other one, so
whichever you want you can name. Send your own `radius` and the question does not
arise.

Stone colours may be `c` or `color`, and `'B'`/`'W'` or `'black'`/`'white'` —
this repository contains three implementations using two conventions, and making
you translate would only buy an integration bug.

Two fields are easy to leave out, and both change the game:

**`komi` defaults to 0.** That is honest — a record that does not mention komi
was played at none — but it is a large handicap, not a neutral setting. At komi
0 Black wins about nine games in ten. The balanced value measured on this engine
is **0.104**, and it drifts as the models improve, so treat it as a number to
re-measure rather than a constant.

**`passes` is how many consecutive passes precede this position.** A search that
believes nobody has passed does not know that passing now would end the game, so
it can neither pass to close out a win nor see that passing while behind hands
over the result. A board full of stones does not carry this. If your site tracks
it — and it must, to end games at all — send it.

## `chooseMove(position, options)`

| option | default | what it does |
|---|---|---|
| `thinkMillis` | 5000 | time budget |
| `maxSimulations` | unlimited | hard cap on search |
| `seed` | random | search is deterministic given a seed |
| `signal` | — | an `AbortSignal` |
| `onProgress` | — | `(simulations, elapsedMillis)` after each round |

Returns `{ pass, x, y, simulations, elapsed }`. When `pass` is true there is no
`x` or `y`, and **you apply your own pass rule** — the bot has no history and
cannot know whether that pass ends the game.

`thinkMillis` is checked *between* rounds, not during one, so it means "start no
new round after this" rather than "answer by this". An in-flight inference always
finishes. The overrun is one inference: invisible at ~12 ms on WebGPU, seconds on
a slow CPU fallback. Cancelling mid-inference is not available to us, and
discarding a finished evaluation to hit a deadline exactly would be paying for
the wrong thing.

`signal` is checked at the same points. Abort when the player takes a move back
or navigates away; a five-second search left running is a five-second search
competing with the page.

### Difficulty

Prefer `thinkMillis` for a practice opponent: one setting stays sensible on
hardware nobody measured. Prefer `maxSimulations` for anything rated — equal
time is unequal strength across machines, so a ladder built on time is a ladder
that ranks hardware.

For reference, search is worth roughly 250 Elo per doubling on this engine
(measured: 1600 against 800 simulations was +232), and the run that produced the
current model generated its training games at 3,200.

## Errors

Everything thrown is a `VgoBotError` with a `code`, so you never match on message
text:

| code | means |
|---|---|
| `unsupported` | the options or runtime cannot support a bot |
| `invalid-position` | malformed, or the engine considers it unplayable |
| `finished` | the game is already over |
| `aborted` | your `AbortSignal` fired |
| `inference` | the model failed to load or run |

`invalid-position` is worth surfacing during development rather than swallowing:
it means your board and the Rust rules disagree about what is legal, which is a
real bug somewhere and not a bad request.

## Speed

Measured in Chrome on a discrete GPU. One inference costs about the same whether
it carries 1 position or 8, so per-position cost collapses with the leaf batch:

| backend | ms per position | simulations in 5 s |
|---|---|---|
| WebGPU, leaf batch 8 | 1.56 | ~3,200 |
| WebGPU, leaf batch 1 | 13.0 | ~385 |
| WASM CPU fallback | 42–47 | ~110 |

WebGPU ships in Chrome and Edge everywhere, Safari 26+, and Firefox on Windows;
Firefox on Linux is nightly/beta-only as of August 2026. `createBot` defaults to
`executionProviders: ['webgpu', 'wasm']` and ORT takes the first that
initialises, so the fallback is automatic — but it is a different experience
rather than a slightly slower one, and is worth telling the player about.

## Known limits

Stated because they will be noticed:

- **No tree reuse between moves.** Each `chooseMove` searches from scratch, so
  the subtree matching the move actually played is thrown away. That is the
  price of a stateless API and it is a real strength loss at no benefit to the
  player. Fixable without changing the API surface.
- **No analysis output.** There is no win probability and no candidate list —
  only the chosen move. `vgo-serve-move` returns its top candidates and this
  does not, which is an asymmetry rather than a decision.
- **No model identity.** The bot cannot tell you which checkpoint it is running,
  so a site cannot log or display it.
- **Memory is unbounded and unmeasured.** A search node carries a 336 KB fine
  grid at 128² policy resolution. At 3,200 simulations that is potentially
  hostile in a tab, and nobody has measured resident size in a real one. Cap
  `maxSimulations` if you see trouble; this is the most likely thing to bite.
- **One search at a time per bot.** Concurrent `chooseMove` calls share an
  inference session and are not serialised here. Await one before starting the
  next, or build a second bot.
- **`ply` is currently inert.** It is accepted because ply-dependent search
  settings exist, but all of them are off for this use, and its default
  (`stones.length`) is wrong after any capture.

## Layout

    src/index.js       the API. The whole thing
    build.js           folds the wasm-bindgen glue and the inlined WASM into one module
    dist/vgo-bot.js    the built bundle (generated; not in git)
    vendor/            wasm-bindgen output (generated; not in git)
    examples/embed.html  smallest working integration, and the bundle's smoke test
    serve.js           development server for the repository root
    bench/, public/    inference benchmarks and their assets

`reference/js-reference/voronoi_go.html` imports `src/index.js` rather than the
built bundle, deliberately: the API a site integrates against is the one
exercised there, so it cannot rot while still appearing to work.

Design rationale, measurements and the things that turned out to be wrong are in
[`../docs/CLIENT_BOT.md`](../docs/CLIENT_BOT.md).
