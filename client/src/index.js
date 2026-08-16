// The embeddable Voronoi Go bot.
//
// This is the whole public surface. A host site builds a bot once, then hands
// it positions and gets moves back:
//
//     import { createBot } from './vgo-bot.js';
//     import * as ort from './ort.all.min.mjs';
//
//     const bot = await createBot({ ort, modelUrl: '/vgo/model.onnx' });
//     const move = await bot.chooseMove({
//       radius: 1 / 18, komi: 0.104, toMove: 'B',
//       stones: [{ x: 0.5, y: 0.5, c: 'B' }],
//     }, { thinkMillis: 5000 });
//     // -> { pass: false, x: 0.41, y: 0.62, simulations: 3208, elapsed: 5012 }
//
// It deliberately does not know what a game is. There is no board, no history,
// no turn tracking and no notion of the game being over -- the host owns all of
// that and passes a position in. What this owns is one question: given this
// position and this much time, what is the move?
//
// ## Rules and search are the Rust engine, not a reimplementation
//
// Everything about legality, capture, settlement and scoring comes from
// `crates/vgo-core` compiled to WebAssembly, and the search is `crates/vgo-search`
// unchanged. That is the point of the WASM detour: there are already two
// implementations of these rules, they carry mutual "must match" comments, and
// they have still diverged in practice -- a placement capturing exactly one
// stone read as a pass and ended live games, and the fix had to be found twice.
// A third implementation, in a codebase nobody here maintains, would be worse.
//
// ## Why inference is not also in the WASM
//
// It cannot be, on the thread that matters. `session.run()` returns a promise,
// and the thread that would block waiting for it is the thread that must run
// the event loop to resolve it -- blocking there deadlocks rather than merely
// being slow. So the search hands its loop out: it yields a batch of positions
// needing evaluation, we await the network, and we hand the results back.
//
// Owning the loop is what makes the budget a *time* rather than a simulation
// count, which is what lets one setting be sensible on a desktop and a phone.

import initWasm, { Game } from '../vendor/vgo_wasm.js';

/// Replaced by build.js with a base64 string in the bundled build. Null here,
/// in the source tree, where the loader's own default finds vgo_wasm_bg.wasm
/// sitting next to the glue. build.js asserts it found this line, so renaming
/// it breaks the build loudly rather than silently shipping a bundle that
/// fetches a file the host does not have.
const INLINE_WASM_BASE64 = null;

/// The official game's stone radius: `voronoigo.com` plays an 18-unit board
/// with radius 1, so exactly 1/18 of the board.
///
/// **The models were not trained at this radius.** Every run so far used
/// 39/700 = 0.05571..., which is 0.286% larger -- a board 17.95 stone-radii
/// wide instead of 18. That number is not a game constant: it is the reference
/// client's default radius slider sitting at 39 pixels on a 700-pixel board,
/// which is where the recipes copied it from.
///
/// The default here is the game's value rather than the training value, because
/// a host that omits `radius` means "the game", and a bot playing the real game
/// slightly out of distribution is the lesser wrong. Hosts should send their own
/// radius regardless.
export const DEFAULT_RADIUS = 1 / 18;

/// What every run in `runs/` has trained and measured at, for callers that want
/// to reproduce a training position exactly rather than play the live game.
export const TRAINING_RADIUS = 39 / 700;

/// Raster the model reads, and the number of channels in it. Both are fixed by
/// the exported network rather than chosen here.
const RASTER = 128;
const CHANNELS = 5;

/// Leaf batch. Measured in Chrome on a discrete GPU: one inference costs
/// ~10-13 ms almost regardless of how many positions are in it, so per-position
/// cost collapses as the batch grows -- 13.0 ms at batch 1 against 1.56 at
/// batch 8. That is the opposite of the native pipeline, where batch 32 was
/// optimal and extra inference lanes actively hurt.
///
/// Larger is not free: MCTS pays for a wide leaf batch in search *quality*,
/// because concurrent descents under virtual loss collide and select less
/// informatively. Self-play uses 4. Eight is chosen as the point where the
/// throughput win is large and the quality cost is believed small -- believed,
/// not measured, which is the open question here.
const DEFAULT_LEAF_BATCH = 8;

/// Fine cells per coarse sampling region, when drawing candidate moves from the
/// policy map. Sixteen is what every recipe in `runs/` uses -- for generation,
/// for arenas and for tournaments -- so it is what the shipped models are
/// searched with everywhere else.
///
/// Two ways to get this wrong, and neither is visible in the moves:
///
///   * **Zero** does not mean "no pooling". It means candidates come from a
///     quasi-random sequence instead of the network's policy map, so the policy
///     head stops guiding where the search looks at all.
///   * **A different positive value** searches the same model through a
///     different sampler than it is served through anywhere else. Not wrong in
///     the way zero is wrong -- this is a search knob, not a property of the
///     network -- but it makes browser play diverge from measured play for no
///     reason.
///
/// The defaults in `vgo-serve-move` and `pipeline.py` were 4, which no run has
/// used; that is where this constant was first copied from.
const DEFAULT_COARSE_POOL = 16;

/// Five seconds. Long enough to be worth the wait, short enough that a player
/// does not think the page has hung, and around 3,200 simulations on a discrete
/// GPU -- four times what self-play uses.
const DEFAULT_THINK_MILLIS = 5000;

/// Everything this module throws, so a host can tell a bad position from a
/// missing model from a user-cancelled search without matching on message text.
///
/// `code` is one of:
///   'unsupported'       the runtime or the options cannot support a bot
///   'invalid-position'  the position is malformed or not playable
///   'finished'          the game is already over; there is no move to make
///   'aborted'           the caller's AbortSignal fired
///   'inference'         the model failed to load or to run
export class VgoBotError extends Error {
  constructor(code, message, cause) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'VgoBotError';
    this.code = code;
  }
}

let wasmReady = null;

/// Load the WebAssembly engine, once per page.
///
/// Must complete before any other export touches it: until it resolves the
/// bindings are undefined, and the failure reads as "Cannot read properties of
/// undefined (reading 'game_new')", which points at the constructor rather than
/// at the missing init.
function loadEngine(source) {
  wasmReady ??= (async () => {
    if (source) return initWasm({ module_or_path: source });
    if (INLINE_WASM_BASE64) return initWasm({ module_or_path: decodeBase64(INLINE_WASM_BASE64) });
    return initWasm();
  })();
  return wasmReady;
}

function decodeBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/// Accepts every spelling of a colour any of the three implementations uses, so
/// an integration cannot fail on 'B' against 'black'. Rejecting the others and
/// making the host translate would buy nothing.
function colourName(value, what) {
  const text = String(value ?? '').trim().toLowerCase();
  if (text === 'b' || text === 'black') return 'black';
  if (text === 'w' || text === 'white') return 'white';
  throw new VgoBotError('invalid-position', `${what} must be "B" or "W", got ${JSON.stringify(value)}`);
}

function finiteNumber(value, fallback, what) {
  if (value === undefined || value === null) return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new VgoBotError('invalid-position', `${what} must be a finite number, got ${JSON.stringify(value)}`);
  }
  return number;
}

/// Normalise a host's position into what the engine takes.
///
/// Coordinates are the unit square: the board is [0,1] x [0,1] whatever size it
/// is drawn at, so a host scales its own pixels and nothing here depends on the
/// display. Colours may be 'B'/'W' or 'black'/'white', on `c` or `color`.
function readPosition(input) {
  if (!input || typeof input !== 'object') {
    throw new VgoBotError('invalid-position', 'a position object is required');
  }
  const radius = finiteNumber(input.radius, DEFAULT_RADIUS, 'radius');
  if (radius <= 0 || radius >= 0.5) {
    throw new VgoBotError('invalid-position', `radius must be between 0 and 0.5, got ${radius}`);
  }
  // Zero is the honest default -- a record that does not mention komi was played
  // at none -- but it is a large handicap rather than a neutral setting: at komi
  // 0 Black wins about nine games in ten. The balanced value measured on this
  // engine is 0.104, and it drifts as the models improve.
  const komi = finiteNumber(input.komi, 0, 'komi');
  const stones = Array.from(input.stones ?? []).map((stone, index) => {
    if (!stone || typeof stone !== 'object') {
      throw new VgoBotError('invalid-position', `stone ${index} is not an object`);
    }
    return {
      x: finiteNumber(stone.x, undefined, `stone ${index} x`),
      y: finiteNumber(stone.y, undefined, `stone ${index} y`),
      color: colourName(stone.c ?? stone.color, `stone ${index} colour`),
    };
  });
  const passes = Math.max(0, Math.floor(finiteNumber(input.passes, 0, 'passes')));
  if (passes >= 2) {
    throw new VgoBotError('finished', 'two consecutive passes have already ended this game');
  }
  return {
    radius,
    komi,
    stones,
    passes,
    toMove: colourName(input.toMove ?? input.to_move ?? 'B', 'toMove'),
    // Only ever read by ply-dependent search settings, all of which are off
    // here, so an absent one costs nothing.
    ply: Math.max(0, Math.floor(finiteNumber(input.ply, stones.length, 'ply'))),
  };
}

async function resolveOrt(options) {
  if (options.ort) return options.ort;
  if (options.ortUrl) {
    try {
      return await import(/* webpackIgnore: true */ /* @vite-ignore */ options.ortUrl);
    } catch (error) {
      throw new VgoBotError('unsupported', `could not load onnxruntime-web from ${options.ortUrl}`, error);
    }
  }
  throw new VgoBotError(
    'unsupported',
    'pass `ort` (an imported onnxruntime-web module) or `ortUrl` (where to import one from). ' +
    'It is not bundled here: it ships its own multi-megabyte WASM binaries, which a host ' +
    'should serve and cache on its own terms.',
  );
}

/// Build a bot. Loads the engine, the runtime and the model, in that order.
///
/// Expensive and worth doing once: the model is tens of megabytes and the
/// WebGPU pipeline has to be compiled. Do it when the page loads a board, not
/// when the player asks for a move.
///
/// Options:
///   ort / ortUrl        onnxruntime-web, imported or to import. One required.
///   modelUrl            the exported .onnx. Required unless `modelBuffer`.
///   modelBuffer         model bytes, if the host already has them.
///   executionProviders  default ['webgpu', 'wasm']; ORT takes the first that
///                       initialises. WebGPU is ~1.6 ms per position and the
///                       WASM fallback 42-47, so this is a large difference in
///                       kind rather than degree.
///   wasm                override for the engine binary: bytes or a URL. Only
///                       needed when neither the bundled copy nor the file next
///                       to the glue is available.
///   leafBatch           positions per inference. Default 8.
///   coarsePool          policy sampling factor. Default 4. See above; this is
///                       a property of the trained model, not a preference.
export async function createBot(options = {}) {
  const { modelUrl, modelBuffer } = options;
  if (!modelUrl && !modelBuffer) {
    throw new VgoBotError('unsupported', 'pass `modelUrl` or `modelBuffer`');
  }
  const [ort] = await Promise.all([resolveOrt(options), loadEngine(options.wasm)]);

  let session;
  try {
    session = await ort.InferenceSession.create(modelBuffer ?? modelUrl, {
      executionProviders: options.executionProviders ?? ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    });
  } catch (error) {
    throw new VgoBotError('inference', `could not load the model: ${error?.message ?? error}`, error);
  }
  return new Bot(ort, session, options);
}

class Bot {
  #ort; #session; #inputName; #policySize; #leafBatch; #coarsePool; #disposed = false;

  constructor(ort, session, options) {
    this.#ort = ort;
    this.#session = session;
    this.#inputName = session.inputNames[0];
    this.#leafBatch = Math.max(1, Math.floor(options.leafBatch ?? DEFAULT_LEAF_BATCH));
    this.#coarsePool = Math.max(0, Math.floor(options.coarsePool ?? DEFAULT_COARSE_POOL));
    this.#policySize = null;
  }

  /// What this bot is, for a host that wants to show or log it. `policySize` is
  /// null until the first move, because it is read from the model rather than
  /// assumed -- the raster and the policy grid are separate settings and have
  /// differed before.
  get info() {
    return {
      policySize: this.#policySize,
      leafBatch: this.#leafBatch,
      coarsePool: this.#coarsePool,
      raster: RASTER,
      channels: CHANNELS,
    };
  }

  /// Release the inference session. The bot is unusable afterwards.
  dispose() {
    this.#disposed = true;
    return this.#session.release?.();
  }

  /// Choose a move for the side to move in `position`.
  ///
  /// Returns `{ pass, x, y, simulations, elapsed }`. When `pass` is true there
  /// is no `x` or `y`, and the host applies its own pass rule -- this module has
  /// no history and cannot know whether that pass ends the game, which is why
  /// `position.passes` is worth sending.
  ///
  /// Options:
  ///   thinkMillis     time budget, default 5000. Checked between rounds, not
  ///                   during one, so it means "start no new round after this"
  ///                   rather than "answer by this". An in-flight inference
  ///                   always finishes: the overrun is one inference, invisible
  ///                   at 12 ms on WebGPU and seconds on a slow fallback.
  ///                   Cancelling mid-inference is not available to us, and
  ///                   discarding a finished evaluation to hit a deadline
  ///                   exactly would be paying for the wrong thing.
  ///   maxSimulations  hard cap, for a difficulty setting that must be equal
  ///                   across machines rather than equal in wall time.
  ///   seed            search is deterministic given a seed. Random by default,
  ///                   so a rematch is not a replay; fix it to reproduce a game.
  ///   signal          an AbortSignal. Checked between rounds; aborting rejects
  ///                   with code 'aborted'.
  ///   onProgress      called after each round with (simulations, elapsedMillis).
  async chooseMove(position, options = {}) {
    if (this.#disposed) throw new VgoBotError('unsupported', 'this bot has been disposed');
    const {
      thinkMillis = DEFAULT_THINK_MILLIS,
      maxSimulations = 0x3fffffff,
      seed = Math.floor(Math.random() * 0x7fffffff),
      signal,
      onProgress,
    } = options;

    const parsed = readPosition(position);
    const game = new Game(parsed.radius, parsed.komi);
    try {
      game.setStones(parsed.stones, parsed.toMove, parsed.ply, parsed.passes);
    } catch (error) {
      // vgo-core refused it: overlapping stones, a stone off the board, a
      // radius nothing fits at. The host's board and this engine disagree about
      // what is legal, and that is worth surfacing rather than searching anyway.
      throw new VgoBotError('invalid-position', `the engine rejected this position: ${error?.message ?? error}`, error);
    }
    if (game.finished) {
      throw new VgoBotError('finished', 'this game is already over');
    }

    // Read the policy width from the model on first use rather than assuming it.
    this.#policySize ??= await this.#probePolicySize();

    const search = game.search(
      Math.max(1, Math.floor(maxSimulations)),
      BigInt(seed),
      this.#policySize,
      this.#coarsePool,
      this.#leafBatch,
    );
    const started = performance.now();
    const deadline = started + thinkMillis;
    let simulations = 0;

    while (!search.finished && performance.now() < deadline) {
      if (signal?.aborted) throw new VgoBotError('aborted', 'the search was aborted');
      const batch = search.nextBatch(RASTER);
      if (batch.length === 0) break;
      const count = batch.length / (CHANNELS * RASTER * RASTER);
      const outputs = await this.#run(batch, count);
      search.submit(
        toFloat32(outputs.values.data),
        toFloat32(outputs.policy_logits.data),
      );
      simulations = search.simulations;
      onProgress?.(simulations, performance.now() - started);
    }
    if (signal?.aborted) throw new VgoBotError('aborted', 'the search was aborted');

    // `best` is callable before the budget is spent: stopping on a deadline is
    // the intended use, not an error.
    const move = search.best();
    const elapsed = performance.now() - started;
    return move.length === 0
      ? { pass: true, simulations, elapsed }
      : { pass: false, x: move[0], y: move[1], simulations, elapsed };
  }

  async #run(batch, count) {
    try {
      return await this.#session.run({
        [this.#inputName]:
          new this.#ort.Tensor('float32', batch, [count, CHANNELS, RASTER, RASTER]),
      });
    } catch (error) {
      throw new VgoBotError('inference', `the model failed to run: ${error?.message ?? error}`, error);
    }
  }

  async #probePolicySize() {
    const empty = new Float32Array(CHANNELS * RASTER * RASTER);
    const outputs = await this.#run(empty, 1);
    return outputs.policy_logits.dims.at(-1);
  }
}

/// ORT returns whatever type the output tensor holds; the engine takes f32.
function toFloat32(data) {
  return data instanceof Float32Array ? data : Float32Array.from(data);
}
