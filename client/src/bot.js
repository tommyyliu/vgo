// The bot: MCTS in WebAssembly, inference in JavaScript.
//
// The search cannot call the network itself. `session.run()` returns a promise,
// and the thread that would block waiting for it is the same thread that must
// run the event loop to resolve it -- blocking there deadlocks rather than
// merely being slow. So the Rust side hands the loop out: it produces a batch
// of positions needing evaluation, we await the network, and we hand the
// results back.
//
// Because we own the loop, the bot thinks for a *time budget* rather than a
// fixed simulation count, which is what lets the same code behave sensibly on a
// desktop and on a phone.

import initWasm, { Game } from '../vendor/vgo_wasm.js';

/// Load the WebAssembly module.
///
/// Must be awaited before `new Game(...)` or any other export: until it
/// resolves the bindings are undefined and the failure reads as
/// "Cannot read properties of undefined (reading 'game_new')", which points at
/// the constructor rather than at the missing init.
let ready = null;
export function init() {
  ready ??= initWasm();
  return ready;
}

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

/// Fine cells per coarse sampling region, and not a free parameter: it must
/// match how the model was trained or the policy head is read through the wrong
/// sampler. Zero -- which is what `SearchConfig::canary` defaults to -- does not
/// mean "no pooling", it means candidate moves come from a quasi-random
/// sequence and the policy head stops guiding where the search looks at all.
/// The search still returns legal moves either way, so this is invisible except
/// in playing strength. `vgo-serve-move` uses 4.
const DEFAULT_COARSE_POOL = 4;

export class Bot {
  #session; #ort; #policySize; #leafBatch; #coarsePool; #inputName;

  static async create({
    ort, modelUrl, executionProviders,
    leafBatch = DEFAULT_LEAF_BATCH, coarsePool = DEFAULT_COARSE_POOL,
  }) {
    await init();
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders,
      graphOptimizationLevel: 'all',
    });
    const bot = new Bot();
    bot.#session = session;
    bot.#ort = ort;
    bot.#leafBatch = leafBatch;
    bot.#coarsePool = coarsePool;
    bot.#inputName = session.inputNames[0];
    // Read the policy width from the model rather than assuming it: the raster
    // and the policy grid are separate settings and have differed before.
    const probe = await session.run({
      [bot.#inputName]: new ort.Tensor(
        'float32', new Float32Array(CHANNELS * RASTER * RASTER), [1, CHANNELS, RASTER, RASTER],
      ),
    });
    bot.#policySize = probe.policy_logits.dims.at(-1);
    return bot;
  }

  get policySize() { return this.#policySize; }
  get leafBatch() { return this.#leafBatch; }
  get coarsePool() { return this.#coarsePool; }

  /// Choose a move for the side to move, thinking for at most `budgetMillis`.
  ///
  /// The budget is checked before each round, not during one, so it means "start
  /// no new round after this" rather than "answer by this". An in-flight
  /// inference always finishes, and the overrun is one inference: invisible at
  /// 12 ms per round on WebGPU, and seconds on a CPU fallback that is slow
  /// enough to matter. Cancelling mid-inference is not available to us, and
  /// throwing away a completed evaluation to hit a deadline exactly would be
  /// paying for the wrong thing.
  ///
  /// Returns `{ move, simulations, elapsed }` where `move` is `[x, y]`, or `[]`
  /// for a pass. `onProgress` is called after each round so a UI can show that
  /// the bot is thinking rather than hung.
  async chooseMove(game, { budgetMillis = 1500, seed = 1, onProgress } = {}) {
    // `seed` is u64 in Rust, which wasm-bindgen maps to BigInt: passing a
    // Number throws "Cannot convert ... to a BigInt". Coerce here so callers
    // can hand over an ordinary number.
    const search =
      game.search(0x3fffffff, BigInt(seed), this.#policySize, this.#coarsePool, this.#leafBatch);
    const deadline = performance.now() + budgetMillis;
    const started = performance.now();
    let simulations = 0;

    while (!search.finished && performance.now() < deadline) {
      const batch = search.nextBatch(RASTER);
      if (batch.length === 0) break;
      const count = batch.length / (CHANNELS * RASTER * RASTER);
      const outputs = await this.#session.run({
        [this.#inputName]:
          new this.#ort.Tensor('float32', batch, [count, CHANNELS, RASTER, RASTER]),
      });
      const values = outputs.values.data;
      const policies = outputs.policy_logits.data;
      search.submit(
        values instanceof Float32Array ? values : Float32Array.from(values),
        policies instanceof Float32Array ? policies : Float32Array.from(policies),
      );
      simulations = search.simulations;
      if (onProgress) onProgress(simulations, performance.now() - started);
    }
    // `best` is callable before the budget is spent: stopping on a deadline is
    // the intended use, not an error.
    const move = search.best();
    return { move, simulations, elapsed: performance.now() - started };
  }
}

export { Game, RASTER, CHANNELS };
