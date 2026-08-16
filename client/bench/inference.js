// How many simulations fit in a move-time budget?
//
// The whole client design hangs on one number: milliseconds per network
// evaluation in the browser. Everything else -- how strong the bot is, whether
// search is worth having at all -- follows from it, so it is measured rather
// than assumed.
//
// This runs under Node with onnxruntime-web's WASM backend, which is the
// client's *fallback* tier (no WebGPU here). Treat it as the floor: the WebGPU
// number from bench/inference.html on a real GPU should be better.
//
//   node bench/inference.js [budgetMillis]

import * as ort from 'onnxruntime-web';
import { readFile } from 'node:fs/promises';

const BUDGET_MS = Number(process.argv[2] ?? 5000);
const RASTER = 128;
const CHANNELS = 5;
// MCTS asks for a whole leaf round at once, so these are the batch sizes that
// actually occur. 1 is the root evaluation.
const BATCHES = [1, 2, 4, 8, 16];

function randomInput(batch) {
  const data = new Float32Array(batch * CHANNELS * RASTER * RASTER);
  // Plausible occupancy rather than pure noise: the disc channels are mostly
  // zero in a real position and sparsity can matter to a runtime.
  for (let i = 0; i < data.length; i += 1) data[i] = Math.random() < 0.06 ? 1 : 0;
  return new ort.Tensor('float32', data, [batch, CHANNELS, RASTER, RASTER]);
}

// Threads and SIMD are the difference between a usable CPU fallback and an
// unusable one, and they are not on by default everywhere. In a browser threads
// additionally need cross-origin isolation, so the single-threaded number is
// the one to plan around unless the host sets COOP/COEP.
const threads = Number(process.env.VGO_THREADS ?? 1);
ort.env.wasm.numThreads = threads;
ort.env.wasm.simd = true;
console.log(`wasm backend: ${threads} thread(s), simd on`);

const bytes = await readFile(new URL('../public/model.onnx', import.meta.url));
const session = await ort.InferenceSession.create(bytes, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
console.log(`inputs ${session.inputNames}  outputs ${session.outputNames}`);
console.log(`budget ${BUDGET_MS} ms per move\n`);
console.log('batch   ms/infer   ms/position   rounds in budget   simulations');

for (const batch of BATCHES) {
  const input = randomInput(batch);
  const feeds = { [session.inputNames[0]]: input };
  for (let i = 0; i < 3; i += 1) await session.run(feeds);

  const runs = batch >= 8 ? 8 : 20;
  const started = performance.now();
  for (let i = 0; i < runs; i += 1) await session.run(feeds);
  const perInfer = (performance.now() - started) / runs;

  // One round of MCTS is one inference of `batch` leaves. Rasterization and
  // tree work are excluded here; they are measured separately and are small
  // beside inference.
  const rounds = Math.floor(BUDGET_MS / perInfer);
  console.log(
    `${String(batch).padStart(5)} ${perInfer.toFixed(1).padStart(10)} ` +
    `${(perInfer / batch).toFixed(2).padStart(13)} ${String(rounds).padStart(18)} ` +
    `${String(rounds * batch).padStart(13)}`
  );
}
console.log('\nsimulations = rounds x batch, i.e. what a leaf_batch of that size buys.');
