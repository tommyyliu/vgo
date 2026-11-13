// Replays recorded games through the Rust rules and reports any disagreement.
//
//   bun tools/replay.ts <dir-or-sgf>...
//
// Prints one line per game that disagrees, then a summary with call timings.

import { readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";
import { replayRecord } from "../src/replay-record";
import { loadRules } from "../src/rules";

const wasm = readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url));
const rules = loadRules(wasm);

function files(path: string): string[] {
  if (!statSync(path).isDirectory()) return [path];
  return readdirSync(path).filter((name) => name.endsWith(".sgf")).sort().map((name) => join(path, name));
}

const paths = process.argv.slice(2).flatMap(files);
if (!paths.length) {
  console.error("usage: bun tools/replay.ts <dir-or-sgf>...");
  process.exit(2);
}

let games = 0, moves = 0, agreed = 0, captures = 0, stones = 0, capped = 0;
const started = performance.now();
for (const path of paths) {
  games++;
  try {
    const report = replayRecord(rules, readFileSync(path, "utf8"));
    moves += report.moves;
    if (!report.finished) capped++;
    for (const step of report.steps) {
      captures += step.removed.length;
      stones = Math.max(stones, step.after.stones.length);
    }
    if (report.problems.length) console.log(`${path}: ${report.problems.join("; ")}`);
    else agreed++;
  } catch (error) {
    console.log(`${path}: ${(error as Error).message}`);
  }
}
const playMs = (performance.now() - started) / Math.max(1, moves);

// Hover cost: one snap and one play per frame, on the last position of the longest game.
let worst = { text: "", moves: -1 };
for (const path of paths) {
  const text = readFileSync(path, "utf8");
  const count = (text.match(/;[BW]\[/g) ?? []).length;
  if (count > worst.moves) worst = { text, moves: count };
}
const late = replayRecord(rules, worst.text).steps.at(-3)?.after;
let hoverMs = NaN;
if (late) {
  const position = { ...late, passes: 0, finished: false };
  const rounds = 200;
  const t0 = performance.now();
  for (let k = 0; k < rounds; k++) {
    const at = rules.snap(position, { x: (k * 0.618) % 1, y: (k * 0.414) % 1 });
    if (at) rules.play(position, { kind: "place", x: at.x, y: at.y });
  }
  hoverMs = (performance.now() - t0) / rounds;
  console.log(`hover: ${hoverMs.toFixed(2)} ms per snap+play on a ${position.stones.length}-stone position`);
}

console.log(`${agreed}/${games} games agree with their records (${capped} stopped at a move cap, not two passes); ${moves} moves, ${captures} stones removed, up to ${stones} stones on the board`);
console.log(`replay: ${playMs.toFixed(2)} ms per move including parsing`);
process.exit(agreed === games ? 0 : 1);
