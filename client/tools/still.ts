// Renders one position of a record to SVG in the wooden look: bun tools/still.ts <sgf> <moves> <px> <out.svg>
import { readFileSync, writeFileSync } from "fs";
import { wood } from "../src/looks";
import { boardModel } from "../src/model";
import { paintBoard, svgFile } from "../src/paint";
import { replayRecord } from "../src/replay-record";
import { loadRules } from "../src/rules";

const [path, movesText, pxText, out] = process.argv.slice(2);
if (!path || !movesText || !pxText || !out) throw new Error("usage: bun tools/still.ts <sgf> <moves> <px> <out.svg>");
const rules = loadRules(readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url)));
const steps = replayRecord(rules, readFileSync(path, "utf8")).steps;
const moves = Math.min(Number(movesText), steps.length);
const step = steps[moves - 1];
if (!step) throw new Error(`no move ${moves}`);
const px = Number(pxText);
writeFileSync(out, svgFile(paintBoard(boardModel(rules, step), wood, { px }), px));
console.log(`move ${moves} of ${steps.length}: ${step.after.stones.length} stones`);
