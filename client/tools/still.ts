// Renders one position of a record to SVG: bun tools/still.ts <sgf> <ply> <gap|ink|core> <px> <out.svg>
import { readFileSync, writeFileSync } from "fs";
import { frontChoices, wood } from "../src/looks";
import { boardModel } from "../src/model";
import { paintBoard, svgFile } from "../src/paint";
import { replayRecord } from "../src/replay-record";
import { loadRules } from "../src/rules";

const [path, plyText, frontName, pxText, out] = process.argv.slice(2);
if (!path || !plyText || !frontName || !pxText || !out) throw new Error("usage: bun tools/still.ts <sgf> <ply> <gap|ink|core> <px> <out.svg>");
const rules = loadRules(readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url)));
const steps = replayRecord(rules, readFileSync(path, "utf8")).steps;
const ply = Math.min(Number(plyText), steps.length);
const step = steps[ply - 1];
if (!step) throw new Error(`no move ${ply}`);
const choice = frontChoices[frontName as keyof typeof frontChoices];
if (!choice) throw new Error(`front must be gap, ink or core`);
const px = Number(pxText);
const t0 = performance.now();
const model = boardModel(rules, step);
const t1 = performance.now();
writeFileSync(out, svgFile(paintBoard(model, { ...wood, front: choice.front }, { px }), px));
console.log(`move ${ply} of ${steps.length}: ${step.after.stones.length} stones, model ${(t1 - t0).toFixed(1)} ms, ${model.front.length} front runs, ${model.settled.length} settled loops`);
