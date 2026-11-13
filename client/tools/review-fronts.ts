// Builds review 1: four recorded positions in the wooden look, with the three fronts.
//
//   bun tools/review-fronts.ts <archive dir> <out.html>
//
// <archive dir> is artifacts/old/main/experiments-2026-10-14. The page is static: the
// boards are painted here, so the published page needs no wasm.

import { readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { frontChoices, wood, type Look } from "../src/looks";
import { boardModel, type BoardModel } from "../src/model";
import { paintFront, paintLand, paintStones } from "../src/paint";
import { replayRecord } from "../src/replay-record";
import { loadRules, type Rules } from "../src/rules";

const [archive, out] = process.argv.slice(2);
if (!archive || !out) throw new Error("usage: bun tools/review-fronts.ts <archive dir> <out.html>");
const rules = loadRules(readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url)));

interface Pick {
  readonly title: string;
  readonly about: string;
  readonly file: string;
  /** Show the board after this many moves; 0 means the last. */
  readonly moves: number;
}

const picks: Pick[] = [
  { title: "An opening", about: "Ten moves into a close game. Most land is still open.", file: "norm1-sim-r18-sgf/res-ab-game029-candW-natural.sgf", moves: 10 },
  { title: "A middle game", about: "The same game at move 36, with settled land on both sides.", file: "norm1-sim-r18-sgf/res-ab-game029-candW-natural.sgf", moves: 36 },
  { title: "Before a capture", about: "One move before White takes eight stones.", file: "norm1-sim-r18-sgf/res-ab-game028-candB-natural.sgf", moves: 39 },
  { title: "A finished game, small stones", about: "Two passes ended it; 188 stones at radius 1/38.", file: "norm1-sim-r38-sgf/res-ab-game018-candW-natural.sgf", moves: 0 },
];

function modelAt(rules: Rules, pick: Pick): { model: BoardModel; moves: number; total: number; result: string } {
  const report = replayRecord(rules, readFileSync(join(archive!, pick.file), "utf8"));
  if (report.problems.length) throw new Error(`${pick.file}: ${report.problems.join("; ")}`);
  const total = report.steps.length;
  const moves = pick.moves || total;
  const step = report.steps[moves - 1];
  if (!step) throw new Error(`${pick.file} has no move ${moves}`);
  const final = report.steps.at(-1)!.analysis;
  return { model: boardModel(rules, step), moves, total, result: `${final.winner ?? "tie"}+${final.margin.toFixed(3)}` };
}

function board(model: BoardModel, look: Look, px: number, fronts: (keyof typeof frontChoices)[], label: string): string {
  const view = { px };
  const layers = fronts.map((name) => `<g class="front ${name}">${paintFront(model, frontChoices[name].front, view)}</g>`).join("");
  return `<div class="board"><svg viewBox="0 0 1 1" role="img" aria-label="${label}">${paintLand(model, look)}${layers}${paintStones(model, look, view)}</svg></div>`;
}

const names = Object.keys(frontChoices) as (keyof typeof frontChoices)[];
let positions = "";
let dense: BoardModel | null = null;
for (const pick of picks) {
  const { model, moves, total, result } = modelAt(rules, pick);
  if (pick.moves === 0) dense = model;
  const source = `${pick.file.split("/").pop()} · move ${moves} of ${total} · ${model.position.stones.length} stones · final result ${result}`;
  positions += `<section><h2>${pick.title}</h2><p class="mut">${pick.about}</p>${board(model, wood, 720, names, pick.title)}<p class="source">${source}</p></section>\n`;
}
if (!dense) throw new Error("no finished game picked");
const phones = names.map((name) =>
  `<figure>${board(dense!, wood, 360, [name], `${frontChoices[name].label}, phone size`)}<figcaption>${frontChoices[name].label}</figcaption></figure>`).join("");

const sw = (color: string) => `<i class="sw" style="background:${color}"></i>`;
const legend = [
  `<span>${sw(wood.B.open)}${sw(wood.W.open)} open land: a stone could still take it</span>`,
  `<span>${sw(wood.B.settled)}${sw(wood.W.settled)} settled land: no stone can</span>`,
].join("");

const archiveGames = Number(process.env.REPLAYED_GAMES ?? 672);
const about = Object.fromEntries(names.map((name) => [name, frontChoices[name].about]));
const page = readFileSync(new URL("../pages/front-review.html", import.meta.url), "utf8")
  .replace("<!--LEGEND-->", legend)
  .replace("<!--RULES-->", `All ${archiveGames} archived arena games replay move by move through the Rust rules, compiled to WebAssembly, to their recorded results. Every board here is drawn from those rules.`)
  .replace("<!--POSITIONS-->", positions)
  .replace("<!--PHONES-->", phones)
  .replace("/*ABOUT*/{}", JSON.stringify(about));
writeFileSync(out, page);
console.log(`${out}: ${(page.length / 1024).toFixed(0)} KB`);
