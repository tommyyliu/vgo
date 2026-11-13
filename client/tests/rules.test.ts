// The rules door against recorded games, and the replay check against games altered
// to disagree with it: a check that never fails proves nothing.

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { replayRecord } from "../src/replay-record";
import { emptyPosition, isStep, loadRules, type Position } from "../src/rules";
import { parse, setupOf } from "../src/sgf";

const rules = loadRules(readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url)));
const dir = new URL("./fixtures/", import.meta.url).pathname;
const records = readdirSync(dir).filter((name) => name.endsWith(".sgf")).map((name) => ({ name, text: readFileSync(join(dir, name), "utf8") }));

/** The `n`th move value of a record, as written: "0.40234375,0.53515625". */
function nthMove(text: string, n: number): RegExpExecArray {
  const pattern = /;([BW])\[([^\]]*)\]/g;
  let match: RegExpExecArray | null = null;
  for (let k = 0; k <= n; k++) match = pattern.exec(text);
  if (!match) throw new Error(`record has no move ${n}`);
  return match;
}

describe("recorded games", () => {
  test("there are fixtures, natural and capped", () => {
    expect(records.length).toBeGreaterThanOrEqual(5);
    expect(records.some((r) => r.name.includes("capped"))).toBe(true);
  });

  for (const { name, text } of records) {
    test(`${name} replays to its recorded result`, () => {
      const report = replayRecord(rules, text);
      expect(report.problems).toEqual([]);
      expect(report.moves).toBeGreaterThan(10);
      expect(report.finished).toBe(!name.includes("capped"));
    });
  }
});

describe("the replay check fails when the record is wrong", () => {
  const { text } = records.find((r) => r.name.includes("natural"))!;

  test("a different result", () => {
    const flipped = text.replace(/RE\[([BW])\+/, (_, w: string) => `RE[${w === "B" ? "W" : "B"}+`);
    expect(flipped).not.toBe(text);
    expect(replayRecord(rules, flipped).problems.join()).toContain("recorded");
  });

  test("a different margin", () => {
    const shifted = text.replace(/RE\[([BW])\+([0-9.]+)\]/, (_, w: string, m: string) => `RE[${w}+${(Number(m) + 0.01).toFixed(3)}]`);
    expect(replayRecord(rules, shifted).problems.join()).toContain("recorded");
  });

  test("a stone placed on top of an earlier one", () => {
    const first = nthMove(text, 0), third = nthMove(text, 2);
    expect(first[1]).toBe(third[1]);
    const stacked = text.slice(0, third.index) + `;${third[1]}[${first[2]}]` + text.slice(third.index + third[0].length);
    expect(replayRecord(rules, stacked).problems).toEqual(["move 3: refused (illegal-placement)"]);
  });

  test("the same side moving twice", () => {
    const second = nthMove(text, 1);
    const wrong = second[1] === "B" ? "W" : "B";
    const doubled = text.slice(0, second.index) + `;${wrong}[${second[2]}]` + text.slice(second.index + second[0].length);
    expect(replayRecord(rules, doubled).problems[0]).toContain("was to move");
  });

  test("a coordinate rounded to 5 decimals can make a recorded move illegal", () => {
    // Every snapped move in the archive sits within 1e-6 of a neighbour's reach; find one
    // whose rounding the rules refuse, which is why records keep full precision.
    let refusedSomewhere = false;
    for (const record of records) {
      const rounded = record.text.replace(/;([BW])\[([0-9.]+),([0-9.]+)\]/g, (_, c: string, x: string, y: string) =>
        `;${c}[${Number(Number(x).toFixed(5))},${Number(Number(y).toFixed(5))}]`);
      if (replayRecord(rules, rounded).problems.length) refusedSomewhere = true;
    }
    expect(refusedSomewhere).toBe(true);
  });
});

describe("the door", () => {
  const start: Position = emptyPosition(1 / 18, 0.1);

  test("an empty board has no cells and is playable", () => {
    const analysis = rules.analyze(start);
    expect(analysis.cells).toEqual([]);
    expect(analysis.playable).toBe(true);
  });

  test("snap moves a point off a stone to a spot the rules accept", () => {
    const first = rules.play(start, { kind: "place", x: 0.5, y: 0.5 });
    if (!isStep(first)) throw new Error(first.refused);
    const inside = { x: 0.5 + 0.01, y: 0.5 };
    expect(isStep(rules.play(first.after, { kind: "place", ...inside }))).toBe(false);
    const snapped = rules.snap(first.after, inside)!;
    expect(Math.hypot(snapped.x - 0.5, snapped.y - 0.5)).toBeGreaterThanOrEqual(2 / 18);
    expect(isStep(rules.play(first.after, { kind: "place", ...snapped }))).toBe(true);
  });

  test("a legal point snaps to itself", () => {
    expect(rules.snap(start, { x: 0.3, y: 0.7 })).toEqual({ x: 0.3, y: 0.7 });
  });

  test("two passes end the game, and the door says so", () => {
    const one = rules.play(start, { kind: "pass" });
    if (!isStep(one)) throw new Error(one.refused);
    const two = rules.play(one.after, { kind: "pass" });
    if (!isStep(two)) throw new Error(two.refused);
    expect([one.after.finished, two.after.finished]).toEqual([false, true]);
    expect(rules.play(two.after, { kind: "pass" })).toEqual({ refused: "finished" });
  });

  test("overlapping setup stones are refused, not played on", () => {
    const bad: Position = { ...start, stones: [{ x: 0.5, y: 0.5, c: "B" }, { x: 0.51, y: 0.5, c: "W" }] };
    expect(rules.analyze(bad).playable).toBe(false);
    expect(rules.play(bad, { kind: "pass" })).toEqual({ refused: "invalid-position" });
  });

  test("the front is read from cells: every edge across stones names a real neighbour", () => {
    const { text } = records[0]!;
    const report = replayRecord(rules, text);
    const step = report.steps[Math.floor(report.steps.length / 2)]!;
    const n = step.after.stones.length;
    for (const cell of step.analysis.cells) for (const edge of cell.edges) {
      expect(edge.across).toBeGreaterThanOrEqual(-1);
      expect(edge.across).toBeLessThan(n);
    }
    expect(setupOf(parse(text)).radius).toBe(step.after.radius);
  });
});
