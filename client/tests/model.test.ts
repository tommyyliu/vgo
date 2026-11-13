// The board model reshapes the rules' output; these check it reshapes without losing
// or inventing anything.

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { boardModel, frontSegments, joinSegments } from "../src/model";
import { settledLoops } from "../src/settled";
import { paintBoard } from "../src/paint";
import { wood } from "../src/looks";
import { checkRecord } from "../src/check-record";
import type { Step } from "../src/rules";
import { testRules } from "./wasm";

const rules = testRules();
const dir = new URL("./fixtures/", import.meta.url).pathname;
const steps: Step[] = readdirSync(dir).filter((n) => n.endsWith(".sgf")).flatMap((name) => {
  const all = checkRecord(rules, readFileSync(join(dir, name), "utf8")).steps;
  return [all[Math.floor(all.length / 3)]!, all[Math.floor((2 * all.length) / 3)]!];
});
const length = (line: readonly (readonly [number, number])[]) =>
  line.slice(1).reduce((sum, p, k) => sum + Math.hypot(p[0] - line[k]![0], p[1] - line[k]![1]), 0);

describe("front", () => {
  test("each front edge is seen the same way from both of its cells", () => {
    for (const step of steps) {
      const { cells } = step.analysis;
      cells.forEach((cell, i) => {
        for (const edge of cell.edges) {
          if (edge.across < 0) continue;
          const back = cells[edge.across]!.edges.find((e) => e.across === i);
          expect(back).toBeDefined();
          const ends = [[edge.a, edge.b], [edge.b, edge.a]].some(([p, q]) =>
            Math.hypot(p!.x - back!.a.x, p!.y - back!.a.y) < 1e-9 && Math.hypot(q!.x - back!.b.x, q!.y - back!.b.y) < 1e-9);
          expect(ends).toBe(true);
        }
      });
    }
  });

  test("only edges between different colours are front", () => {
    for (const step of steps) {
      const segments = frontSegments(step.after, step.analysis);
      let expected = 0;
      step.analysis.cells.forEach((cell, i) => {
        for (const e of cell.edges) if (e.across > i && step.after.stones[e.across]!.c !== step.after.stones[i]!.c) expected++;
      });
      expect(segments.length).toBe(expected);
      expect(expected).toBeGreaterThan(0);
    }
  });

  test("joining keeps every bit of length", () => {
    for (const step of steps) {
      const segments = frontSegments(step.after, step.analysis);
      const joined = joinSegments(segments);
      const before = segments.reduce((sum, s) => sum + length(s), 0);
      const after = joined.reduce((sum, line) => sum + length(line), 0);
      expect(Math.abs(before - after)).toBeLessThan(1e-9);
      expect(joined.length).toBeLessThan(segments.length);
    }
  });
});

describe("settled shape", () => {
  test("every stone's own loop contains the stone", () => {
    // R_s always contains the disc of radius r around s: no legal centre is nearer than 2r.
    for (const step of steps) {
      const model = boardModel(step);
      const { stones, radius } = step.after;
      expect(model.settled.length).toBe(stones.length);
      model.settled.forEach((loop, i) => {
        const s = stones[i]!;
        const nearest = Math.min(...loop.map((p) => Math.hypot(p[0] - s.x, p[1] - s.y)));
        expect(nearest).toBeGreaterThanOrEqual(radius * 0.999);
      });
    }
  });
});

describe("paint", () => {
  test("ids differ between boards painted into one page", () => {
    const model = boardModel(steps[0]!);
    const ids = (svg: string) => [...svg.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
    const first = ids(paintBoard(model, wood, { px: 720 })), second = ids(paintBoard(model, wood, { px: 720 }));
    expect(first.length).toBeGreaterThan(0);
    expect(first.some((id) => second.includes(id))).toBe(false);
  });
});

describe("settled shape port", () => {
  // The port must draw what js-app/src/geometry/settled-contour.js draws. Given the same
  // legal-set corners and the original's tolerance, every loop should match point for point.
  const root = new URL("../../js-app/src/", import.meta.url).pathname;
  const files = ["engine/numeric.js", "engine/model.js", "geometry/voronoi.js", "geometry/legal-set.js", "geometry/settled-contour.js"];
  const g = globalThis as unknown as { VGO?: Record<string, any> };
  const saved = g.VGO;
  g.VGO = undefined;
  (0, eval)(files.map((f) => readFileSync(join(root, f), "utf8")).join("\n"));
  const original = g.VGO!;
  g.VGO = saved;

  test("matches the original on recorded positions", () => {
    let points = 0;
    for (const step of steps) {
      const p = step.after;
      const vertices = step.analysis.legalVertices;
      const ported = settledLoops(p, step.analysis, false, original.settledContour.TOLERANCE);
      const js = original.settledContour.compute(
        original.model.createPosition({ radius: p.radius, stones: p.stones, toMove: p.toMove, komi: p.komi }),
        vertices.map((v) => [v.x, v.y]),
      ).loops as number[][][];
      expect(ported.length).toBe(js.length);
      ported.forEach((loop, i) => {
        expect(loop.length).toBe(js[i]!.length);
        loop.forEach((q, k) => {
          expect(Math.abs(q[0] - js[i]![k]![0]!)).toBeLessThan(1e-12);
          expect(Math.abs(q[1] - js[i]![k]![1]!)).toBeLessThan(1e-12);
          points++;
        });
      });
    }
    expect(points).toBeGreaterThan(10000);
  });
});
