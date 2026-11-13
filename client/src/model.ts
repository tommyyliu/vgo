// The board model: everything paint reads, reshaped from the rules door's output.
// It decides nothing.

import type { Analysis, Color, PlayedMove, Position, Rules, Stone, Step } from "./rules";
import { settledLoops, type Loop } from "./settled";

export type Polyline = readonly (readonly [number, number])[];

export interface BoardModel {
  readonly position: Position;
  readonly analysis: Analysis;
  /** Edges between differently coloured cells, joined into runs. */
  readonly front: readonly Polyline[];
  /** Closed loops whose union (nonzero) is the settled region. */
  readonly settled: readonly Loop[];
  /** Each side's cells, for filling land and clipping the settled shape. */
  readonly land: Readonly<Record<Color, readonly Polyline[]>>;
  readonly last: { readonly move: PlayedMove; readonly removed: readonly Stone[] } | null;
}

const JOIN_TOLERANCE = 1e-9;

/** Chains segments that share an end into polylines, so a stroke runs on across corners. */
export function joinSegments(segments: readonly (readonly [readonly [number, number], readonly [number, number]])[]): Polyline[] {
  const same = (a: readonly [number, number], b: readonly [number, number]) =>
    Math.abs(a[0] - b[0]) <= JOIN_TOLERANCE && Math.abs(a[1] - b[1]) <= JOIN_TOLERANCE;
  const pending = segments.map((s) => [s[0], s[1]] as [readonly [number, number], readonly [number, number]]);
  const lines: Polyline[] = [];
  while (pending.length) {
    const first = pending.pop()!;
    const line: (readonly [number, number])[] = [first[0], first[1]];
    for (let grew = true; grew;) {
      grew = false;
      for (let k = 0; k < pending.length; k++) {
        const [a, b] = pending[k]!;
        const head = line[0]!, tail = line[line.length - 1]!;
        if (same(a, tail)) line.push(b);
        else if (same(b, tail)) line.push(a);
        else if (same(b, head)) line.unshift(a);
        else if (same(a, head)) line.unshift(b);
        else continue;
        pending.splice(k, 1);
        grew = true;
        break;
      }
    }
    lines.push(line);
  }
  return lines;
}

/** Edges whose two stones differ in colour, each once. */
export function frontSegments(position: Position, analysis: Analysis) {
  const out: [readonly [number, number], readonly [number, number]][] = [];
  analysis.cells.forEach((cell, i) => {
    for (const edge of cell.edges) {
      const j = edge.across;
      if (j <= i) continue; // the board's edge (-1), or a pair already taken from the other side
      if (position.stones[i]!.c === position.stones[j]!.c) continue;
      out.push([[edge.a.x, edge.a.y], [edge.b.x, edge.b.y]]);
    }
  });
  return out;
}

export function boardModel(rules: Rules, source: Step | { readonly position: Position; readonly analysis: Analysis }): BoardModel {
  const isStep = "after" in source;
  const position = isStep ? source.after : source.position;
  const analysis = source.analysis;
  const legalSetEmpty = position.stones.length > 0 && rules.snap(position, position.stones[0]!) === null;
  const land: Record<Color, Polyline[]> = { B: [], W: [] };
  analysis.cells.forEach((cell, i) => {
    land[position.stones[i]!.c].push(cell.polygon.map((p) => [p.x, p.y] as const));
  });
  return {
    position,
    analysis,
    front: joinSegments(frontSegments(position, analysis)),
    settled: settledLoops(position, analysis, legalSetEmpty),
    land,
    last: isStep ? { move: source.move, removed: source.removed } : null,
  };
}
