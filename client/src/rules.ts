// The rules door: the only place the client learns a rule.
//
// Everything here is vgo-core, compiled to WebAssembly from crates/vgo-wasm/src/rules.rs:
// the same code self-play runs. The functions are stateless and take and return plain
// data, so a position can be cached by game-tree node with nothing held in wasm memory.

import { initSync, rulesAnalyze, rulesPlay, rulesSnap } from "../vendor/vgo_wasm.js";

export type Color = "B" | "W";
export type Ruleset = "vgo" | "official";
export interface Point { readonly x: number; readonly y: number }
export interface Stone { readonly x: number; readonly y: number; readonly c: Color }

export interface Position {
  readonly ruleset: Ruleset;
  readonly radius: number;
  readonly komi: number;
  readonly stones: readonly Stone[];
  readonly toMove: Color;
  /** Consecutive passes so far. Whether that ends the game is Rust's call: see `Step.finished`. */
  readonly passes: number;
}

export interface Edge {
  readonly a: Point;
  readonly b: Point;
  /** The stone across this edge, or -1 at the board's edge. */
  readonly across: number;
}

export interface Cell {
  /** The stone's Voronoi cell, clipped to the board. */
  readonly polygon: readonly Point[];
  readonly edges: readonly Edge[];
  readonly area: number;
}

export interface Analysis {
  readonly playable: boolean;
  /** One per stone, in position order. */
  readonly cells: readonly Cell[];
  /** Group id per stone. */
  readonly group: readonly number[];
  readonly settledGroups: readonly number[];
  /** Corners of the legal set, which the settled shape is built from. */
  readonly legalVertices: readonly Point[];
  /** Area held, before komi. */
  readonly score: { readonly B: number; readonly W: number };
  /** Komi applied; null on a tie. */
  readonly winner: Color | null;
  readonly margin: number;
  /** Edges the geometry could not classify. Zero on every sound position. */
  readonly unclassifiedEdges: number;
  /** No stone fits anywhere, so every point is settled. */
  readonly legalSetEmpty: boolean;
}

export type Move = { readonly kind: "place"; readonly x: number; readonly y: number } | { readonly kind: "pass" };
export type PlayedMove = Move & { readonly color: Color };

export interface Step {
  readonly before: Position;
  readonly after: Position;
  readonly move: PlayedMove;
  /** Whether the move ended the game. */
  readonly finished: boolean;
  /** Every stone the move took off: the opponent's first, then the mover's own. */
  readonly removed: readonly Stone[];
  /** Of `after`. */
  readonly analysis: Analysis;
}

export type Refusal = "finished" | "invalid-position" | "illegal-placement" | "self-capture";

export interface Rules {
  analyze(position: Position): Analysis;
  /** The legal stone centre nearest `at`, or null when no stone fits anywhere. */
  snap(position: Position, at: Point): Point | null;
  play(position: Position, move: Move): Step | { readonly refused: Refusal };
}

export function emptyPosition(radius: number, komi: number, ruleset: Ruleset = "vgo"): Position {
  return { ruleset, radius, komi, stones: [], toMove: "B", passes: 0 };
}

interface PlayIo {
  refused?: Refusal;
  after?: Position;
  finished: boolean;
  removed: Stone[];
  analysis?: Analysis & { winner?: Color };
}

function analysisFrom(io: Analysis & { winner?: Color | null }): Analysis {
  return { ...io, winner: io.winner ?? null };
}

/** Instantiates the wasm from its bytes. Synchronous, so it works the same in bun and the browser. */
export function loadRules(wasm: BufferSource): Rules {
  initSync({ module: wasm });
  return {
    analyze: (position) => analysisFrom(rulesAnalyze(position)),
    snap: (position, at) => rulesSnap(position, at.x, at.y) ?? null,
    play: (position, move) => {
      const io = rulesPlay(position, move) as PlayIo;
      if (io.refused !== undefined) return { refused: io.refused };
      if (io.after === undefined || io.analysis === undefined) throw new Error("rulesPlay returned neither a result nor a refusal");
      return {
        before: position,
        after: io.after,
        move: { ...move, color: position.toMove },
        finished: io.finished,
        removed: io.removed,
        analysis: analysisFrom(io.analysis),
      };
    },
  };
}

export function isStep(result: Step | { readonly refused: Refusal }): result is Step {
  return !("refused" in result);
}
