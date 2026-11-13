// SGF, as Voronoi Go writes it: (;FF[4]GM[VGO]SZ[1]RA[r]KM[k]PL[B];B[x,y];W[];...)
//
// `parse` reads any SGF into a tree of property maps and knows nothing of the game.
// `setupOf` and `moveOf` read the Voronoi Go properties off a node. Coordinates are
// kept exactly as written: `Number` reads the shortest round-trip decimal the Rust
// recorder writes back to the identical double, and legality turns on 1e-6.

import type { Color, PlayedMove, Position, Ruleset, Stone } from "./rules";

export interface SgfNode {
  readonly props: ReadonlyMap<string, readonly string[]>;
  /** The next node of this line first, then any variations. */
  readonly children: readonly SgfNode[];
}

export class SgfError extends Error {}

/** Parses the first game tree in `text`. */
export function parse(text: string): SgfNode {
  let at = 0;
  const fail = (what: string): never => {
    throw new SgfError(`${what} at offset ${at}`);
  };
  const skipSpace = () => {
    while (at < text.length && /\s/.test(text[at]!)) at++;
  };

  function value(): string {
    if (text[at] !== "[") fail("expected [");
    at++;
    let out = "";
    while (at < text.length && text[at] !== "]") {
      if (text[at] === "\\") at++;
      out += text[at] ?? "";
      at++;
    }
    if (text[at] !== "]") fail("unterminated value");
    at++;
    return out;
  }

  function node(): Map<string, string[]> {
    at++; // ;
    const props = new Map<string, string[]>();
    for (;;) {
      skipSpace();
      const match = /^[A-Za-z]+/.exec(text.slice(at, at + 16));
      if (!match) return props;
      const ident = match[0].replace(/[a-z]/g, "");
      at += match[0].length;
      skipSpace();
      const values: string[] = [];
      while (text[at] === "[") {
        values.push(value());
        skipSpace();
      }
      if (!values.length) fail(`property ${ident} has no value`);
      props.set(ident, (props.get(ident) ?? []).concat(values));
    }
  }

  // A game tree: "(" then a sequence of nodes, then nested game trees, then ")".
  function gameTree(): SgfNode {
    skipSpace();
    if (text[at] !== "(") fail("expected (");
    at++;
    skipSpace();
    const sequence: Map<string, string[]>[] = [];
    while (text[at] === ";") {
      sequence.push(node());
      skipSpace();
    }
    if (!sequence.length) fail("empty game tree");
    const variations: SgfNode[] = [];
    while (text[at] === "(") {
      variations.push(gameTree());
      skipSpace();
    }
    if (text[at] !== ")") fail("expected )");
    at++;
    let children = variations;
    for (let k = sequence.length - 1; k >= 0; k--) {
      const built: SgfNode = { props: sequence[k]!, children };
      children = [built];
    }
    return children[0]!;
  }

  return gameTree();
}

function one(props: ReadonlyMap<string, readonly string[]>, key: string): string | undefined {
  return props.get(key)?.[0];
}

function number(text: string, what: string): number {
  const value = Number(text);
  if (text.trim() === "" || !Number.isFinite(value)) throw new SgfError(`${what} is not a number: ${JSON.stringify(text)}`);
  return value;
}

function point(text: string, c: Color): Stone {
  const parts = text.split(",");
  if (parts.length !== 2) throw new SgfError(`not a point: ${JSON.stringify(text)}`);
  return { x: number(parts[0]!, "x"), y: number(parts[1]!, "y"), c };
}

/** The position before move 1, from a root node. */
export function setupOf(root: SgfNode): Position {
  const props = root.props;
  if (one(props, "GM") !== "VGO") throw new SgfError("not a Voronoi Go record (GM[VGO] missing)");
  const radius = one(props, "RA");
  if (radius === undefined) throw new SgfError("no stone radius (RA[])");
  const rules = one(props, "RU") ?? "vgo";
  if (rules !== "vgo" && rules !== "official") throw new SgfError(`unknown ruleset RU[${rules}]`);
  const toMove = one(props, "PL") ?? "B";
  if (toMove !== "B" && toMove !== "W") throw new SgfError(`PL[${toMove}] is not B or W`);
  const stones = [
    ...(props.get("AB") ?? []).map((text) => point(text, "B")),
    ...(props.get("AW") ?? []).map((text) => point(text, "W")),
  ];
  return {
    ruleset: rules as Ruleset,
    radius: number(radius, "RA"),
    komi: number(one(props, "KM") ?? "0", "KM"),
    stones,
    toMove,
    passes: 0,
  };
}

/** The move a node makes, or null for a node with none (the root, or a comment). */
export function moveOf(node: SgfNode): PlayedMove | null {
  for (const c of ["B", "W"] as const) {
    const text = one(node.props, c);
    if (text === undefined) continue;
    if (text === "") return { kind: "pass", color: c };
    const stone = point(text, c);
    return { kind: "place", x: stone.x, y: stone.y, color: c };
  }
  return null;
}

/** The recorded result, from RE[] on any node of the main line: "W+0.653" → W by 0.653. */
export function resultOf(root: SgfNode): { winner: Color | null; margin: number } | null {
  for (let node: SgfNode | undefined = root; node; node = node.children[0]) {
    const text = one(node.props, "RE");
    if (text === undefined) continue;
    if (text === "0" || text === "Draw") return { winner: null, margin: 0 };
    const match = /^([BW])\+([0-9.]+)$/.exec(text);
    return match ? { winner: match[1] as Color, margin: Number(match[2]) } : null;
  }
  return null;
}
