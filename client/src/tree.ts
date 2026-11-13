// The game tree: a setup and a tree of moves, each tagged with the colour that played it.
//
// Choices (see the design doc's Game tree tab):
//   - a flat, immutable map of nodes by id; an edit returns a new tree sharing what it left alone;
//   - a node's path never changes (edits add a leaf, remove a subtree, or reorder siblings),
//     so anything derived from a node, such as its position, can be cached by id for good;
//   - the tree knows no rules: replay.ts applies them.

import type { Color, PlayedMove, Position, Ruleset, Stone } from "./rules";
import { moveOf, parse, setupOf, SgfError, type SgfNode } from "./sgf";

export type NodeId = number;

export interface Setup {
  readonly ruleset: Ruleset;
  readonly radius: number;
  readonly komi: number;
  readonly stones: readonly Stone[];
  readonly toMove: Color;
}

export interface Node {
  readonly id: NodeId;
  /** Null only at the root. */
  readonly parent: NodeId | null;
  /** Null only at the root. */
  readonly move: PlayedMove | null;
  /** children[0] is the main continuation. */
  readonly children: readonly NodeId[];
  /** SGF properties the tree does not interpret, kept for the round trip. */
  readonly props: Readonly<Record<string, readonly string[]>>;
}

export interface GameTree {
  readonly setup: Setup;
  readonly root: NodeId;
  readonly nodes: ReadonlyMap<NodeId, Node>;
  readonly nextId: NodeId;
}

export function setupPosition(setup: Setup): Position {
  return { ...setup, passes: 0, finished: false };
}

export function create(setup: Setup, props: Node["props"] = {}): GameTree {
  const root: Node = { id: 0, parent: null, move: null, children: [], props };
  return { setup, root: 0, nodes: new Map([[0, root]]), nextId: 1 };
}

export function node(tree: GameTree, id: NodeId): Node {
  const found = tree.nodes.get(id);
  if (!found) throw new Error(`no node ${id}`);
  return found;
}

function sameMove(a: PlayedMove, b: PlayedMove): boolean {
  if (a.kind !== b.kind || a.color !== b.color) return false;
  return a.kind === "pass" || (b.kind === "place" && a.x === b.x && a.y === b.y);
}

/** A child of `at` playing `move`: the existing one if a child already plays it, else a new last child. */
export function add(tree: GameTree, at: NodeId, move: PlayedMove, props: Node["props"] = {}): { tree: GameTree; node: NodeId } {
  const parent = node(tree, at);
  const existing = parent.children.find((id) => sameMove(node(tree, id).move!, move));
  if (existing !== undefined) return { tree, node: existing };
  const id = tree.nextId;
  const nodes = new Map(tree.nodes);
  nodes.set(id, { id, parent: at, move, children: [], props });
  nodes.set(at, { ...parent, children: [...parent.children, id] });
  return { tree: { ...tree, nodes, nextId: id + 1 }, node: id };
}

/** Drops `id` and everything below it. The root cannot be removed. */
export function remove(tree: GameTree, id: NodeId): GameTree {
  const target = node(tree, id);
  if (target.parent === null) throw new Error("the root cannot be removed");
  const nodes = new Map(tree.nodes);
  const stack = [id];
  while (stack.length) {
    const next = stack.pop()!;
    stack.push(...node(tree, next).children);
    nodes.delete(next);
  }
  const parent = node(tree, target.parent);
  nodes.set(parent.id, { ...parent, children: parent.children.filter((child) => child !== id) });
  return { ...tree, nodes };
}

/** Makes the path to `id` the main line: at each ancestor, that branch moves to children[0]. */
export function promote(tree: GameTree, id: NodeId): GameTree {
  const nodes = new Map(tree.nodes);
  for (let child = node(tree, id); child.parent !== null; child = node(tree, child.parent)) {
    const parent = node(tree, child.parent);
    if (parent.children[0] === child.id) continue;
    nodes.set(parent.id, { ...parent, children: [child.id, ...parent.children.filter((c) => c !== child.id)] });
  }
  return { ...tree, nodes };
}

/** Root to `id`, inclusive. */
export function path(tree: GameTree, id: NodeId): NodeId[] {
  const out: NodeId[] = [];
  for (let at: Node | undefined = node(tree, id); at; at = at.parent === null ? undefined : node(tree, at.parent)) out.push(at.id);
  return out.reverse();
}

/** From `id` along children[0] to the end, inclusive. */
export function line(tree: GameTree, id: NodeId): NodeId[] {
  const out = [id];
  for (let at = node(tree, id); at.children.length; at = node(tree, at.children[0]!)) out.push(at.children[0]!);
  return out;
}

/** Moves from the root. */
export function ply(tree: GameTree, id: NodeId): number {
  return path(tree, id).length - 1;
}

/** The next (+1) or previous (-1) variation at the same branch point, or `id` itself at either end. */
export function sibling(tree: GameTree, id: NodeId, step: 1 | -1): NodeId {
  const at = node(tree, id);
  if (at.parent === null) return id;
  const siblings = node(tree, at.parent).children;
  return siblings[siblings.indexOf(id) + step] ?? id;
}

// SGF ------------------------------------------------------------------------------------

/** Root properties the setup is written from; everything else is kept as given. */
const SETUP_PROPS = new Set(["FF", "GM", "SZ", "RA", "KM", "RU", "PL", "AB", "AW"]);

function others(props: ReadonlyMap<string, readonly string[]>, skip: (key: string) => boolean): Node["props"] {
  const out: Record<string, readonly string[]> = {};
  for (const [key, values] of props) if (!skip(key)) out[key] = values;
  return out;
}

/** Reads a record. `problems` lists what could not be read; rule problems come from replay. */
export function fromSgf(text: string): { tree: GameTree; problems: string[] } {
  const problems: string[] = [];
  const root = parse(text);
  const position = setupOf(root);
  let tree = create(
    { ruleset: position.ruleset, radius: position.radius, komi: position.komi, stones: position.stones, toMove: position.toMove },
    others(root.props, (key) => SETUP_PROPS.has(key)),
  );
  const walk = (sgf: SgfNode, parent: NodeId) => {
    for (const child of sgf.children) {
      const move = moveOf(child);
      if (!move) {
        problems.push("a node with no move was skipped");
        walk(child, parent);
        continue;
      }
      const added = add(tree, parent, move, others(child.props, (key) => key === "B" || key === "W"));
      tree = added.tree;
      walk(child, added.node);
    }
  };
  walk(root, tree.root);
  return { tree, problems };
}

const escape = (value: string) => value.replace(/[\\\]]/g, (c) => "\\" + c);
// The shortest decimal that reads back to the same double, as the Rust recorder writes.
const number = (value: number) => String(value);

function propsText(props: Node["props"]): string {
  let out = "";
  for (const [key, values] of Object.entries(props)) out += key + values.map((v) => `[${escape(v)}]`).join("");
  return out;
}

/** Writes the dialect the Rust recorder writes, with variations as SGF subtrees. */
export function toSgf(tree: GameTree): string {
  const { setup } = tree;
  const stones = (c: Color) => setup.stones.filter((s) => s.c === c).map((s) => `[${number(s.x)},${number(s.y)}]`).join("");
  let head = `;FF[4]GM[VGO]SZ[1]RA[${number(setup.radius)}]KM[${number(setup.komi)}]`;
  if (setup.ruleset !== "vgo") head += `RU[${setup.ruleset}]`;
  head += `PL[${setup.toMove}]`;
  if (stones("B")) head += "AB" + stones("B");
  if (stones("W")) head += "AW" + stones("W");
  head += propsText(node(tree, tree.root).props);

  const sequence = (id: NodeId): string => {
    const at = node(tree, id);
    const move = at.move!;
    let out = `;${move.color}[${move.kind === "pass" ? "" : `${number(move.x)},${number(move.y)}`}]` + propsText(at.props);
    out += children(at);
    return out;
  };
  const children = (at: Node): string => {
    if (at.children.length === 1) return sequence(at.children[0]!);
    return at.children.map((child) => `(${sequence(child)})`).join("");
  };
  return `(${head}${children(node(tree, tree.root))})`;
}

export { SgfError };
