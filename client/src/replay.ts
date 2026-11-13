// Replay: the rules applied to a game tree, cached by node id.
//
// A node's path never changes (tree.ts), so its step is computed once and kept. Positions
// live only here, recomputed through Rust, never in the tree or a file.

import { isStep, type Position, type Refusal, type Rules, type Step } from "./rules";
import { node, remove, setupPosition, type GameTree, type NodeId } from "./tree";

export interface Replay {
  /** The step that reached `id`; null at the root. Throws if the rules refuse a move on its path. */
  step(tree: GameTree, id: NodeId): Step | null;
  /** The position at `id`. */
  position(tree: GameTree, id: NodeId): Position;
}

export class Refused extends Error {
  constructor(readonly node: NodeId, readonly reason: Refusal | "wrong-colour") {
    super(`move at node ${node} refused: ${reason}`);
  }
}

export function createReplay(rules: Rules): Replay {
  const steps = new Map<NodeId, Step>();
  const replay: Replay = {
    step(tree, id) {
      const at = node(tree, id);
      if (at.parent === null) return null;
      const cached = steps.get(id);
      if (cached) return cached;
      const before = replay.position(tree, at.parent);
      const move = at.move!;
      if (move.color !== before.toMove) throw new Refused(id, "wrong-colour");
      const result = rules.play(before, move);
      if (!isStep(result)) throw new Refused(id, result.refused);
      steps.set(id, result);
      return result;
    },
    position(tree, id) {
      const at = node(tree, id);
      return at.parent === null ? setupPosition(tree.setup) : replay.step(tree, id)!.after;
    },
  };
  return replay;
}

/**
 * Replays every node of a loaded tree. A branch is cut at the first move the rules refuse,
 * and reported; nothing is repaired silently.
 */
export function checkTree(replay: Replay, tree: GameTree): { tree: GameTree; problems: string[] } {
  const problems: string[] = [];
  const stack = [tree.root];
  let checked = tree;
  while (stack.length) {
    const id = stack.pop()!;
    try {
      replay.step(checked, id);
      stack.push(...node(checked, id).children);
    } catch (error) {
      if (!(error instanceof Refused)) throw error;
      problems.push(`move ${depth(checked, id)} cut: ${error.reason}`);
      checked = remove(checked, id);
    }
  }
  return { tree: checked, problems };
}

function depth(tree: GameTree, id: NodeId): number {
  let n = 0;
  for (let at = node(tree, id); at.parent !== null; at = node(tree, at.parent)) n++;
  return n;
}
