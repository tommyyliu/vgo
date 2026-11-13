// Replays a recorded game's main line through the rules door and says whether the
// rules agree with the record: every move legal, by the side to move, ending the game,
// with the recorded result.

import { isStep, type Rules, type Step } from "./rules";
import { moveOf, parse, resultOf, setupOf, type SgfNode } from "./sgf";

export interface ReplayReport {
  readonly moves: number;
  readonly steps: readonly Step[];
  /** False for a game stopped at a move cap rather than ended by two passes. */
  readonly finished: boolean;
  /** Empty when the rules agree with the record. */
  readonly problems: readonly string[];
}

export function replayRecord(rules: Rules, text: string): ReplayReport {
  const problems: string[] = [];
  const steps: Step[] = [];
  const root = parse(text);
  let position = setupOf(root);
  let moves = 0;
  for (let node: SgfNode | undefined = root.children[0]; node; node = node.children[0]) {
    const move = moveOf(node);
    if (!move) continue;
    moves++;
    if (move.color !== position.toMove) {
      problems.push(`move ${moves}: ${move.color} played but ${position.toMove} was to move`);
      break;
    }
    const result = rules.play(position, move);
    if (!isStep(result)) {
      problems.push(`move ${moves}: refused (${result.refused})`);
      break;
    }
    steps.push(result);
    position = result.after;
  }
  if (!problems.length) {
    const recorded = resultOf(root);
    const last = steps.at(-1);
    if (recorded && last) {
      const { winner, margin } = last.analysis;
      if (winner !== recorded.winner || Math.abs(margin - recorded.margin) > 0.0005 + 1e-9) {
        problems.push(`result ${winner ?? "tie"}+${margin.toFixed(3)}, recorded ${recorded.winner ?? "tie"}+${recorded.margin}`);
      }
    }
    for (const step of steps) {
      if (step.analysis.unclassifiedEdges) problems.push(`unclassified edges after a ${step.move.color} move`);
    }
  }
  return { moves, steps, finished: position.finished, problems };
}
