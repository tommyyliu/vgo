// Replays a recorded game's main line through the rules and says whether the rules agree
// with the record: every move legal and by the side to move, with the recorded result.

import { createReplay, Refused } from "./replay";
import type { Rules, Step } from "./rules";
import { resultOf, parse } from "./sgf";
import { fromSgf, line } from "./tree";

export interface RecordCheck {
  readonly moves: number;
  readonly steps: readonly Step[];
  /** False for a game stopped at a move cap rather than ended by two passes. */
  readonly finished: boolean;
  /** Empty when the rules agree with the record. */
  readonly problems: readonly string[];
}

export function checkRecord(rules: Rules, text: string): RecordCheck {
  const { tree, problems: unread } = fromSgf(text);
  const problems = [...unread];
  const replay = createReplay(rules);
  const ids = line(tree, tree.root).slice(1);
  const steps: Step[] = [];
  for (const [k, id] of ids.entries()) {
    try {
      steps.push(replay.step(tree, id)!);
    } catch (error) {
      if (!(error instanceof Refused)) throw error;
      problems.push(error.reason === "wrong-colour" ? `move ${k + 1}: played out of turn` : `move ${k + 1}: refused (${error.reason})`);
      break;
    }
  }
  const last = steps.at(-1);
  if (!problems.length && last) {
    const recorded = resultOf(parse(text));
    const { winner, margin } = last.analysis;
    if (recorded && (winner !== recorded.winner || Math.abs(margin - recorded.margin) > 0.0005 + 1e-9)) {
      problems.push(`result ${winner ?? "tie"}+${margin.toFixed(3)}, recorded ${recorded.winner ?? "tie"}+${recorded.margin}`);
    }
    if (steps.some((step) => step.analysis.unclassifiedEdges)) problems.push("the geometry left an edge unclassified");
  }
  return {
    moves: ids.length,
    steps,
    finished: last?.finished ?? false,
    problems,
  };
}
