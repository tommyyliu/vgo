// Review 2: walk a recorded game, and choose the last-move mark and how captures show.

import { wood } from "../src/looks";
import { boardModel, type BoardModel } from "../src/model";
import { pageData, pageRules } from "../src/page";
import { paintBoard, paintLastMove, paintRemoved, type CaptureStyle, type LastMoveStyle } from "../src/paint";
import { checkTree, createReplay } from "../src/replay";
import { connectNotes } from "../src/review-notes";
import type { Color, Step } from "../src/rules";
import { fromSgf, line, type NodeId } from "../src/tree";

const rules = pageRules();
const record = pageData().record;
if (!record) throw new Error("built without a record");
const replay = createReplay(rules);
const { tree } = checkTree(replay, fromSgf(record).tree);
const ids: NodeId[] = line(tree, tree.root);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const board = $<HTMLDivElement>("board"), svg = board.querySelector("svg")!;
const slider = $<HTMLInputElement>("move"), label = $("move-label"), status = $("status-line"), areas = $("areas");

let at = 0;
let lastStyle: LastMoveStyle = "dot";
let captureStyle: CaptureStyle = "fade";

const models = new Map<NodeId, BoardModel>();
function modelAt(index: number): BoardModel {
  const id = ids[index]!;
  let model = models.get(id);
  if (!model) {
    const step = replay.step(tree, id);
    const position = replay.position(tree, id);
    model = step ? boardModel(rules, step) : boardModel(rules, { position, analysis: rules.analyze(position) });
    models.set(id, model);
  }
  return model;
}

const name = (c: Color) => (c === "B" ? "Black" : "White");
const percent = (x: number) => `${(100 * x).toFixed(1)}%`;

function describe(step: Step | null, index: number): string {
  if (!step) return `Start. ${name(tree.setup.toMove)} to move.`;
  const mover = step.move.color;
  let text = step.move.kind === "pass" ? `Move ${index}: ${name(mover)} passed.` : `Move ${index}: ${name(mover)} played.`;
  const taken = step.removed.filter((s) => s.c !== mover).length, lost = step.removed.length - taken;
  if (taken) text += ` ${name(mover)} took ${taken} stone${taken > 1 ? "s" : ""}.`;
  if (lost) text += ` ${lost} of ${name(mover)}'s own stone${lost > 1 ? "s" : ""} died with it.`;
  if (step.after.finished) {
    const { winner, margin } = step.analysis;
    text += winner ? ` The game is over: ${name(winner)} wins by ${percent(margin)} of the board.` : " The game is over: a tie.";
  }
  return text;
}

function render(animate: boolean) {
  const model = modelAt(at);
  const view = { px: board.clientWidth || 720 };
  svg.innerHTML = paintBoard(model, wood, view) + paintRemoved(model, wood, view, captureStyle) + paintLastMove(model, wood, view, lastStyle);
  if (!animate) svg.querySelectorAll(".vgo-fade").forEach((g) => g.remove());
  slider.value = String(at);
  label.textContent = `Move ${at} of ${ids.length - 1}`;
  status.textContent = describe(replay.step(tree, ids[at]!), at);
  const { score } = model.analysis;
  areas.textContent = model.position.stones.length
    ? `Black holds ${percent(score.B)} of the board, White ${percent(score.W)}; komi gives White ${percent(tree.setup.komi)}.`
    : "";
}

function go(index: number) {
  const next = Math.max(0, Math.min(ids.length - 1, index));
  if (next === at) return;
  const forwardOne = next === at + 1;
  at = next;
  render(forwardOne);
}

slider.max = String(ids.length - 1);
slider.addEventListener("input", () => go(Number(slider.value)));
$("first").addEventListener("click", () => go(0));
$("back").addEventListener("click", () => go(at - 1));
$("forward").addEventListener("click", () => go(at + 1));
$("last").addEventListener("click", () => go(ids.length - 1));
document.addEventListener("keydown", (e) => {
  const target = e.target as HTMLElement;
  if (target.closest("textarea, input")) return;
  const moves: Record<string, () => number> = { ArrowLeft: () => at - 1, ArrowRight: () => at + 1, Home: () => 0, End: () => ids.length - 1 };
  const to = moves[e.key];
  if (!to) return;
  e.preventDefault();
  go(to());
});

function segmented(group: string, onChoose: (value: string) => void) {
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>(`[data-group="${group}"] button`));
  const choose = (value: string) => {
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.value === value));
    onChoose(value);
  };
  buttons.forEach((b) => b.addEventListener("click", () => choose(b.dataset.value!)));
}
segmented("last", (v) => { lastStyle = v as LastMoveStyle; render(false); });
segmented("capture", (v) => { captureStyle = v as CaptureStyle; render(false); });

// Every capture on the main line, as a jump to the move before it.
const jumps = $("captures");
ids.forEach((id, index) => {
  const step = replay.step(tree, id);
  if (!step?.removed.length) return;
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = `Move ${index}: ${step.removed.length}`;
  button.title = `Go to move ${index - 1}, then step forward to see the capture`;
  button.addEventListener("click", () => { go(index - 1); board.focus({ preventScroll: true }); });
  jumps.append(button);
});

new ResizeObserver(() => render(false)).observe(board);
connectNotes($<HTMLFormElement>("notes"), $("notes-status"), "feedback/stepping");
// #move-40 opens at move 40, so a position can be linked.
const fromHash = () => {
  const match = /^#move-(\d+)$/.exec(location.hash);
  if (match) go(Number(match[1]));
};
window.addEventListener("hashchange", fromHash);
render(false);
fromHash();
