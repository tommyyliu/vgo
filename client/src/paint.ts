// Paint: a board model and a look in, SVG out, in board units (0 to 1). No DOM, so
// the browser, the diagram tool and video frames share it. Widths are screen pixels,
// converted with the board's on-screen size.

import { side, type FrontStyle, type Look } from "./looks";
import type { BoardModel, Polyline } from "./model";
import type { Color } from "./rules";

export interface View {
  /** The board's size on screen, in CSS pixels. */
  readonly px: number;
}

// clipPath ids are global to an HTML document, so every board painted gets its own.
let painted = 0;

const f = (n: number) => String(+n.toFixed(5));

function pathData(lines: readonly Polyline[], closed: boolean): string {
  let d = "";
  for (const line of lines) {
    if (line.length < 2) continue;
    d += "M" + f(line[0]![0]) + " " + f(line[0]![1]);
    for (let k = 1; k < line.length; k++) d += "L" + f(line[k]![0]) + " " + f(line[k]![1]);
    if (closed) d += "Z";
  }
  return d;
}

/** Board, land in two tones per side. */
export function paintLand(model: BoardModel, look: Look): string {
  const id = `vgo${painted++}`;
  let out = `<rect width="1" height="1" fill="${look.board}"/>`;
  const settled = pathData(model.settled, true);
  // The settled outline is written once and drawn for each side through its cells.
  if (settled) out += `<defs><path id="${id}s" d="${settled}" fill-rule="nonzero"/></defs>`;
  for (const c of ["B", "W"] as Color[]) {
    const cells = pathData(model.land[c], true);
    if (!cells) continue;
    out += `<path d="${cells}" fill="${side(look, c).open}"/>`;
    if (settled) {
      out += `<clipPath id="${id}${c}"><path d="${cells}"/></clipPath>`;
      out += `<use href="#${id}s" fill="${side(look, c).settled}" clip-path="url(#${id}${c})"/>`;
    }
  }
  return out;
}

export function paintFront(model: BoardModel, front: FrontStyle, view: View): string {
  const d = pathData(model.front, false);
  if (!d || front.width <= 0) return "";
  const u = 1 / view.px;
  const line = (color: string, width: number) =>
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="${f(width * u)}" stroke-linecap="round" stroke-linejoin="round"/>`;
  return line(front.color, front.width) + (front.core ? line(front.core.color, front.core.width) : "");
}

export function paintStones(model: BoardModel, look: Look, view: View): string {
  const r = model.position.radius, w = look.rimWidth / view.px;
  let out = "";
  for (const stone of model.position.stones) {
    const colors = side(look, stone.c);
    // The rim sits inside the stone's edge, so the stone covers exactly its disc.
    out += `<circle cx="${f(stone.x)}" cy="${f(stone.y)}" r="${f(r - w / 2)}" fill="${colors.stone}" stroke="${colors.rim}" stroke-width="${f(w)}"/>`;
  }
  return out;
}

export function paintBoard(model: BoardModel, look: Look, view: View): string {
  return paintLand(model, look) + paintFront(model, look.front, view) + paintStones(model, look, view);
}

/** A standalone SVG document of `body`, drawn `px` wide. */
export function svgFile(body: string, px: number, label = "Voronoi Go board"): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="${px}" height="${px}" role="img" aria-label="${label}">${body}</svg>`;
}

/** How the stone just played is marked. Review 2 picks one. */
export type LastMoveStyle = "dot" | "ring" | "none";

export function paintLastMove(model: BoardModel, look: Look, view: View, style: LastMoveStyle): string {
  const last = model.last;
  if (style === "none" || !last || last.move.kind !== "place") return "";
  const { x, y, color } = last.move;
  // A stone that died on arrival is not on the board to mark.
  if (!model.position.stones.some((s) => s.x === x && s.y === y)) return "";
  const r = model.position.radius, mark = side(look, color).mark;
  if (style === "dot") return `<circle cx="${f(x)}" cy="${f(y)}" r="${f(0.3 * r)}" fill="${mark}"/>`;
  const w = 2 / view.px;
  return `<circle cx="${f(x)}" cy="${f(y)}" r="${f(0.55 * r)}" fill="none" stroke="${mark}" stroke-width="${f(w)}"/>`;
}

/** How stones the last move removed are shown. Review 2 picks one. */
export type CaptureStyle = "instant" | "fade" | "outline";

/**
 * The removed stones, drawn over the board after the move. "fade" draws them whole in a
 * group with class `vgo-fade` for the page to animate out; "outline" leaves a dashed ring
 * where each stood while the move is current.
 */
export function paintRemoved(model: BoardModel, look: Look, view: View, style: CaptureStyle): string {
  const removed = model.last?.removed ?? [];
  if (style === "instant" || !removed.length) return "";
  const r = model.position.radius, w = look.rimWidth / view.px;
  if (style === "fade") {
    const stones = removed.map((s) => `<circle cx="${f(s.x)}" cy="${f(s.y)}" r="${f(r - w / 2)}" fill="${side(look, s.c).stone}" stroke="${side(look, s.c).rim}" stroke-width="${f(w)}"/>`);
    return `<g class="vgo-fade">${stones.join("")}</g>`;
  }
  const dash = `${f(4 / view.px)} ${f(3 / view.px)}`, ring = 1.5 / view.px;
  return removed.map((s) => `<circle cx="${f(s.x)}" cy="${f(s.y)}" r="${f(r - ring / 2)}" fill="none" stroke="${side(look, s.c).rim}" stroke-width="${f(ring)}" stroke-dasharray="${dash}"/>`).join("");
}
