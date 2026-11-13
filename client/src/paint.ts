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
  if (!d) return "";
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
