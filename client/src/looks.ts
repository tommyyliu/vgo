// Looks: colours only. Every look draws the same things; widths are screen pixels.

import type { Color } from "./rules";

export interface SideColors {
  /** The stone itself. */
  readonly stone: string;
  readonly rim: string;
  /** Land a stone could still take. */
  readonly open: string;
  /** Land no stone can take. */
  readonly settled: string;
  /** A mark drawn on this side's stone, such as the last move. */
  readonly mark: string;
}

/** The line where the two sides' land meets. Any colour works; width 0 draws none. */
export interface FrontStyle {
  readonly color: string;
  readonly width: number;
  readonly core?: { readonly color: string; readonly width: number };
}

export interface Look {
  readonly name: string;
  /** Around the board. */
  readonly page: string;
  /** Under the land; it shows only where nothing covers it. */
  readonly board: string;
  readonly front: FrontStyle;
  readonly rimWidth: number;
  readonly B: SideColors;
  readonly W: SideColors;
}

export function side(look: Look, c: Color): SideColors {
  return c === "B" ? look.B : look.W;
}

/**
 * The wooden board: walnut for Black's land, maple for White's. The default.
 * Its front is a 5 px pale gap, chosen at review 1 over a dark line and a gap with a core.
 */
export const wood: Look = {
  name: "wood",
  page: "#f3eee4",
  board: "#e2c79a",
  front: { color: "#fbf6ea", width: 5 },
  rimWidth: 1.5,
  B: { stone: "#1c1a18", rim: "rgba(20,16,10,.75)", open: "#b88a5c", settled: "#7d5535", mark: "#fbf6ea" },
  W: { stone: "#fffdf8", rim: "rgba(20,16,10,.75)", open: "#f4e4c0", settled: "#dfbf80", mark: "#1c1a18" },
};
