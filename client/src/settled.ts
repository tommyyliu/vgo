// The settled shape, for drawing: where no stone can ever take land from its owner.
//
// Ported from js-app/src/geometry/settled-contour.js, which locates it analytically
// (spec/LEMMAS.md A15-A17). A15 removes the Voronoi partition: the settled set is the
// union over stones of R_s = { x : |x - s| <= dist(x, L) }, L the legal set. A16 makes
// each R_s star-shaped about its stone with a single-valued radial boundary, so a
// two-dimensional level set becomes one scalar equation per direction, and A17 solves
// it in closed form as a minimum over the A6 candidate families. Nothing samples a
// field and nothing iterates to a root.
//
// Presentation only: captures are Rust's verdict. The legal set's corners come from
// Rust (`Analysis.legalVertices`); the point test below is the legal set's definition.
// This is the `vgo` ruleset's shape; `official` needs its own and is not built yet.

import type { Analysis, Point, Position } from "./rules";

// Chord deviation in board units: a quarter pixel on a 720 px board. The engine's
// renderer used 2e-5, a hundredth of a pixel, which made every board half a megabyte.
const TOLERANCE = 3.5e-4;
const BASE_RAYS = 16;
const MAX_DEPTH = 9;
const EXIT_MARGIN = 0.02; // keeps an unbounded direction outside the board
const EDGE_EPSILON = 1e-10;
const COORDINATE_EPSILON = 1e-7;

export type Loop = readonly (readonly [number, number])[];

const BOARD_LOOP: Loop = [
  [-EXIT_MARGIN, -EXIT_MARGIN], [1 + EXIT_MARGIN, -EXIT_MARGIN],
  [1 + EXIT_MARGIN, 1 + EXIT_MARGIN], [-EXIT_MARGIN, 1 + EXIT_MARGIN],
];

/** Whether a stone centre fits at (x, y): inside the inset square and clear of every stone. */
function fits(position: Position, x: number, y: number): boolean {
  const r = position.radius;
  if (x < r - COORDINATE_EPSILON || x > 1 - r + COORDINATE_EPSILON || y < r - COORDINATE_EPSILON || y > 1 - r + COORDINATE_EPSILON) return false;
  const minimum = 2 * r - COORDINATE_EPSILON;
  const minimumSquared = minimum * minimum;
  for (const stone of position.stones) {
    const dx = x - stone.x, dy = y - stone.y;
    if (dx * dx + dy * dy < minimumSquared) return false;
  }
  return true;
}

/**
 * Closed loops whose union (nonzero fill) is the settled region. `legalSetEmpty` must
 * come from the rules: when no stone fits anywhere, every point is settled.
 */
export function settledLoops(position: Position, analysis: Analysis, legalSetEmpty: boolean, tolerance = TOLERANCE): Loop[] {
  if (position.ruleset !== "vgo") throw new Error(`no settled shape for the ${position.ruleset} ruleset yet`);
  const stones = position.stones, count = stones.length;
  if (!count) return [];
  if (legalSetEmpty) return [BOARD_LOOP];

  const r = position.radius;
  const diameter = 2 * r;
  const diameterSquared = diameter * diameter;
  const vertices: readonly Point[] = analysis.legalVertices;

  const stoneX = new Float64Array(count), stoneY = new Float64Array(count);
  stones.forEach((stone, i) => { stoneX[i] = stone.x; stoneY[i] = stone.y; });

  // Per-stone candidate orderings. t_c >= |c - s| / 2 by the triangle inequality (A17),
  // so an ordering by distance permits an exact early stop.
  const stoneOrder = new Int32Array(count), stoneFloor = new Float64Array(count);
  const vertexTotal = vertices.length;
  const vertexOrder = new Int32Array(vertexTotal), vertexFloor = new Float64Array(vertexTotal);

  let originX = 0, originY = 0, capDistance = 0;

  function radius(ux: number, uy: number): number {
    let best = capDistance;

    for (let index = 0; index < count; index++) {
      if (stoneFloor[index]! >= best) break;
      const a = stoneOrder[index]!;
      const wx = originX - stoneX[a]!, wy = originY - stoneY[a]!;
      const along = ux * wx + uy * wy;
      const numerator = diameterSquared - (wx * wx + wy * wy);
      for (let branch = 0; branch < 2; branch++) {
        const denominator = 2 * (branch ? along - diameter : along + diameter);
        if (denominator === 0) continue;
        const t = numerator / denominator;
        if (!(t > 0) || t >= best) continue;
        const px = originX + t * ux - stoneX[a]!, py = originY + t * uy - stoneY[a]!;
        const span = Math.hypot(px, py);
        if (span < EDGE_EPSILON) continue;
        if (Math.abs(Math.abs(span - diameter) - t) > EDGE_EPSILON) continue;
        const footX = stoneX[a]! + diameter * px / span;
        const footY = stoneY[a]! + diameter * py / span;
        if (!fits(position, footX, footY)) continue;
        best = t;
      }
    }

    for (let side = 0; side < 4; side++) {
      const vertical = side < 2;
      const line = side % 2 ? 1 - r : r;
      const component = vertical ? ux : uy;
      const offset = line - (vertical ? originX : originY);
      for (let branch = 0; branch < 2; branch++) {
        const denominator = branch ? component - 1 : component + 1;
        if (denominator === 0) continue;
        const t = offset / denominator;
        if (!(t > 0) || t >= best) continue;
        const reachedX = originX + t * ux, reachedY = originY + t * uy;
        const footX = vertical ? line : reachedX;
        const footY = vertical ? reachedY : line;
        if (Math.abs(Math.hypot(footX - reachedX, footY - reachedY) - t) > EDGE_EPSILON) continue;
        if (!fits(position, footX, footY)) continue;
        best = t;
      }
    }

    for (let index = 0; index < vertexTotal; index++) {
      if (vertexFloor[index]! >= best) break;
      const vertex = vertices[vertexOrder[index]!]!;
      const gx = vertex.x - originX, gy = vertex.y - originY;
      const along = ux * gx + uy * gy;
      if (!(along > 0)) continue;
      const t = (gx * gx + gy * gy) / (2 * along);
      if (t < best) best = t;
    }
    return best;
  }

  // A16: a convex board is left once, so truncating here loses nothing inside it.
  function exitDistance(ux: number, uy: number): number {
    let t = Infinity;
    if (ux > 0) t = Math.min(t, (1 - originX) / ux);
    else if (ux < 0) t = Math.min(t, -originX / ux);
    if (uy > 0) t = Math.min(t, (1 - originY) / uy);
    else if (uy < 0) t = Math.min(t, -originY / uy);
    return t + EXIT_MARGIN;
  }

  const point = (t: number, ux: number, uy: number): [number, number] => [originX + t * ux, originY + t * uy];

  // Subdivide until the chord sits within tolerance of the true curve, so points
  // concentrate where the boundary bends or runs off the board.
  function flatten(loop: [number, number][], angleA: number, tA: number, ax: number, ay: number,
    angleB: number, tB: number, bx: number, by: number, depth: number): void {
    const middle = 0.5 * (angleA + angleB);
    const ux = Math.cos(middle), uy = Math.sin(middle);
    capDistance = exitDistance(ux, uy);
    const tMiddle = radius(ux, uy);
    const chordX = originX + 0.5 * (tA * ax + tB * bx);
    const chordY = originY + 0.5 * (tA * ay + tB * by);
    const deviation = Math.hypot(originX + tMiddle * ux - chordX, originY + tMiddle * uy - chordY);
    if (depth >= MAX_DEPTH || deviation <= tolerance) {
      loop.push(point(tMiddle, ux, uy), point(tB, bx, by));
      return;
    }
    flatten(loop, angleA, tA, ax, ay, middle, tMiddle, ux, uy, depth + 1);
    flatten(loop, middle, tMiddle, ux, uy, angleB, tB, bx, by, depth + 1);
  }

  const loops: Loop[] = [];
  const angles = new Float64Array(BASE_RAYS + 1), radii = new Float64Array(BASE_RAYS + 1);
  const dirX = new Float64Array(BASE_RAYS + 1), dirY = new Float64Array(BASE_RAYS + 1);

  for (let own = 0; own < count; own++) {
    originX = stoneX[own]!; originY = stoneY[own]!;

    const byDistance = (x: number, y: number) => Math.hypot(x - originX, y - originY);
    const sortedStones = Array.from({ length: count }, (_, k) => k).sort((a, b) => byDistance(stoneX[a]!, stoneY[a]!) - byDistance(stoneX[b]!, stoneY[b]!));
    sortedStones.forEach((a, k) => {
      stoneOrder[k] = a;
      stoneFloor[k] = Math.max(0, byDistance(stoneX[a]!, stoneY[a]!) - diameter) / 2;
    });
    const sortedVertices = Array.from({ length: vertexTotal }, (_, k) => k).sort((a, b) => byDistance(vertices[a]!.x, vertices[a]!.y) - byDistance(vertices[b]!.x, vertices[b]!.y));
    sortedVertices.forEach((v, k) => {
      vertexOrder[k] = v;
      vertexFloor[k] = byDistance(vertices[v]!.x, vertices[v]!.y) / 2;
    });

    for (let k = 0; k <= BASE_RAYS; k++) {
      const angle = 2 * Math.PI * k / BASE_RAYS;
      const ux = Math.cos(angle), uy = Math.sin(angle);
      angles[k] = angle; dirX[k] = ux; dirY[k] = uy;
      capDistance = exitDistance(ux, uy);
      radii[k] = radius(ux, uy);
    }

    const loop: [number, number][] = [point(radii[0]!, dirX[0]!, dirY[0]!)];
    for (let k = 0; k < BASE_RAYS; k++) {
      flatten(loop, angles[k]!, radii[k]!, dirX[k]!, dirY[k]!, angles[k + 1]!, radii[k + 1]!, dirX[k + 1]!, dirY[k + 1]!, 0);
    }
    loop.pop(); // the closing point repeats the first
    loops.push(loop);
  }
  return loops;
}
