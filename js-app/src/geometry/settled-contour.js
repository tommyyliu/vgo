(function (root) {
  "use strict";

  const N = root.VGO.numeric;
  const legalSet = root.VGO.legalSet;

  /* The settled region, located analytically.

     A15 removes the Voronoi partition: the settled set is the union over stones
     of R_s = { x : ||x-s|| <= dist(x,L) }, because a non-nearest stone only
     makes the test harder. A16 makes each R_s star-shaped about its stone with a
     single-valued radial boundary T(u) >= r, so a two-dimensional level set
     becomes one scalar equation per direction. A17 solves that equation in
     closed form as a minimum over the A6 candidate families.

     Nothing here samples a field and nothing iterates to a root. A candidate is
     admitted only once the center realising it is verified legal, which is
     exactly what restricts a full circle or line to the surviving part of
     boundary L, so boundary L is never constructed.

     The result is presentation data. Capture does not use it. */

  const TOLERANCE = 2e-5;      // chord deviation, board units
  const BASE_RAYS = 16;
  const MAX_DEPTH = 9;
  const EXIT_MARGIN = 0.02;    // keeps an unbounded direction outside the board
  const BOARD_LOOP = [[-EXIT_MARGIN, -EXIT_MARGIN], [1 + EXIT_MARGIN, -EXIT_MARGIN],
                      [1 + EXIT_MARGIN, 1 + EXIT_MARGIN], [-EXIT_MARGIN, 1 + EXIT_MARGIN]];

  function compute(position, knownVertices) {
    const stones = position.stones, count = stones.length;
    if (!count) return result([], 0);

    const r = position.radius;
    const diameter = 2 * r;
    const diameterSquared = diameter * diameter;
    const vertices = Array.isArray(knownVertices) ? knownVertices : legalSet.vertices(position);

    /* L empty means dist(.,L) is infinite everywhere, so every point is settled.
       Deciding it once also keeps the sweep away from its only bad regime, where
       no candidate is ever admissible and every ray runs to the board edge. */
    if (legalSet.distance(position, stones[0].x, stones[0].y, vertices) === Infinity) {
      return result([BOARD_LOOP], 0);
    }

    const stoneX = new Float64Array(count), stoneY = new Float64Array(count);
    for (let i = 0; i < count; i++) { stoneX[i] = stones[i].x; stoneY[i] = stones[i].y; }

    // Per-stone candidate orderings. t_c >= ||c-s||/2 by the triangle inequality
    // (A17), so an ordering by distance permits an exact early stop.
    const stoneOrder = new Int32Array(count), stoneFloor = new Float64Array(count);
    const vertexTotal = vertices.length;
    const vertexOrder = new Int32Array(vertexTotal), vertexFloor = new Float64Array(vertexTotal);

    let originX = 0, originY = 0, capDistance = 0, evaluations = 0;

    function radius(ux, uy) {
      evaluations++;
      let best = capDistance;

      for (let index = 0; index < count; index++) {
        if (stoneFloor[index] >= best) break;
        const a = stoneOrder[index];
        const wx = originX - stoneX[a], wy = originY - stoneY[a];
        const along = ux * wx + uy * wy;
        const numerator = diameterSquared - (wx * wx + wy * wy);
        for (let branch = 0; branch < 2; branch++) {
          const denominator = 2 * (branch ? along - diameter : along + diameter);
          if (denominator === 0) continue;
          const t = numerator / denominator;
          if (!(t > 0) || t >= best) continue;
          const px = originX + t * ux - stoneX[a], py = originY + t * uy - stoneY[a];
          const span = Math.hypot(px, py);
          if (span < N.edgeEpsilon) continue;
          if (Math.abs(Math.abs(span - diameter) - t) > N.edgeEpsilon) continue;
          const footX = stoneX[a] + diameter * px / span;
          const footY = stoneY[a] + diameter * py / span;
          if (!legalSet.contains(position, footX, footY)) continue;
          best = t;
        }
      }

      for (let side = 0; side < 4; side++) {
        const vertical = side < 2;
        const line = (side % 2) ? 1 - r : r;
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
          if (Math.abs(Math.hypot(footX - reachedX, footY - reachedY) - t) > N.edgeEpsilon) continue;
          if (!legalSet.contains(position, footX, footY)) continue;
          best = t;
        }
      }

      for (let index = 0; index < vertexTotal; index++) {
        if (vertexFloor[index] >= best) break;
        const vertex = vertices[vertexOrder[index]];
        const gx = vertex[0] - originX, gy = vertex[1] - originY;
        const along = ux * gx + uy * gy;
        if (!(along > 0)) continue;
        const t = (gx * gx + gy * gy) / (2 * along);
        if (t < best) best = t;
      }

      return best;
    }

    // A16: a convex board is left once, so truncating here loses nothing inside it.
    function exitDistance(ux, uy) {
      let t = Infinity;
      if (ux > 0) t = Math.min(t, (1 - originX) / ux);
      else if (ux < 0) t = Math.min(t, -originX / ux);
      if (uy > 0) t = Math.min(t, (1 - originY) / uy);
      else if (uy < 0) t = Math.min(t, -originY / uy);
      return t + EXIT_MARGIN;
    }

    function point(t, ux, uy) { return [originX + t * ux, originY + t * uy]; }

    // Subdivide until the chord sits within tolerance of the true curve, so
    // points concentrate where the boundary bends or runs off the board.
    function flatten(loop, angleA, tA, ax, ay, angleB, tB, bx, by, depth) {
      const middle = 0.5 * (angleA + angleB);
      const ux = Math.cos(middle), uy = Math.sin(middle);
      capDistance = exitDistance(ux, uy);
      const tMiddle = radius(ux, uy);
      const chordX = 0.5 * (originX + tA * ax + originX + tB * bx);
      const chordY = 0.5 * (originY + tA * ay + originY + tB * by);
      const deviation = Math.hypot(originX + tMiddle * ux - chordX,
                                   originY + tMiddle * uy - chordY);
      if (depth >= MAX_DEPTH || deviation <= TOLERANCE) {
        loop.push(point(tMiddle, ux, uy), point(tB, bx, by));
        return;
      }
      flatten(loop, angleA, tA, ax, ay, middle, tMiddle, ux, uy, depth + 1);
      flatten(loop, middle, tMiddle, ux, uy, angleB, tB, bx, by, depth + 1);
    }

    const loops = [];
    const angles = new Float64Array(BASE_RAYS + 1), radii = new Float64Array(BASE_RAYS + 1);
    const dirX = new Float64Array(BASE_RAYS + 1), dirY = new Float64Array(BASE_RAYS + 1);

    for (let self = 0; self < count; self++) {
      originX = stoneX[self]; originY = stoneY[self];

      for (let k = 0; k < count; k++) stoneOrder[k] = k;
      stoneOrder.sort(function (a, b) {
        return Math.hypot(stoneX[a] - originX, stoneY[a] - originY) -
               Math.hypot(stoneX[b] - originX, stoneY[b] - originY);
      });
      for (let k = 0; k < count; k++) {
        const a = stoneOrder[k];
        const separation = Math.hypot(stoneX[a] - originX, stoneY[a] - originY);
        stoneFloor[k] = Math.max(0, separation - diameter) / 2;
      }

      for (let k = 0; k < vertexTotal; k++) vertexOrder[k] = k;
      vertexOrder.sort(function (a, b) {
        return Math.hypot(vertices[a][0] - originX, vertices[a][1] - originY) -
               Math.hypot(vertices[b][0] - originX, vertices[b][1] - originY);
      });
      for (let k = 0; k < vertexTotal; k++) {
        const v = vertices[vertexOrder[k]];
        vertexFloor[k] = Math.hypot(v[0] - originX, v[1] - originY) / 2;
      }

      for (let k = 0; k <= BASE_RAYS; k++) {
        const angle = 2 * Math.PI * k / BASE_RAYS;
        const ux = Math.cos(angle), uy = Math.sin(angle);
        angles[k] = angle; dirX[k] = ux; dirY[k] = uy;
        capDistance = exitDistance(ux, uy);
        radii[k] = radius(ux, uy);
      }

      const loop = [point(radii[0], dirX[0], dirY[0])];
      for (let k = 0; k < BASE_RAYS; k++) {
        flatten(loop, angles[k], radii[k], dirX[k], dirY[k],
                angles[k + 1], radii[k + 1], dirX[k + 1], dirY[k + 1], 0);
      }
      loop.pop();                                    // the closing point repeats the first
      loops.push(loop);
    }

    return result(loops, evaluations);
  }

  /* Loops are closed and simple by construction (A16: star-shaped about the
     stone, sampled at increasing angle), so the open-chain and bad-vertex
     diagnostics of the sampled renderer are structurally zero. They overlap,
     so the union is a nonzero-winding fill, not even-odd. */
  function result(loops, evaluations) {
    return {
      loops: loops,
      fillRule: "nonzero",
      evaluations: evaluations,
      tolerance: TOLERANCE,
      openChains: 0,
      badVertices: 0,
      segments: loops.reduce(function (total, loop) { return total + loop.length; }, 0),
    };
  }

  root.VGO.settledContour = Object.freeze({ compute: compute, TOLERANCE: TOLERANCE });
})(globalThis);
