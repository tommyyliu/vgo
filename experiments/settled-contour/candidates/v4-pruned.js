(function (root) {
  "use strict";

  const N = root.VGO.numeric;

  /* v4 — v3 plus exact distance pruning.

     For any candidate legal point c realising the crossing on its ray,
     |c-s| <= |c-x(t)| + |x(t)-s| = 2t, so t >= |c-s|/2. Visiting candidates in
     increasing distance from the stone lets the scan stop as soon as
     |c-s|/2 >= best: no reach heuristic, no spatial grid, and no approximation.

     The stone's own exclusion circle sits first (distance 0) and yields t = r
     immediately whenever the radial escape is open, so in open play the scan
     terminates after a handful of candidates however many stones are on the
     board. */

  const CAP_MARGIN = 0.02;

  function build(tolerance, baseRays, maxDepth) {
    return function (position) {
      const stones = position.stones, count = stones.length;
      if (!count) return "";

      const r = position.radius, dia = 2 * r, diaSquared = dia * dia;
      const epsilon = N.coordinateEpsilon;
      const insetLow = r - epsilon, insetHigh = 1 - r + epsilon;
      const clearSquared = (dia - epsilon) * (dia - epsilon);

      const stoneX = new Float64Array(count), stoneY = new Float64Array(count);
      for (let i = 0; i < count; i++) { stoneX[i] = stones[i].x; stoneY[i] = stones[i].y; }

      function legal(x, y, skipA, skipB) {
        if (x < insetLow || x > insetHigh || y < insetLow || y > insetHigh) return false;
        for (let k = 0; k < count; k++) {
          if (k === skipA || k === skipB) continue;
          const dx = x - stoneX[k], dy = y - stoneY[k];
          if (dx * dx + dy * dy < clearSquared) return false;
        }
        return true;
      }

      const vertexX = [], vertexY = [];
      function offer(x, y, skipA, skipB) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        if (!legal(x, y, skipA, skipB)) return;
        vertexX.push(x); vertexY.push(y);
      }
      for (const cx of [r, 1 - r]) for (const cy of [r, 1 - r]) offer(cx, cy, -1, -1);
      for (let a = 0; a < count; a++) {
        for (const edge of [r, 1 - r]) {
          let disc = diaSquared - (edge - stoneX[a]) * (edge - stoneX[a]);
          if (disc >= -epsilon) {
            const off = Math.sqrt(Math.max(0, disc));
            offer(edge, stoneY[a] + off, a, -1); offer(edge, stoneY[a] - off, a, -1);
          }
          disc = diaSquared - (edge - stoneY[a]) * (edge - stoneY[a]);
          if (disc >= -epsilon) {
            const off = Math.sqrt(Math.max(0, disc));
            offer(stoneX[a] + off, edge, a, -1); offer(stoneX[a] - off, edge, a, -1);
          }
        }
        for (let b = a + 1; b < count; b++) {
          const dx = stoneX[b] - stoneX[a], dy = stoneY[b] - stoneY[a];
          const separation = Math.hypot(dx, dy);
          if (separation < N.edgeEpsilon || separation > 2 * dia + epsilon) continue;
          const along = separation / 2;
          const heightSquared = diaSquared - along * along;
          if (heightSquared < -epsilon) continue;
          const height = Math.sqrt(Math.max(0, heightSquared));
          const midX = (stoneX[a] + stoneX[b]) / 2, midY = (stoneY[a] + stoneY[b]) / 2;
          const ux = dx / separation, uy = dy / separation;
          offer(midX - uy * height, midY + ux * height, a, b);
          offer(midX + uy * height, midY - ux * height, a, b);
        }
      }
      const vertexTotal = vertexX.length;

      /* L empty => dist(.,L) is infinite everywhere => the whole board is
         settled. Without this the sweep still answers correctly, but every ray
         runs to the board edge finding no candidate, which is the one regime
         where the quadtree wins (its boundary shrinks to nothing). Emptiness is
         decided by the shipping predicate so the notion matches the oracle. */
      const pairs = new Array(vertexTotal);
      for (let k = 0; k < vertexTotal; k++) pairs[k] = [vertexX[k], vertexY[k]];
      if (root.VGO.legalSet.distance(position, stoneX[0], stoneY[0], pairs) === Infinity) {
        return { d: "M0 0L1 0L1 1L0 1Z", fillRule: "nonzero" };
      }

      // Per-stone orderings, built once and reused by every ray of that stone.
      const stoneOrder = new Int32Array(count);
      const stoneFloor = new Float64Array(count);      // lower bound on t from this circle
      const vertexOrder = new Int32Array(vertexTotal);
      const vertexFloor = new Float64Array(vertexTotal);

      let sx = 0, sy = 0, capDistance = 0;

      function solve(ux, uy) {
        let best = capDistance;

        for (let index = 0; index < count; index++) {
          if (stoneFloor[index] >= best) break;         // every later circle is farther still
          const a = stoneOrder[index];
          const wx = sx - stoneX[a], wy = sy - stoneY[a];
          const along = ux * wx + uy * wy;
          const numerator = diaSquared - (wx * wx + wy * wy);
          for (let branch = 0; branch < 2; branch++) {
            const denominator = 2 * (branch ? along - dia : along + dia);
            if (denominator === 0) continue;
            const t = numerator / denominator;
            if (!(t > 0) || t >= best) continue;
            const px = sx + t * ux - stoneX[a], py = sy + t * uy - stoneY[a];
            const span = Math.hypot(px, py);
            if (span < N.edgeEpsilon) continue;
            if (Math.abs(Math.abs(span - dia) - t) > 1e-9) continue;
            if (!legal(stoneX[a] + dia * px / span, stoneY[a] + dia * py / span, a, -1)) continue;
            best = t;
          }
        }

        for (let side = 0; side < 4; side++) {
          const vertical = side < 2;
          const line = (side % 2) ? 1 - r : r;
          const component = vertical ? ux : uy;
          const offset = line - (vertical ? sx : sy);
          for (let branch = 0; branch < 2; branch++) {
            const denominator = branch ? component - 1 : component + 1;
            if (denominator === 0) continue;
            const t = offset / denominator;
            if (!(t > 0) || t >= best) continue;
            const footX = vertical ? line : sx + t * ux;
            const footY = vertical ? sy + t * uy : line;
            const dx = footX - (sx + t * ux), dy = footY - (sy + t * uy);
            if (Math.abs(Math.hypot(dx, dy) - t) > 1e-9) continue;
            if (!legal(footX, footY, -1, -1)) continue;
            best = t;
          }
        }

        for (let index = 0; index < vertexTotal; index++) {
          if (vertexFloor[index] >= best) break;        // sorted, so the rest cannot win
          const v = vertexOrder[index];
          const gx = vertexX[v] - sx, gy = vertexY[v] - sy;
          const along = ux * gx + uy * gy;
          if (!(along > 0)) continue;
          const t = (gx * gx + gy * gy) / (2 * along);
          if (t < best) best = t;
        }

        return best;
      }

      function exitDistance(ux, uy) {
        let t = Infinity;
        if (ux > 0) t = Math.min(t, (1 - sx) / ux); else if (ux < 0) t = Math.min(t, -sx / ux);
        if (uy > 0) t = Math.min(t, (1 - sy) / uy); else if (uy < 0) t = Math.min(t, -sy / uy);
        return t + CAP_MARGIN;
      }

      const parts = [];
      function emit(t, ux, uy) {
        parts.push((sx + t * ux).toFixed(6) + " " + (sy + t * uy).toFixed(6));
      }

      function refine(angleA, tA, ax, ay, angleB, tB, bx, by, depth) {
        const middle = 0.5 * (angleA + angleB);
        const ux = Math.cos(middle), uy = Math.sin(middle);
        capDistance = exitDistance(ux, uy);
        const tM = solve(ux, uy);
        const chordX = 0.5 * ((sx + tA * ax) + (sx + tB * bx));
        const chordY = 0.5 * ((sy + tA * ay) + (sy + tB * by));
        if (depth >= maxDepth ||
            Math.hypot(sx + tM * ux - chordX, sy + tM * uy - chordY) <= tolerance) {
          emit(tM, ux, uy);
          emit(tB, bx, by);
          return;
        }
        refine(angleA, tA, ax, ay, middle, tM, ux, uy, depth + 1);
        refine(middle, tM, ux, uy, angleB, tB, bx, by, depth + 1);
      }

      let data = "";
      const angles = new Float64Array(baseRays + 1);
      const radii = new Float64Array(baseRays + 1);
      const dirX = new Float64Array(baseRays + 1), dirY = new Float64Array(baseRays + 1);

      for (let self = 0; self < count; self++) {
        sx = stoneX[self]; sy = stoneY[self];

        for (let k = 0; k < count; k++) stoneOrder[k] = k;
        const stoneKeys = Array.prototype.slice.call(stoneOrder).sort(function (a, b) {
          return Math.hypot(stoneX[a] - sx, stoneY[a] - sy) - Math.hypot(stoneX[b] - sx, stoneY[b] - sy);
        });
        for (let k = 0; k < count; k++) {
          stoneOrder[k] = stoneKeys[k];
          const distance = Math.hypot(stoneX[stoneKeys[k]] - sx, stoneY[stoneKeys[k]] - sy);
          stoneFloor[k] = Math.max(0, distance - dia) / 2;
        }

        for (let k = 0; k < vertexTotal; k++) vertexOrder[k] = k;
        const vertexKeys = Array.prototype.slice.call(vertexOrder).sort(function (a, b) {
          return Math.hypot(vertexX[a] - sx, vertexY[a] - sy) - Math.hypot(vertexX[b] - sx, vertexY[b] - sy);
        });
        for (let k = 0; k < vertexTotal; k++) {
          vertexOrder[k] = vertexKeys[k];
          vertexFloor[k] = Math.hypot(vertexX[vertexKeys[k]] - sx, vertexY[vertexKeys[k]] - sy) / 2;
        }

        parts.length = 0;
        for (let k = 0; k <= baseRays; k++) {
          const angle = 2 * Math.PI * k / baseRays;
          const ux = Math.cos(angle), uy = Math.sin(angle);
          angles[k] = angle; dirX[k] = ux; dirY[k] = uy;
          capDistance = exitDistance(ux, uy);
          radii[k] = solve(ux, uy);
        }
        emit(radii[0], dirX[0], dirY[0]);
        for (let k = 0; k < baseRays; k++) {
          refine(angles[k], radii[k], dirX[k], dirY[k],
                 angles[k + 1], radii[k + 1], dirX[k + 1], dirY[k + 1], 0);
        }
        data += "M" + parts.join("L") + "Z";
      }

      return { d: data, fillRule: "nonzero" };
    };
  }

  root.BENCH.register("v4-pruned", "closed form + adaptive + distance pruning, tol 2e-4", build(2e-4, 16, 7));
  root.BENCH.register("v4-pruned-fine", "closed form + adaptive + distance pruning, tol 2e-5", build(2e-5, 16, 9));
})(globalThis);
