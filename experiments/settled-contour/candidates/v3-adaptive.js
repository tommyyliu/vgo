(function (root) {
  "use strict";

  const N = root.VGO.numeric;

  /* v3 — closed-form radial boundary with adaptive angular flattening.

     Two corrections over v2:

     1. T(theta) is genuinely infinite for rays that leave the board: h(t) then
        never crosses zero, because dist(.,L) grows as fast as t. That is not an
        empty legal set, it just means the star region is unbounded in that
        direction. Capping T at the ray's board-exit distance plus a margin keeps
        the polygon outside the board where it belongs, and makes the fully
        sealed board (L empty, every ray infinite) fall out for free.

     2. Uniform rays cannot follow a radial function that jumps from ~r to
        off-board between neighbours. Angles are subdivided until the chord sits
        within tolerance of the true curve, so points concentrate exactly where
        the boundary bends. */

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

      // Vertices of L, computed once for the position (same rules as legalSet.vertices).
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
      const vertexCount = vertexX.length;

      let sx = 0, sy = 0, self = 0, capDistance = 0;

      // T(theta): the first t where dist(x(t),L) = t, as the minimum over the
      // three feature families A6 enumerates. Every root below is closed form.
      function solve(ux, uy) {
        let best = Infinity;

        for (let a = 0; a < count; a++) {
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

        for (let v = 0; v < vertexCount; v++) {
          const gx = vertexX[v] - sx, gy = vertexY[v] - sy;
          const along = ux * gx + uy * gy;
          if (!(along > 0)) continue;
          const t = (gx * gx + gy * gy) / (2 * along);
          if (t < best) best = t;
        }

        return best < capDistance ? best : capDistance;
      }

      // Distance at which the ray leaves the board, plus a margin, so an
      // unbounded direction is emitted just outside and never clips inward.
      function exitDistance(ux, uy) {
        let t = Infinity;
        if (ux > 0) t = Math.min(t, (1 - sx) / ux); else if (ux < 0) t = Math.min(t, -sx / ux);
        if (uy > 0) t = Math.min(t, (1 - sy) / uy); else if (uy < 0) t = Math.min(t, -sy / uy);
        return t + CAP_MARGIN;
      }

      let data = "";
      const parts = [];

      function emit(angle, t, ux, uy) {
        parts.push((sx + t * ux).toFixed(6) + " " + (sy + t * uy).toFixed(6));
      }

      // Flatten [angleA, angleB] until the chord matches the true curve.
      function refine(angleA, tA, ax, ay, angleB, tB, bx, by, depth) {
        const middle = 0.5 * (angleA + angleB);
        const ux = Math.cos(middle), uy = Math.sin(middle);
        const saved = capDistance;
        capDistance = exitDistance(ux, uy);
        const tM = solve(ux, uy);
        capDistance = saved;
        const px = sx + tM * ux, py = sy + tM * uy;
        const chordX = 0.5 * ((sx + tA * ax) + (sx + tB * bx));
        const chordY = 0.5 * ((sy + tA * ay) + (sy + tB * by));
        if (depth >= maxDepth || Math.hypot(px - chordX, py - chordY) <= tolerance) {
          emit(middle, tM, ux, uy);
          emit(angleB, tB, bx, by);
          return;
        }
        refine(angleA, tA, ax, ay, middle, tM, ux, uy, depth + 1);
        refine(middle, tM, ux, uy, angleB, tB, bx, by, depth + 1);
      }

      for (self = 0; self < count; self++) {
        sx = stoneX[self]; sy = stoneY[self];
        parts.length = 0;

        const angles = new Float64Array(baseRays + 1);
        const radii = new Float64Array(baseRays + 1);
        const dirX = new Float64Array(baseRays + 1), dirY = new Float64Array(baseRays + 1);
        for (let k = 0; k <= baseRays; k++) {
          const angle = 2 * Math.PI * k / baseRays;
          const ux = Math.cos(angle), uy = Math.sin(angle);
          angles[k] = angle; dirX[k] = ux; dirY[k] = uy;
          capDistance = exitDistance(ux, uy);
          radii[k] = solve(ux, uy);
        }

        emit(angles[0], radii[0], dirX[0], dirY[0]);
        for (let k = 0; k < baseRays; k++) {
          refine(angles[k], radii[k], dirX[k], dirY[k],
                 angles[k + 1], radii[k + 1], dirX[k + 1], dirY[k + 1], 0);
        }

        data += "M" + parts.join("L") + "Z";
      }

      return { d: data, fillRule: "nonzero" };
    };
  }

  root.BENCH.register("v3-adaptive", "closed form + adaptive flattening, tol 2e-4", build(2e-4, 16, 7));
  root.BENCH.register("v3-adaptive-fine", "closed form + adaptive flattening, tol 2e-5", build(2e-5, 16, 9));
})(globalThis);
