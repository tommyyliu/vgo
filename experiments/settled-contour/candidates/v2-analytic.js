(function (root) {
  "use strict";

  const N = root.VGO.numeric;
  const FULL_BOARD = "M0 0L1 0L1 1L0 1Z";

  /* v2 — closed-form radial boundary, local features only.

     Along a ray from s, h(t) = dist(x(t),L) - t is non-increasing, and it is the
     lower envelope of h_f over the boundary features f of L. Each h_f is itself
     non-increasing, so the crossing is simply T = min_f T_f. Every T_f has a
     closed form, so the bisection disappears:

       own / other exclusion circle (centre q, radius dia = 2r)
         |x(t) - q| = dia -/+ t  =>  t = (dia^2 - |s-q|^2) / (2(u.(s-q) +/- dia))
         the t^2 terms cancel, so the root is linear. For q = s this yields
         t = dia/2 = r exactly: the open-space disk, recovered analytically.

       inset edge (line X)     |X - x(t).x| = t      =>  t = (X - s.x)/(1 +/- u.x)
       vertex v of L           |v - x(t)| = t        =>  t = |v-s|^2 / (2 u.(v-s))

     A6 proves a closest legal centre is always one of these three types, so the
     minimum over them is exact. Each candidate is accepted only after its
     realising point is verified legal, which is what trims the features to the
     actual boundary of L without ever constructing that boundary.

     Locality: the realising point lies within 2T of s, so only stones within
     2T + dia can generate a relevant feature. The reach is verified after the
     sweep and the stone is recomputed if the bound was too tight. */

  function build(rays) {
    const cosine = new Float64Array(rays), sine = new Float64Array(rays);
    for (let index = 0; index < rays; index++) {
      const angle = 2 * Math.PI * index / rays;
      cosine[index] = Math.cos(angle); sine[index] = Math.sin(angle);
    }

    return function (position) {
      const stones = position.stones, count = stones.length;
      if (!count) return "";

      const r = position.radius, dia = 2 * r;
      const epsilon = N.coordinateEpsilon;
      const insetLow = r - epsilon, insetHigh = 1 - r + epsilon;
      const clearSquared = (dia - epsilon) * (dia - epsilon);
      const diaSquared = dia * dia;

      // Local stone neighbourhood, ordered by distance so that any radius is a prefix.
      const order = new Int32Array(count);
      const distances = new Float64Array(count);
      const localX = new Float64Array(count), localY = new Float64Array(count);

      // Scratch for local vertices of L.
      let vertexX = new Float64Array(256), vertexY = new Float64Array(256);
      let vertexCount = 0;
      function pushVertex(x, y) {
        if (vertexCount === vertexX.length) {
          const grownX = new Float64Array(vertexCount * 2), grownY = new Float64Array(vertexCount * 2);
          grownX.set(vertexX); grownY.set(vertexY);
          vertexX = grownX; vertexY = grownY;
        }
        vertexX[vertexCount] = x; vertexY[vertexCount] = y; vertexCount++;
      }

      let blockerCount = 0;
      function legal(x, y, skipA, skipB) {
        if (x < insetLow || x > insetHigh || y < insetLow || y > insetHigh) return false;
        for (let k = 0; k < blockerCount; k++) {
          if (k === skipA || k === skipB) continue;
          const dx = x - localX[k], dy = y - localY[k];
          if (dx * dx + dy * dy < clearSquared) return false;
        }
        return true;
      }

      let data = "";
      let sawInfinite = false;

      for (let index = 0; index < count; index++) {
        const stone = stones[index];
        const sx = stone.x, sy = stone.y;

        for (let j = 0; j < count; j++) {
          const dx = stones[j].x - sx, dy = stones[j].y - sy;
          distances[j] = Math.hypot(dx, dy);
          order[j] = j;
        }
        const sorted = Array.prototype.slice.call(order).sort(function (a, b) {
          return distances[a] - distances[b];
        });
        for (let k = 0; k < count; k++) {
          localX[k] = stones[sorted[k]].x; localY[k] = stones[sorted[k]].y;
        }
        const sortedDistance = new Float64Array(count);
        for (let k = 0; k < count; k++) sortedDistance[k] = distances[sorted[k]];

        let reach = 3 * dia;
        let radii = new Float64Array(rays);
        let attempt = 0;

        for (;;) {
          attempt++;
          // Generators inside `reach`, blockers inside `reach + 2*dia`; both prefixes.
          let generatorCount = 0;
          blockerCount = 0;
          while (generatorCount < count && sortedDistance[generatorCount] <= reach) generatorCount++;
          while (blockerCount < count && sortedDistance[blockerCount] <= reach + 2 * dia) blockerCount++;

          // Local vertices of L, mirroring legalSet.vertices with the same skip rules.
          vertexCount = 0;
          const acceptRadius = reach + dia;
          function offer(x, y, skipA, skipB) {
            if (!Number.isFinite(x) || !Number.isFinite(y)) return;
            const dx = x - sx, dy = y - sy;
            if (dx * dx + dy * dy > acceptRadius * acceptRadius) return;
            if (!legal(x, y, skipA, skipB)) return;
            pushVertex(x, y);
          }
          for (const cx of [r, 1 - r]) for (const cy of [r, 1 - r]) offer(cx, cy, -1, -1);
          for (let a = 0; a < generatorCount; a++) {
            const ax = localX[a], ay = localY[a];
            for (const edge of [r, 1 - r]) {
              const dxDisc = diaSquared - (edge - ax) * (edge - ax);
              if (dxDisc >= -epsilon) {
                const off = Math.sqrt(Math.max(0, dxDisc));
                offer(edge, ay + off, a, -1); offer(edge, ay - off, a, -1);
              }
              const dyDisc = diaSquared - (edge - ay) * (edge - ay);
              if (dyDisc >= -epsilon) {
                const off = Math.sqrt(Math.max(0, dyDisc));
                offer(ax + off, edge, a, -1); offer(ax - off, edge, a, -1);
              }
            }
            for (let b = a + 1; b < generatorCount; b++) {
              const dx = localX[b] - ax, dy = localY[b] - ay;
              const separation = Math.hypot(dx, dy);
              if (separation < N.edgeEpsilon || separation > 2 * dia + epsilon) continue;
              const along = separation / 2;
              const heightSquared = diaSquared - along * along;
              if (heightSquared < -epsilon) continue;
              const height = Math.sqrt(Math.max(0, heightSquared));
              const midX = (ax + localX[b]) / 2, midY = (ay + localY[b]) / 2;
              const ux = dx / separation, uy = dy / separation;
              offer(midX - uy * height, midY + ux * height, a, b);
              offer(midX + uy * height, midY - ux * height, a, b);
            }
          }

          let largest = 0;
          for (let ray = 0; ray < rays; ray++) {
            const ux = cosine[ray], uy = sine[ray];
            let best = Infinity;

            // exclusion circles
            for (let a = 0; a < generatorCount; a++) {
              const wx = sx - localX[a], wy = sy - localY[a];
              const along = ux * wx + uy * wy;
              const numerator = diaSquared - (wx * wx + wy * wy);
              for (let branch = 0; branch < 2; branch++) {
                const denominator = 2 * (branch ? along - dia : along + dia);
                if (denominator === 0) continue;
                const t = numerator / denominator;
                if (!(t > 0) || t >= best) continue;
                const px = sx + t * ux - localX[a], py = sy + t * uy - localY[a];
                const span = Math.hypot(px, py);
                if (span < N.edgeEpsilon) continue;
                if (Math.abs(Math.abs(span - dia) - t) > 1e-9) continue;   // wrong branch
                const footX = localX[a] + dia * px / span, footY = localY[a] + dia * py / span;
                if (!legal(footX, footY, a, -1)) continue;
                best = t;
              }
            }

            // inset edges
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
                const reachedX = sx + t * ux, reachedY = sy + t * uy;
                if (Math.abs(Math.hypot(footX - reachedX, footY - reachedY) - t) > 1e-9) continue;
                if (Math.hypot(footX - sx, footY - sy) > acceptRadius) continue;
                if (!legal(footX, footY, -1, -1)) continue;
                best = t;
              }
            }

            // vertices of L
            for (let v = 0; v < vertexCount; v++) {
              const gx = vertexX[v] - sx, gy = vertexY[v] - sy;
              const along = ux * gx + uy * gy;
              if (!(along > 0)) continue;
              const t = (gx * gx + gy * gy) / (2 * along);
              if (t < best) best = t;
            }

            radii[ray] = best;
            if (best > largest) largest = best;
          }

          // The realising point sits within 2T of s, so reach must cover 2T + dia.
          if (Number.isFinite(largest) && 2 * largest + dia <= reach) break;
          if (blockerCount === count && generatorCount === count) {
            if (!Number.isFinite(largest)) sawInfinite = true;
            break;
          }
          reach = Number.isFinite(largest) ? 2 * largest + 2 * dia : reach * 2;
          if (attempt > 8) { reach = 4; }
        }

        if (sawInfinite) break;
        for (let ray = 0; ray < rays; ray++) {
          const t = radii[ray];
          data += (ray ? "L" : "M") + (sx + t * cosine[ray]).toFixed(6) +
                  " " + (sy + t * sine[ray]).toFixed(6);
        }
        data += "Z";
      }

      // dist(.,L) is infinite exactly when L is empty, and then everything is settled.
      if (sawInfinite) return { d: FULL_BOARD, fillRule: "nonzero" };
      return { d: data, fillRule: "nonzero" };
    };
  }

  root.BENCH.register("v2-analytic-128", "closed-form radial boundary, 128 rays", build(128));
  root.BENCH.register("v2-analytic-64", "closed-form radial boundary, 64 rays", build(64));
})(globalThis);
