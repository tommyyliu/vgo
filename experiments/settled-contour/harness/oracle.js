(function (root) {
  "use strict";

  const legalSet = root.VGO.legalSet;

  /* The ground truth. This is the same scalar field settled-contour.js samples:
     g(x) = dist(x, L) - dist(x, nearest stone centre), settled where g > 0.
     It is exact (A6 enumerates every feature a closest legal centre can lie on),
     so a candidate is judged against the predicate itself, never against another
     approximation. */
  function gField(position, x, y, vertices) {
    const stones = position.stones;
    if (!stones.length) return -Infinity;
    let nearestSquared = Infinity;
    for (const stone of stones) {
      const dx = x - stone.x, dy = y - stone.y;
      const squared = dx * dx + dy * dy;
      if (squared < nearestSquared) nearestSquared = squared;
    }
    const freeDistance = legalSet.distance(position, x, y, vertices);
    return freeDistance === Infinity ? Infinity : freeDistance - Math.sqrt(nearestSquared);
  }

  /* A stratified grid of sample points with the exact classification cached.
     Computed once per case and shared by every candidate. */
  function sample(position, side) {
    const vertices = legalSet.vertices(position);
    const count = side * side;
    const xs = new Float64Array(count);
    const ys = new Float64Array(count);
    const inside = new Uint8Array(count);
    const magnitude = new Float64Array(count);
    let settledCount = 0;
    for (let iy = 0; iy < side; iy++) {
      for (let ix = 0; ix < side; ix++) {
        const index = iy * side + ix;
        const x = (ix + 0.5) / side, y = (iy + 0.5) / side;
        const value = gField(position, x, y, vertices);
        xs[index] = x; ys[index] = y;
        inside[index] = value > 0 ? 1 : 0;
        magnitude[index] = Math.abs(value);
        settledCount += inside[index];
      }
    }
    return { side: count && side, xs: xs, ys: ys, inside: inside,
             magnitude: magnitude, count: count, settledCount: settledCount };
  }

  /* Candidates emit an SVG path in board units. Rasterising it through Path2D
     means a polygon candidate and an analytic-arc candidate are scored by
     exactly the same rule. */
  const SCALE = 8192;
  let context = null;

  /* A candidate returns either a path string (scored even-odd, matching the
     shipping renderer) or { d, fillRule } when its representation overlaps —
     per-stone regions may overlap, and nonzero winding is then the union. */
  function pathTester(result) {
    if (!context) {
      const canvas = document.createElement("canvas");
      canvas.width = 1; canvas.height = 1;
      context = canvas.getContext("2d");
    }
    const pathData = typeof result === "string" ? result : (result && result.d);
    const fillRule = (result && result.fillRule) || "evenodd";
    if (!pathData) return function () { return false; };
    const scaled = new Path2D();
    scaled.addPath(new Path2D(pathData), new DOMMatrix([SCALE, 0, 0, SCALE, 0, 0]));
    return function (x, y) {
      return context.isPointInPath(scaled, x * SCALE, y * SCALE, fillRule);
    };
  }

  /* g is 2-Lipschitz, so |g(x)| / 2 is a lower bound on the distance from x to
     the true boundary. A disagreement with tiny |g| is boundary jitter; a
     disagreement with large |g| is a real geometric error. */
  function score(pathData, samples) {
    const test = pathTester(pathData);
    let mismatches = 0, worst = 0, worstX = 0, worstY = 0, falseIn = 0, falseOut = 0;
    for (let index = 0; index < samples.count; index++) {
      const claimed = test(samples.xs[index], samples.ys[index]) ? 1 : 0;
      if (claimed === samples.inside[index]) continue;
      mismatches++;
      if (claimed) falseIn++; else falseOut++;
      const error = samples.magnitude[index] / 2;
      if (error > worst) { worst = error; worstX = samples.xs[index]; worstY = samples.ys[index]; }
    }
    return {
      mismatches: mismatches,
      mismatchRate: samples.count ? mismatches / samples.count : 0,
      falseIn: falseIn,
      falseOut: falseOut,
      worstError: worst,
      worstAt: [worstX, worstY],
    };
  }

  root.ORACLE = Object.freeze({ gField: gField, sample: sample, score: score, pathTester: pathTester });
})(globalThis);
