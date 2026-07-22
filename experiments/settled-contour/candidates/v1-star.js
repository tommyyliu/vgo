(function (root) {
  "use strict";

  const legalSet = root.VGO.legalSet;

  /* v1 — star-shaped polar sweep, numeric root find.

     Settled = union over stones of R_s = { x : dist(x,L) >= |x-s| }, because a
     non-nearest stone only makes the test harder: |x-s| >= |x-s(x)|. So the
     Voronoi partition is not needed at all.

     Along any ray from s, h(t) = dist(x(t),L) - t is non-increasing (dist(.,L)
     is 1-Lipschitz), and h(0) = dist(s,L) >= 2r > 0. So R_s is star-shaped about
     s with a single-valued radial boundary T(theta), and T >= r always. One
     monotone root find per ray replaces the entire quadtree.

     This version still calls the shipping legalSet.distance, so it isolates the
     structural win from the evaluation-cost win. */

  const FULL_BOARD = "M0 0L1 0L1 1L0 1Z";

  function build(rays, bisections) {
    return function (position) {
      const stones = position.stones;
      if (!stones.length) return "";
      const vertices = legalSet.vertices(position);
      const radius = position.radius;

      // dist(.,L) is infinite exactly when no legal placement exists anywhere,
      // and then every point of the board is settled.
      if (legalSet.distance(position, stones[0].x, stones[0].y, vertices) === Infinity) {
        return { d: FULL_BOARD, fillRule: "nonzero" };
      }

      const cos = new Float64Array(rays), sin = new Float64Array(rays);
      for (let index = 0; index < rays; index++) {
        const angle = 2 * Math.PI * index / rays;
        cos[index] = Math.cos(angle); sin[index] = Math.sin(angle);
      }

      let data = "";
      for (const stone of stones) {
        const height = function (t, ux, uy) {
          return legalSet.distance(position, stone.x + t * ux, stone.y + t * uy, vertices) - t;
        };
        let ring = "";
        for (let index = 0; index < rays; index++) {
          const ux = cos[index], uy = sin[index];
          let low = radius, high = 2 * radius;          // T >= r is guaranteed
          let guard = 0;
          while (height(high, ux, uy) > 0 && guard++ < 12) { low = high; high *= 2; }
          for (let step = 0; step < bisections; step++) {
            const middle = 0.5 * (low + high);
            if (height(middle, ux, uy) > 0) low = middle; else high = middle;
          }
          const t = 0.5 * (low + high);
          data += (index ? "L" : "M") + (stone.x + t * ux).toFixed(6) + " " + (stone.y + t * uy).toFixed(6);
          ring = "Z";
        }
        data += ring;
      }
      return { d: data, fillRule: "nonzero" };
    };
  }

  root.BENCH.register("v1-star-128", "polar sweep, 128 rays, 24 bisections", build(128, 24));
  root.BENCH.register("v1-star-64", "polar sweep, 64 rays, 20 bisections", build(64, 20));
})(globalThis);
