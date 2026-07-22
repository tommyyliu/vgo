(function (root) {
  "use strict";

  // Baseline: the shipping implementation, wrapped to the candidate contract.
  // Lipschitz-pruned quadtree over the whole board, marching triangles at the
  // leaves, 28 bisection steps per sign-crossing edge.
  function polygonPath(loops) {
    let data = "";
    for (const loop of loops) {
      data += "M" + loop[0][0].toFixed(6) + " " + loop[0][1].toFixed(6);
      for (let index = 1; index < loop.length; index++) {
        data += "L" + loop[index][0].toFixed(6) + " " + loop[index][1].toFixed(6);
      }
      data += "Z";
    }
    return data;
  }

  // Honours whatever fill rule the shipping module declares: the sampled
  // renderer nested its loops and needed even-odd, the analytic one overlaps
  // per-stone regions and needs nonzero.
  function shipping(position) {
    const contour = root.VGO.settledContour.compute(position);
    return { d: polygonPath(contour.loops), fillRule: contour.fillRule || "evenodd" };
  }

  root.BENCH.register("v0-shipping", "whatever reference/src currently ships", shipping);
})(globalThis);
