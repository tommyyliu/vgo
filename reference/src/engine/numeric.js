(function (root) {
  "use strict";

  const numeric = Object.freeze({
    coordinateEpsilon: 1e-7,
    edgeEpsilon: 1e-10,
    collinearEpsilon: 1e-11,
    captureMargin: 1e-7,
    comparisonEpsilon: 1e-10,

    compare(a, b, epsilon) {
      const tolerance = epsilon === undefined ? this.comparisonEpsilon : epsilon;
      if (a < b - tolerance) return -1;
      if (a > b + tolerance) return 1;
      return 0;
    },

    near(a, b, epsilon) {
      return this.compare(a, b, epsilon) === 0;
    },
  });

  root.VGO = root.VGO || {};
  root.VGO.numeric = numeric;
})(globalThis);
