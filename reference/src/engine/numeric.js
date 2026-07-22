(function (root) {
  "use strict";

  const floatBuffer = new ArrayBuffer(8);
  const floatView = new DataView(floatBuffer);

  function nextUp(value) {
    if (Number.isNaN(value) || value === Infinity) return value;
    if (value === 0) return Number.MIN_VALUE;
    floatView.setFloat64(0, value, false);
    let bits = floatView.getBigUint64(0, false);
    bits = value > 0 ? bits + 1n : bits - 1n;
    floatView.setBigUint64(0, bits, false);
    return floatView.getFloat64(0, false);
  }

  function nextDown(value) {
    return -nextUp(-value);
  }

  function subtractInterval(left, right) {
    const value = left - right;
    return { lower: nextDown(value), upper: nextUp(value) };
  }

  function squareInterval(interval) {
    const first = interval.lower * interval.lower;
    const second = interval.upper * interval.upper;
    return {
      lower: interval.lower <= 0 && interval.upper >= 0
        ? 0
        : nextDown(Math.min(first, second)),
      upper: nextUp(Math.max(first, second)),
    };
  }

  function squaredDistanceInterval(firstX, firstY, secondX, secondY) {
    const x = squareInterval(subtractInterval(firstX, secondX));
    const y = squareInterval(subtractInterval(firstY, secondY));
    return {
      lower: nextDown(x.lower + y.lower),
      upper: nextUp(x.upper + y.upper),
    };
  }

  function fromFloat(value) {
    floatView.setFloat64(0, value, false);
    const bits = floatView.getBigUint64(0, false);
    const negative = (bits >> 63n) !== 0n;
    const rawExponent = Number((bits >> 52n) & 0x7ffn);
    const fraction = bits & ((1n << 52n) - 1n);
    const significand = rawExponent === 0 ? fraction : (1n << 52n) | fraction;
    return {
      coefficient: negative ? -significand : significand,
      exponent: rawExponent === 0 ? -1074 : rawExponent - 1023 - 52,
    };
  }

  function subtractDyadic(left, right) {
    const exponent = Math.min(left.exponent, right.exponent);
    return {
      coefficient:
        (left.coefficient << BigInt(left.exponent - exponent)) -
        (right.coefficient << BigInt(right.exponent - exponent)),
      exponent: exponent,
    };
  }

  function squareDyadic(value) {
    return {
      coefficient: value.coefficient * value.coefficient,
      exponent: value.exponent * 2,
    };
  }

  function addDyadic(left, right) {
    const exponent = Math.min(left.exponent, right.exponent);
    return {
      coefficient:
        (left.coefficient << BigInt(left.exponent - exponent)) +
        (right.coefficient << BigInt(right.exponent - exponent)),
      exponent: exponent,
    };
  }

  function exactSquaredDistance(firstX, firstY, secondX, secondY) {
    return addDyadic(
      squareDyadic(subtractDyadic(fromFloat(firstX), fromFloat(secondX))),
      squareDyadic(subtractDyadic(fromFloat(firstY), fromFloat(secondY)))
    );
  }

  function compareDyadic(left, right) {
    const exponent = Math.min(left.exponent, right.exponent);
    const leftValue = left.coefficient << BigInt(left.exponent - exponent);
    const rightValue = right.coefficient << BigInt(right.exponent - exponent);
    return leftValue < rightValue ? -1 : (leftValue > rightValue ? 1 : 0);
  }

  function strictlyCloser(originX, originY, candidateX, candidateY, thresholdX, thresholdY) {
    const candidateInterval = squaredDistanceInterval(
      originX, originY, candidateX, candidateY
    );
    const thresholdInterval = squaredDistanceInterval(
      originX, originY, thresholdX, thresholdY
    );
    const marginInterval = {
      lower: nextDown(thresholdInterval.lower - candidateInterval.upper),
      upper: nextUp(thresholdInterval.upper - candidateInterval.lower),
    };
    const candidateSquared = (originX - candidateX) ** 2 + (originY - candidateY) ** 2;
    const thresholdSquared = (originX - thresholdX) ** 2 + (originY - thresholdY) ** 2;
    const signedMargin = thresholdSquared - candidateSquared;
    const result = {
      isStrictlyLess: false,
      signedMargin: signedMargin,
      errorBound: Math.max(
        Math.abs(signedMargin - marginInterval.lower),
        Math.abs(marginInterval.upper - signedMargin)
      ),
      usedExactFallback: false,
    };
    if (marginInterval.lower > 0) {
      result.isStrictlyLess = true;
      return Object.freeze(result);
    }
    if (marginInterval.upper <= 0) return Object.freeze(result);

    result.usedExactFallback = true;
    result.isStrictlyLess = compareDyadic(
      exactSquaredDistance(originX, originY, candidateX, candidateY),
      exactSquaredDistance(originX, originY, thresholdX, thresholdY)
    ) < 0;
    return Object.freeze(result);
  }

  const numeric = Object.freeze({
    coordinateEpsilon: 1e-7,
    edgeEpsilon: 1e-10,
    collinearEpsilon: 1e-11,
    comparisonEpsilon: 1e-10,
    strictlyCloser: strictlyCloser,

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
