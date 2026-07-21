(function (root) {
  "use strict";

  const legalSet = root.VGO.legalSet;

  function compute(position, maxDepth) {
    const stones = position.stones;
    const legalVertices = legalSet.vertices(position);
    const leafCount = 1 << maxDepth;
    const latticeSize = 2 * leafCount;
    const samples = new Map();
    const edgeRoots = new Map();
    const segments = [];
    let evaluations = 0;

    function exactValue(x, y) {
      if (!stones.length) return -Infinity;
      let nearestSquared = Infinity;
      for (const stone of stones) {
        nearestSquared = Math.min(nearestSquared, (x - stone.x) ** 2 + (y - stone.y) ** 2);
      }
      evaluations++;
      const freeDistance = legalSet.distance(position, x, y, legalVertices);
      return freeDistance === Infinity ? Infinity : freeDistance - Math.sqrt(nearestSquared);
    }

    function sample(ix, iy) {
      const key = ix + "," + iy;
      if (!samples.has(key)) samples.set(key, exactValue(ix / latticeSize, iy / latticeSize));
      return { ix: ix, iy: iy, value: samples.get(key) };
    }

    const inside = function (point) { return point.value > 0; };
    const coordinate = function (point) { return [point.ix / latticeSize, point.iy / latticeSize]; };
    const pointKey = function (point) {
      return Math.round(point[0] * 1e9) + "," + Math.round(point[1] * 1e9);
    };
    const latticeKey = function (point) { return point.ix + "," + point.iy; };

    function edgeRoot(a, b) {
      const keyA = latticeKey(a), keyB = latticeKey(b);
      const key = keyA < keyB ? keyA + "|" + keyB : keyB + "|" + keyA;
      if (edgeRoots.has(key)) return edgeRoots.get(key);
      if (a.value === 0) return coordinate(a);
      if (b.value === 0) return coordinate(b);
      let ax = a.ix / latticeSize, ay = a.iy / latticeSize, av = a.value;
      let bx = b.ix / latticeSize, by = b.iy / latticeSize;
      const aInside = av > 0;
      for (let iteration = 0; iteration < 28; iteration++) {
        const midX = (ax + bx) / 2, midY = (ay + by) / 2;
        const midValue = exactValue(midX, midY);
        if ((midValue > 0) === aInside) {
          ax = midX; ay = midY; av = midValue;
        } else {
          bx = midX; by = midY;
        }
      }
      const result = [(ax + bx) / 2, (ay + by) / 2];
      edgeRoots.set(key, result);
      return result;
    }

    function addSegment(a, b) {
      if (pointKey(a) !== pointKey(b)) segments.push([a, b]);
    }

    function contourTriangle(a, b, c) {
      const cuts = [];
      for (const edge of [[a, b], [b, c], [c, a]]) {
        if (inside(edge[0]) !== inside(edge[1])) cuts.push(edgeRoot(edge[0], edge[1]));
      }
      if (cuts.length === 2) addSegment(cuts[0], cuts[1]);
    }

    function leaf(ix, iy) {
      const topLeft = sample(ix, iy), topRight = sample(ix + 2, iy);
      const bottomRight = sample(ix + 2, iy + 2), bottomLeft = sample(ix, iy + 2);
      const center = sample(ix + 1, iy + 1);
      contourTriangle(topLeft, topRight, center);
      contourTriangle(topRight, bottomRight, center);
      contourTriangle(bottomRight, bottomLeft, center);
      contourTriangle(bottomLeft, topLeft, center);
    }

    function visit(ix, iy, size, depth) {
      const half = size / 2;
      const center = sample(ix + half, iy + half);
      const side = size / latticeSize;
      const lipschitzBound = Math.SQRT2 * side;
      if (center.value > lipschitzBound || center.value < -lipschitzBound) return;
      if (depth === maxDepth) {
        leaf(ix, iy);
        return;
      }
      visit(ix, iy, half, depth + 1);
      visit(ix + half, iy, half, depth + 1);
      visit(ix + half, iy + half, half, depth + 1);
      visit(ix, iy + half, half, depth + 1);
    }
    visit(0, 0, latticeSize, 0);

    function addBoardInterval(a, b) {
      const aInside = inside(a), bInside = inside(b);
      if (aInside && bInside) addSegment(coordinate(a), coordinate(b));
      else if (aInside !== bInside) {
        const rootPoint = edgeRoot(a, b);
        addSegment(aInside ? coordinate(a) : rootPoint, aInside ? rootPoint : coordinate(b));
      }
    }
    for (let i = 0; i < leafCount; i++) {
      const a = 2 * i, b = a + 2;
      addBoardInterval(sample(a, 0), sample(b, 0));
      addBoardInterval(sample(latticeSize, a), sample(latticeSize, b));
      addBoardInterval(sample(b, latticeSize), sample(a, latticeSize));
      addBoardInterval(sample(0, b), sample(0, a));
    }

    const incident = new Map();
    segments.forEach(function (segment, segmentIndex) {
      segment.forEach(function (point) {
        const key = pointKey(point);
        const list = incident.get(key) || [];
        list.push(segmentIndex);
        incident.set(key, list);
      });
    });
    const degreeHistogram = {};
    for (const list of incident.values()) {
      degreeHistogram[list.length] = (degreeHistogram[list.length] || 0) + 1;
    }
    const badVertices = Array.from(incident.values()).filter(function (list) { return list.length !== 2; }).length;
    const used = new Uint8Array(segments.length);
    const loops = [];
    let openChains = 0;
    for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex++) {
      if (used[segmentIndex]) continue;
      const start = segments[segmentIndex][0];
      const loop = [start];
      let current = start, nextSegment = segmentIndex;
      for (let guard = 0; nextSegment >= 0 && guard <= segments.length; guard++) {
        used[nextSegment] = 1;
        const segment = segments[nextSegment];
        const next = pointKey(segment[0]) === pointKey(current) ? segment[1] : segment[0];
        loop.push(next);
        current = next;
        if (pointKey(current) === pointKey(start)) break;
        nextSegment = (incident.get(pointKey(current)) || []).find(function (index) { return !used[index]; });
        if (nextSegment === undefined) nextSegment = -1;
      }
      if (loop.length >= 4 && pointKey(loop[loop.length - 1]) === pointKey(start)) {
        loop.pop();
        const clean = loop.filter(function (point, index) {
          const previous = loop[(index + loop.length - 1) % loop.length];
          const next = loop[(index + 1) % loop.length];
          return Math.abs((point[0] - previous[0]) * (next[1] - point[1]) -
            (point[1] - previous[1]) * (next[0] - point[0])) > 1e-12;
        });
        if (clean.length >= 3) loops.push(clean);
      } else {
        openChains++;
      }
    }

    return {
      loops: loops,
      evaluations: evaluations,
      openChains: openChains,
      badVertices: badVertices,
      degreeHistogram: degreeHistogram,
      segments: segments.length,
      maxDepth: maxDepth,
    };
  }

  root.VGO.settledContour = Object.freeze({ compute: compute });
})(globalThis);
