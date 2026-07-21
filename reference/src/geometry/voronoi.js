(function (root) {
  "use strict";

  const N = root.VGO.numeric;

  function distanceSquared(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  function normalizePolygon(points) {
    if (points.length < 3) return points.slice();
    const deduped = [];
    for (const point of points) {
      if (!deduped.length || distanceSquared(point, deduped[deduped.length - 1]) > N.edgeEpsilon ** 2) {
        deduped.push(point);
      }
    }
    if (deduped.length > 1 && distanceSquared(deduped[0], deduped[deduped.length - 1]) <= N.edgeEpsilon ** 2) {
      deduped.pop();
    }

    let polygon = deduped;
    let changed = true;
    while (changed && polygon.length >= 3) {
      changed = false;
      const clean = [];
      for (let i = 0; i < polygon.length; i++) {
        const a = polygon[(i + polygon.length - 1) % polygon.length];
        const b = polygon[i];
        const c = polygon[(i + 1) % polygon.length];
        const abx = b[0] - a[0], aby = b[1] - a[1];
        const bcx = c[0] - b[0], bcy = c[1] - b[1];
        const cross = abx * bcy - aby * bcx;
        const scale = Math.hypot(abx, aby) + Math.hypot(bcx, bcy);
        if (Math.abs(cross) <= N.collinearEpsilon * Math.max(1, scale)) {
          changed = true;
        } else {
          clean.push(b);
        }
      }
      polygon = clean;
    }
    return polygon;
  }

  function clipHalfPlane(polygon, constraint) {
    const out = [];
    const count = polygon.length;
    for (let k = 0; k < count; k++) {
      const a = polygon[k];
      const b = polygon[(k + 1) % count];
      const fa = constraint.nx * a[0] + constraint.ny * a[1] - constraint.offset;
      const fb = constraint.nx * b[0] + constraint.ny * b[1] - constraint.offset;
      const aInside = fa <= N.coordinateEpsilon;
      const bInside = fb <= N.coordinateEpsilon;
      if (aInside) out.push(a);
      if (aInside !== bInside) {
        const denominator = fa - fb;
        if (Math.abs(denominator) > Number.EPSILON) {
          const t = fa / denominator;
          out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
      }
    }
    return normalizePolygon(out);
  }

  function polygonArea(polygon) {
    let twiceArea = 0;
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      twiceArea += a[0] * b[1] - b[0] * a[1];
    }
    return Math.abs(twiceArea) / 2;
  }

  function boardConstraint(side) {
    if (side === "left") return { nx: -1, ny: 0, offset: 0, source: { kind: "board", side: side } };
    if (side === "right") return { nx: 1, ny: 0, offset: 1, source: { kind: "board", side: side } };
    if (side === "top") return { nx: 0, ny: -1, offset: 0, source: { kind: "board", side: side } };
    return { nx: 0, ny: 1, offset: 1, source: { kind: "board", side: side } };
  }

  function bisectorConstraint(stones, i, j) {
    const a = stones[i], b = stones[j];
    const nx = b.x - a.x;
    const ny = b.y - a.y;
    return {
      nx: nx,
      ny: ny,
      offset: nx * (a.x + b.x) / 2 + ny * (a.y + b.y) / 2,
      source: { kind: "bisector", stones: [i, j], other: j },
    };
  }

  function constraintResidual(constraint, point) {
    const length = Math.hypot(constraint.nx, constraint.ny);
    if (length === 0) return Infinity;
    return Math.abs(constraint.nx * point[0] + constraint.ny * point[1] - constraint.offset) / length;
  }

  function edgeSource(constraints, a, b) {
    let best = null;
    let bestResidual = Infinity;
    for (const constraint of constraints) {
      const residual = Math.max(constraintResidual(constraint, a), constraintResidual(constraint, b));
      if (residual < bestResidual) {
        bestResidual = residual;
        best = constraint.source;
      }
    }
    return bestResidual <= N.coordinateEpsilon * 4 ? best : null;
  }

  function compute(position) {
    const stones = position.stones;
    const count = stones.length;
    const cells = [];
    const adjacency = Array.from({ length: count }, function () { return new Set(); });
    const diagnostics = { unclassifiedEdges: 0, degenerateEdges: 0 };

    for (let i = 0; i < count; i++) {
      let polygon = [[0, 0], [1, 0], [1, 1], [0, 1]];
      const constraints = [
        boardConstraint("left"), boardConstraint("right"),
        boardConstraint("top"), boardConstraint("bottom"),
      ];
      for (let j = 0; j < count && polygon.length; j++) {
        if (j === i) continue;
        const constraint = bisectorConstraint(stones, i, j);
        constraints.push(constraint);
        polygon = clipHalfPlane(polygon, constraint);
      }

      polygon = normalizePolygon(polygon);
      const edges = [];
      for (let k = 0; k < polygon.length; k++) {
        const a = polygon[k];
        const b = polygon[(k + 1) % polygon.length];
        if (Math.hypot(b[0] - a[0], b[1] - a[1]) <= N.edgeEpsilon) {
          diagnostics.degenerateEdges++;
          continue;
        }
        const source = edgeSource(constraints, a, b);
        if (!source) diagnostics.unclassifiedEdges++;
        const other = source && source.kind === "bisector" ? source.other : -1;
        edges.push({ A: a, B: b, j: other, source: source });
        if (other >= 0) adjacency[i].add(other);
      }
      cells.push({ polygon: polygon, poly: polygon, area: polygonArea(polygon), edges: edges });
    }

    for (let i = 0; i < count; i++) {
      for (const j of adjacency[i]) adjacency[j].add(i);
    }

    const parent = new Int32Array(count);
    for (let i = 0; i < count; i++) parent[i] = i;
    function find(index) {
      let rootIndex = index;
      while (parent[rootIndex] !== rootIndex) rootIndex = parent[rootIndex];
      while (parent[index] !== index) {
        const next = parent[index];
        parent[index] = rootIndex;
        index = next;
      }
      return rootIndex;
    }
    for (let i = 0; i < count; i++) {
      for (const j of adjacency[i]) {
        if (stones[i].c !== stones[j].c) continue;
        const a = find(i), b = find(j);
        if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
      }
    }
    const group = new Int32Array(count);
    for (let i = 0; i < count; i++) group[i] = find(i);

    return { cells: cells, adj: adjacency, group: group, diagnostics: diagnostics };
  }

  root.VGO.voronoi = Object.freeze({
    compute: compute,
    polygonArea: polygonArea,
    normalizePolygon: normalizePolygon,
  });
})(globalThis);
