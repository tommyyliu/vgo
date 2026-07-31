(function (root) {
  "use strict";

  const N = root.VGO.numeric;

  function inInset(position, x, y) {
    const r = position.radius;
    return x >= r - N.coordinateEpsilon && x <= 1 - r + N.coordinateEpsilon &&
           y >= r - N.coordinateEpsilon && y <= 1 - r + N.coordinateEpsilon;
  }

  function clearOfStones(position, x, y, skipA, skipB) {
    const minimum = 2 * position.radius - N.coordinateEpsilon;
    const minimumSquared = minimum * minimum;
    for (let i = 0; i < position.stones.length; i++) {
      if (i === skipA || i === skipB) continue;
      const stone = position.stones[i];
      const dx = x - stone.x, dy = y - stone.y;
      if (dx * dx + dy * dy < minimumSquared) return false;
    }
    return true;
  }

  function contains(position, x, y) {
    return Number.isFinite(x) && Number.isFinite(y) &&
      inInset(position, x, y) && clearOfStones(position, x, y, -1, -1);
  }

  function vertices(position) {
    const stones = position.stones;
    const r = position.radius;
    const diameter = 2 * r;
    const result = [];
    const seen = new Set();
    function push(x, y, skipA, skipB) {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      if (!inInset(position, x, y) || !clearOfStones(position, x, y, skipA, skipB)) return;
      const key = Math.round(x / N.coordinateEpsilon) + "," + Math.round(y / N.coordinateEpsilon);
      if (!seen.has(key)) {
        seen.add(key);
        result.push([x, y]);
      }
    }

    for (const x of [r, 1 - r]) {
      for (const y of [r, 1 - r]) push(x, y, -1, -1);
    }
    for (let i = 0; i < stones.length; i++) {
      const stone = stones[i];
      for (const x of [r, 1 - r]) {
        const discriminant = diameter * diameter - (x - stone.x) ** 2;
        if (discriminant >= -N.coordinateEpsilon) {
          const offset = Math.sqrt(Math.max(0, discriminant));
          push(x, stone.y + offset, i, -1);
          push(x, stone.y - offset, i, -1);
        }
      }
      for (const y of [r, 1 - r]) {
        const discriminant = diameter * diameter - (y - stone.y) ** 2;
        if (discriminant >= -N.coordinateEpsilon) {
          const offset = Math.sqrt(Math.max(0, discriminant));
          push(stone.x + offset, y, i, -1);
          push(stone.x - offset, y, i, -1);
        }
      }
    }
    for (let i = 0; i < stones.length; i++) {
      for (let j = i + 1; j < stones.length; j++) {
        const dx = stones[j].x - stones[i].x;
        const dy = stones[j].y - stones[i].y;
        const distance = Math.hypot(dx, dy);
        if (distance < N.edgeEpsilon || distance > 2 * diameter + N.coordinateEpsilon) continue;
        const along = distance / 2;
        const heightSquared = diameter * diameter - along * along;
        if (heightSquared < -N.coordinateEpsilon) continue;
        const height = Math.sqrt(Math.max(0, heightSquared));
        const midX = (stones[i].x + stones[j].x) / 2;
        const midY = (stones[i].y + stones[j].y) / 2;
        const ux = dx / distance, uy = dy / distance;
        push(midX - uy * height, midY + ux * height, i, j);
        push(midX + uy * height, midY - ux * height, i, j);
      }
    }
    return result;
  }

  function visitCandidates(position, x, y, knownVertices, visit) {
    if (contains(position, x, y) && visit(x, y)) return true;
    const r = position.radius;
    const diameter = 2 * r;

    for (const stone of position.stones) {
      const dx = x - stone.x, dy = y - stone.y;
      const radialDistance = Math.hypot(dx, dy);
      const directions = radialDistance < N.edgeEpsilon
        ? [[1, 0], [-1, 0], [0, 1], [0, -1]]
        : [[dx / radialDistance, dy / radialDistance]];
      for (const direction of directions) {
        const candidateX = stone.x + diameter * direction[0];
        const candidateY = stone.y + diameter * direction[1];
        if (contains(position, candidateX, candidateY) && visit(candidateX, candidateY)) {
          return true;
        }
      }
    }

    for (const candidate of [[r, y], [1 - r, y], [x, r], [x, 1 - r]]) {
      if (contains(position, candidate[0], candidate[1]) && visit(candidate[0], candidate[1])) {
        return true;
      }
    }
    for (const candidate of knownVertices || vertices(position)) {
      if (visit(candidate[0], candidate[1])) return true;
    }
    return false;
  }

  function distance(position, x, y, knownVertices) {
    let best = Infinity;
    visitCandidates(position, x, y, knownVertices, function (candidateX, candidateY) {
      best = Math.min(best, Math.hypot(x - candidateX, y - candidateY));
      return false;
    });
    return best;
  }

  function escapeWitness(position, vertexX, vertexY, stoneX, stoneY, knownVertices) {
    let witness = null;
    visitCandidates(position, vertexX, vertexY, knownVertices, function (x, y) {
      if (N.strictlyCloser(vertexX, vertexY, x, y, stoneX, stoneY).isStrictlyLess) {
        witness = [x, y];
        return true;
      }
      return false;
    });
    return witness;
  }

  function nearest(position, x, y) {
    if (contains(position, x, y)) return { x: x, y: y, legal: true, snapped: false };
    const r = position.radius;
    // Pushed past each constraint rather than placed on it, so a snapped point
    // survives being re-checked by an implementation whose arithmetic
    // associates differently. See numeric.snapMargin.
    const diameter = 2 * r + N.snapMargin;
    const inset = Math.min(r + N.snapMargin, 0.5);
    const lo = Math.min(inset, 1 - inset), hi = Math.max(inset, 1 - inset);
    const clamp = function (value) { return Math.min(hi, Math.max(lo, value)); };
    const candidates = [[clamp(x), clamp(y)]];
    for (const stone of position.stones) {
      const dx = x - stone.x, dy = y - stone.y;
      const radialDistance = Math.hypot(dx, dy);
      const directions = radialDistance < N.edgeEpsilon
        ? [[1, 0], [-1, 0], [0, 1], [0, -1]]
        : [[dx / radialDistance, dy / radialDistance]];
      for (const direction of directions) {
        candidates.push([
          clamp(stone.x + diameter * direction[0]),
          clamp(stone.y + diameter * direction[1]),
        ]);
      }
    }
    candidates.push([inset, clamp(y)], [1 - inset, clamp(y)], [clamp(x), inset], [clamp(x), 1 - inset]);
    for (const vertex of vertices(position)) candidates.push(vertex);

    let best = null;
    let bestDistance = Infinity;
    for (const candidate of candidates) {
      if (!contains(position, candidate[0], candidate[1])) continue;
      const candidateDistance = (candidate[0] - x) ** 2 + (candidate[1] - y) ** 2;
      if (candidateDistance < bestDistance) {
        bestDistance = candidateDistance;
        best = candidate;
      }
    }
    return best
      ? { x: best[0], y: best[1], legal: true, snapped: true }
      : { x: x, y: y, legal: false, snapped: false };
  }

  root.VGO.legalSet = Object.freeze({
    inInset: inInset,
    clearOfStones: clearOfStones,
    contains: contains,
    vertices: vertices,
    distance: distance,
    escapeWitness: escapeWitness,
    nearest: nearest,
  });
})(globalThis);
