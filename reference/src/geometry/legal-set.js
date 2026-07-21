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

  function distance(position, x, y, knownVertices) {
    if (contains(position, x, y)) return 0;
    const r = position.radius;
    const diameter = 2 * r;
    let best = Infinity;

    for (let i = 0; i < position.stones.length; i++) {
      const stone = position.stones[i];
      const dx = x - stone.x, dy = y - stone.y;
      const radialDistance = Math.hypot(dx, dy);
      const directions = radialDistance < N.edgeEpsilon
        ? [[1, 0], [-1, 0], [0, 1], [0, -1]]
        : [[dx / radialDistance, dy / radialDistance]];
      for (const direction of directions) {
        const candidateX = stone.x + diameter * direction[0];
        const candidateY = stone.y + diameter * direction[1];
        if (contains(position, candidateX, candidateY)) {
          best = Math.min(best, Math.hypot(x - candidateX, y - candidateY));
        }
      }
    }

    const feet = [[r, y], [1 - r, y], [x, r], [x, 1 - r]];
    for (const foot of feet) {
      if (contains(position, foot[0], foot[1])) {
        best = Math.min(best, Math.hypot(x - foot[0], y - foot[1]));
      }
    }
    const candidates = knownVertices || vertices(position);
    for (const vertex of candidates) {
      best = Math.min(best, Math.hypot(x - vertex[0], y - vertex[1]));
    }
    return best;
  }

  function nearest(position, x, y) {
    if (contains(position, x, y)) return { x: x, y: y, legal: true, snapped: false };
    const r = position.radius;
    const diameter = 2 * r;
    const clamp = function (value) { return Math.min(1 - r, Math.max(r, value)); };
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
    candidates.push([r, clamp(y)], [1 - r, clamp(y)], [clamp(x), r], [clamp(x), 1 - r]);
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
    nearest: nearest,
  });
})(globalThis);
