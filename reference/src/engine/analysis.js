(function (root) {
  "use strict";

  const N = root.VGO.numeric;
  const model = root.VGO.model;
  const voronoi = root.VGO.voronoi;
  const legalSet = root.VGO.legalSet;

  function analyze(position) {
    const validation = model.validate(position);
    const geometry = voronoi.compute(position);
    const freeVertices = legalSet.vertices(position);
    const aliveGroups = new Set();
    const evidence = new Map();

    if (validation.playable) {
      for (let i = 0; i < position.stones.length; i++) {
        const group = geometry.group[i];
        if (aliveGroups.has(group)) continue;
        for (const vertex of geometry.cells[i].poly) {
          const stone = position.stones[i];
          const witness = legalSet.escapeWitness(
            position, vertex[0], vertex[1], stone.x, stone.y, freeVertices
          );
          if (witness) {
            aliveGroups.add(group);
            evidence.set(group, {
              stone: i,
              vertex: vertex,
              freeDistance: Math.hypot(vertex[0] - witness[0], vertex[1] - witness[1]),
              influenceRadius: Math.hypot(vertex[0] - stone.x, vertex[1] - stone.y),
            });
            break;
          }
        }
      }
    }

    const settledGroups = new Set();
    for (let i = 0; i < position.stones.length; i++) {
      const group = geometry.group[i];
      if (!aliveGroups.has(group)) settledGroups.add(group);
    }

    const scores = { B: 0, W: 0 };
    for (let i = 0; i < position.stones.length; i++) {
      scores[position.stones[i].c] += geometry.cells[i].area;
    }
    const scoreDelta = scores.B - scores.W;
    const winner = Math.abs(scoreDelta) <= N.comparisonEpsilon ? null : (scoreDelta > 0 ? "B" : "W");

    return {
      position: position,
      validation: validation,
      geometry: geometry,
      legalVertices: freeVertices,
      aliveGroups: aliveGroups,
      settledGroups: settledGroups,
      survivalEvidence: evidence,
      scores: Object.freeze(scores),
      result: Object.freeze({ winner: winner, tie: winner === null, margin: Math.abs(scoreDelta) }),
    };
  }

  root.VGO.analysis = Object.freeze({ analyze: analyze });
})(globalThis);
