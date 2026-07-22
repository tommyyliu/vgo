(function (root) {
  "use strict";

  const model = root.VGO.model;
  const legalSet = root.VGO.legalSet;
  const game = root.VGO.game;

  const APP_RADIUS = 39 / 700;   // the reference app's default stone radius

  // Deterministic PRNG so every run compares the same boards.
  function prng(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  }

  // Grow a position by repeated legal random placement. Uses the real
  // transaction, so grown boards are always reachable by actual play.
  function grow(count, radius, seed) {
    const random = prng(seed);
    let position = model.createPosition({ stones: [], toMove: "B", radius: radius });
    let guard = 0;
    while (position.stones.length < count && guard++ < 40000) {
      const x = random(), y = random();
      if (!legalSet.contains(position, x, y)) continue;
      const result = game.place(position, x, y);
      if (result.ok) position = result.position;
    }
    return position;
  }

  function fixed(stones, radius) {
    return model.createPosition({ stones: stones, toMove: "B", radius: radius });
  }

  // Every case carries a note describing which part of the geometry it stresses.
  function build() {
    const r = APP_RADIUS;
    const cases = [];

    cases.push({
      name: "empty",
      note: "no stones; settled set is empty",
      position: fixed([], r),
      expectedArea: 0,
    });

    cases.push({
      name: "single",
      note: "one stone in open space; exact answer is a disk of radius r",
      position: fixed([{ x: 0.5, y: 0.5, c: "B" }], r),
      expectedArea: Math.PI * r * r,
    });

    cases.push({
      name: "boot",
      note: "the reference app's opening position",
      position: fixed([
        { x: .30, y: .35, c: "B" }, { x: .45, y: .30, c: "B" }, { x: .40, y: .50, c: "B" },
        { x: .70, y: .65, c: "W" }, { x: .60, y: .78, c: "W" }, { x: .80, y: .45, c: "W" },
      ], r),
    });

    cases.push({
      name: "tangent",
      note: "two stones exactly 2r apart; a degenerate single-point contact",
      position: fixed([
        { x: 0.5 - r, y: 0.5, c: "B" }, { x: 0.5 + r, y: 0.5, c: "W" },
      ], r),
    });

    cases.push({
      name: "corner",
      note: "stones jammed into a corner; board-edge features dominate",
      position: fixed([
        { x: r, y: r, c: "B" }, { x: 3 * r, y: r, c: "W" }, { x: r, y: 3 * r, c: "W" },
        { x: 3 * r, y: 3 * r, c: "B" }, { x: 5 * r, y: r, c: "B" },
      ], r),
    });

    cases.push({
      name: "sealed",
      note: "board fully covered at r=0.25; every group settled, one closed loop",
      position: fixed([
        { x: 0.25, y: 0.25, c: "B" }, { x: 0.75, y: 0.25, c: "W" },
        { x: 0.75, y: 0.75, c: "B" }, { x: 0.25, y: 0.75, c: "W" },
      ], 0.25),
      expectedArea: 1,
    });

    cases.push({ name: "grown-12", note: "sparse midgame", position: grow(12, r, 12345) });
    cases.push({ name: "grown-24", note: "dense midgame", position: grow(24, r, 777) });
    cases.push({ name: "grown-40", note: "very dense; small legal set", position: grow(40, r, 24680) });

    cases.push({
      name: "big-r-8",
      note: "large stones; few cells, large settled fraction",
      position: grow(8, 0.11, 4242),
    });

    // Hexagonal close packing: every interior stone is at exactly 2r from six
    // neighbours, so the legal set degenerates to isolated points and triple
    // contacts are everywhere. The hardest case for the epsilon policy.
    const hexRadius = 0.06, hexStep = 2 * hexRadius, hexRow = hexStep * Math.sin(Math.PI / 3);
    const hexStones = [];
    for (let row = 0; ; row++) {
      const y = hexRadius + row * hexRow;
      if (y > 1 - hexRadius + 1e-12) break;
      for (let column = 0; ; column++) {
        const x = hexRadius + (row % 2 ? hexRadius : 0) + column * hexStep;
        if (x > 1 - hexRadius + 1e-12) break;
        hexStones.push({ x: x, y: y, c: (row + column) % 2 ? "W" : "B" });
      }
    }
    cases.push({
      name: "hex-packed",
      note: "exact hexagonal packing; every contact tangent, triple points everywhere",
      position: fixed(hexStones, hexRadius),
    });

    // The adversary for any sweep that prunes by distance: a packed board with
    // one interior stone removed, so L is nonempty but minuscule. No empty-L
    // fast path applies, and T stays board-scale on almost every ray.
    const gapStones = hexStones.filter(function (stone, index) {
      return index !== Math.floor(hexStones.length / 2);
    });
    cases.push({
      name: "hex-gap",
      note: "hex packing minus one interior stone; L is nonempty but tiny",
      position: fixed(gapStones, hexRadius),
    });

    // Square lattice at exactly 2r: four-way exact tangency, a different degeneracy.
    const gridRadius = 0.07, gridStones = [];
    for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
      const x = gridRadius + i * 2 * gridRadius, y = gridRadius + j * 2 * gridRadius;
      if (x > 1 - gridRadius + 1e-12 || y > 1 - gridRadius + 1e-12) continue;
      gridStones.push({ x: x, y: y, c: (i + j) % 2 ? "W" : "B" });
    }
    cases.push({
      name: "grid-packed",
      note: "square lattice at exactly 2r; four-way exact tangency",
      position: fixed(gridStones, gridRadius),
    });

    cases.push({
      name: "tiny-r",
      note: "small stones, many features per neighbourhood",
      position: grow(30, 0.012, 31337),
    });

    cases.push({
      name: "huge-r",
      note: "one stone nearly filling the board; inset box almost degenerate",
      position: fixed([{ x: 0.5, y: 0.5, c: "B" }], 0.4),
    });

    cases.push({
      name: "invalid-overlap",
      note: "overlapping stones; an invalid diagram must still render consistently",
      position: fixed([
        { x: 0.50, y: 0.50, c: "B" }, { x: 0.53, y: 0.50, c: "W" },
        { x: 0.20, y: 0.20, c: "B" },
      ], r),
    });

    return cases;
  }

  root.CORPUS = Object.freeze({ build: build, grow: grow, prng: prng, APP_RADIUS: APP_RADIUS });
})(globalThis);
