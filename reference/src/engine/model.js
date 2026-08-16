(function (root) {
  "use strict";

  const N = root.VGO.numeric;
  const DEFAULT_RULES = Object.freeze({
    selfCapture: "remove",
    repetition: "unrestricted",
    ending: "two-passes",
    scoring: "voronoi-area",
  });

  function other(color) {
    return color === "B" ? "W" : "B";
  }

  function createPosition(input) {
    const source = input || {};
    const stones = (source.stones || []).map(function (stone) {
      return Object.freeze({ x: Number(stone.x), y: Number(stone.y), c: stone.c });
    });
    return Object.freeze({
      radius: Number(source.radius === undefined ? 0 : source.radius),
      stones: Object.freeze(stones),
      toMove: source.toMove === "W" ? "W" : "B",
      passes: Math.max(0, Math.floor(Number(source.passes) || 0)),
      phase: source.phase === "finished" ? "finished" : "playing",
      // Area subtracted from Black's lead when the game is scored. Voronoi area
      // is a fraction of the board, so komi is one too rather than a stone
      // count: 0.18 means White is spotted eighteen percent of the board.
      //
      // Part of the position because it changes who won: the same stones under
      // different komi have different winners, so a search, a shard, and a
      // model that disagree about it are not playing the same game. `update`
      // copies the position before applying changes, so it carries through
      // every move without each call site remembering to pass it.
      komi: Number(source.komi) || 0,
    });
  }

  function update(position, changes) {
    return createPosition(Object.assign({}, position, changes));
  }

  function validate(position) {
    const issues = [];
    const r = position.radius;
    if (!Number.isFinite(r) || r <= 0 || r >= 0.5) {
      issues.push("Stone radius must be greater than 0 and less than 0.5.");
    }
    if (position.toMove !== "B" && position.toMove !== "W") {
      issues.push("Player to move must be B or W.");
    }

    const minDistance = 2 * r;
    for (let i = 0; i < position.stones.length; i++) {
      const stone = position.stones[i];
      if (!Number.isFinite(stone.x) || !Number.isFinite(stone.y)) {
        issues.push("Stone " + i + " has non-finite coordinates.");
        continue;
      }
      if (stone.c !== "B" && stone.c !== "W") {
        issues.push("Stone " + i + " has an invalid color.");
      }
      if (stone.x < r - N.coordinateEpsilon || stone.x > 1 - r + N.coordinateEpsilon ||
          stone.y < r - N.coordinateEpsilon || stone.y > 1 - r + N.coordinateEpsilon) {
        issues.push("Stone " + i + " does not fit on the board.");
      }
      for (let j = 0; j < i; j++) {
        const otherStone = position.stones[j];
        const distance = Math.hypot(stone.x - otherStone.x, stone.y - otherStone.y);
        if (distance < minDistance - N.coordinateEpsilon) {
          issues.push("Stones " + j + " and " + i + " overlap.");
        }
      }
    }
    return Object.freeze({ playable: issues.length === 0, issues: Object.freeze(issues) });
  }

  root.VGO.model = Object.freeze({
    DEFAULT_RULES: DEFAULT_RULES,
    createPosition: createPosition,
    update: update,
    validate: validate,
    other: other,
  });
})(globalThis);
