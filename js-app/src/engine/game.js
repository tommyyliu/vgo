(function (root) {
  "use strict";

  const model = root.VGO.model;
  const legalSet = root.VGO.legalSet;
  const analysis = root.VGO.analysis;

  function removeSettled(position, currentAnalysis, color) {
    const doomed = new Set();
    for (let i = 0; i < position.stones.length; i++) {
      if (position.stones[i].c === color && currentAnalysis.settledGroups.has(currentAnalysis.geometry.group[i])) {
        doomed.add(i);
      }
    }
    // Callers need the identities, not just the tally: telling a no-op
    // placement from a genuine one turns on *which* stone died, not how many.
    if (!doomed.size) return { position: position, count: 0, doomed: doomed };
    return {
      position: model.update(position, {
        stones: position.stones.filter(function (_, index) { return !doomed.has(index); }),
      }),
      count: doomed.size,
      doomed: doomed,
    };
  }

  function place(position, x, y, rules) {
    const activeRules = rules || model.DEFAULT_RULES;
    const initialAnalysis = analysis.analyze(position);
    if (position.phase !== "playing") {
      return { ok: false, reason: "finished", position: position, analysis: initialAnalysis, events: [] };
    }
    if (!initialAnalysis.validation.playable) {
      return { ok: false, reason: "invalid-position", position: position, analysis: initialAnalysis, events: [] };
    }
    if (!legalSet.contains(position, x, y)) {
      return { ok: false, reason: "illegal-placement", position: position, analysis: initialAnalysis, events: [] };
    }

    const mover = position.toMove;
    let next = model.update(position, {
      stones: position.stones.concat([{ x: x, y: y, c: mover }]),
      passes: 0,
      phase: "playing",
    });
    let nextAnalysis = analysis.analyze(next);

    const enemyRemoval = removeSettled(next, nextAnalysis, model.other(mover));
    next = enemyRemoval.position;
    if (enemyRemoval.count) nextAnalysis = analysis.analyze(next);

    // The placed stone was appended last and belongs to the mover, so it is
    // never in the enemy removal; order-preserving removal leaves it last here.
    const placedIndex = next.stones.length - 1;

    let selfRemoval = { position: next, count: 0, doomed: new Set() };
    if (activeRules.selfCapture === "remove") {
      selfRemoval = removeSettled(next, nextAnalysis, mover);
      next = selfRemoval.position;
      if (selfRemoval.count) nextAnalysis = analysis.analyze(next);
    }

    // A placement that leaves the board exactly as it was is a pass, not a
    // move: a lone stone with no liberties, self-captured on arrival, taking
    // nothing with it. Resetting the pass counter for it made it a *better*
    // stall than passing -- two passes end the game and score it, two no-op
    // suicides end nothing -- and both sides learned to abuse that.
    //
    // The board is unchanged exactly when the stone just placed is the only
    // one removed. Stone counts do not state this -- they collide on every
    // even trade, so a placement that captures one enemy stone reads as a
    // no-op while having changed the board completely. Two such trades in a
    // row ended games at an arbitrary point, scoring a position neither player
    // had finished. Self-capture is legal and global here, so unlike in Go a
    // static count does not imply a capture happened.
    //
    // Must match place in crates/vgo-core/src/game.rs.
    const unchanged = enemyRemoval.count === 0
      && selfRemoval.doomed.size === 1
      && selfRemoval.doomed.has(placedIndex);
    const changed = !unchanged;
    const passes = changed ? 0 : position.passes + 1;
    const finished = activeRules.ending === "two-passes" && passes >= 2;
    next = model.update(next, {
      toMove: finished ? mover : model.other(mover),
      passes: passes,
      phase: finished ? "finished" : "playing",
    });
    nextAnalysis = analysis.analyze(next);
    const events = [];
    if (enemyRemoval.count) events.push({ type: "capture", color: model.other(mover), count: enemyRemoval.count });
    if (selfRemoval.count) events.push({ type: "self-capture", color: mover, count: selfRemoval.count });
    return {
      ok: true,
      position: next,
      analysis: nextAnalysis,
      events: events,
      captured: enemyRemoval.count + selfRemoval.count,
    };
  }

  function pass(position, rules) {
    const activeRules = rules || model.DEFAULT_RULES;
    const currentAnalysis = analysis.analyze(position);
    if (position.phase !== "playing") {
      return { ok: false, reason: "finished", position: position, analysis: currentAnalysis, events: [] };
    }
    if (!currentAnalysis.validation.playable) {
      return { ok: false, reason: "invalid-position", position: position, analysis: currentAnalysis, events: [] };
    }
    const passes = position.passes + 1;
    const finished = activeRules.ending === "two-passes" && passes >= 2;
    const next = model.update(position, {
      passes: passes,
      phase: finished ? "finished" : "playing",
      toMove: finished ? position.toMove : model.other(position.toMove),
    });
    const nextAnalysis = analysis.analyze(next);
    return {
      ok: true,
      position: next,
      analysis: nextAnalysis,
      events: [{ type: finished ? "game-finished" : "pass", result: finished ? nextAnalysis.result : null }],
    };
  }

  root.VGO.game = Object.freeze({
    place: place,
    pass: pass,
    removeSettled: removeSettled,
  });
})(globalThis);
