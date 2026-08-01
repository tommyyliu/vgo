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
    if (!doomed.size) return { position: position, count: 0 };
    return {
      position: model.update(position, {
        stones: position.stones.filter(function (_, index) { return !doomed.has(index); }),
      }),
      count: doomed.size,
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

    let selfRemoval = { position: next, count: 0 };
    if (activeRules.selfCapture === "remove") {
      selfRemoval = removeSettled(next, nextAnalysis, mover);
      next = selfRemoval.position;
      if (selfRemoval.count) nextAnalysis = analysis.analyze(next);
    }

    // A placement that leaves the board exactly as it was is a pass, not a
    // move: a lone stone with no liberties, self-captured on arrival, taking
    // nothing with it. Resetting the pass counter for it made it a *better*
    // stall than passing -- two passes end the game and score it, two no-op
    // suicides end nothing -- and both sides learned to abuse that. Must match
    // after_placement in crates/vgo-core/src/model.rs.
    const changed = next.stones.length !== position.stones.length;
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
