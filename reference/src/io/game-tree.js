(function (root) {
  "use strict";

  const model = root.VGO.model;
  const game = root.VGO.game;

  /* A game record as a tree of moves.

     Positions stay immutable, exactly as everywhere else. The tree around them
     is deliberately mutable: adding a variation is an edit to a record, not a
     new game. A node therefore owns three things that never change once it
     exists — the move that created it, the position that move produced, and its
     parent — plus a child list that grows.

     Caching the position on the node is safe for the same reason. A position
     cannot be mutated, so a node's position is a pure function of the path that
     reaches it and never needs recomputing.

     The first child of a node is its main line. Variations are the remaining
     children, in the order they were played. */

  function node(move, position, parent) {
    return { move: move, position: position, parent: parent, children: [] };
  }

  function create(position) {
    return node(null, position, null);
  }

  function sameMove(move, candidate) {
    if (!move || !candidate) return false;
    if (move.pass || candidate.pass) return Boolean(move.pass) === Boolean(candidate.pass);
    return move.x === candidate.x && move.y === candidate.y;
  }

  // Replaying an existing move navigates to it rather than duplicating it, so
  // stepping back and repeating a line does not fork the tree.
  function existingChild(current, move) {
    for (const child of current.children) {
      if (sameMove(child.move, move)) return child;
    }
    return null;
  }

  function play(current, x, y, rules) {
    const move = { c: current.position.toMove, x: x, y: y, pass: false };
    const known = existingChild(current, move);
    if (known) return { ok: true, node: known, created: false, result: null };
    const result = game.place(current.position, x, y, rules);
    if (!result.ok) return { ok: false, reason: result.reason, node: current };
    const child = node(move, result.position, current);
    current.children.push(child);
    return { ok: true, node: child, created: true, result: result };
  }

  function pass(current, rules) {
    const move = { c: current.position.toMove, pass: true };
    const known = existingChild(current, move);
    if (known) return { ok: true, node: known, created: false, result: null };
    const result = game.pass(current.position, rules);
    if (!result.ok) return { ok: false, reason: result.reason, node: current };
    const child = node(move, result.position, current);
    current.children.push(child);
    return { ok: true, node: child, created: true, result: result };
  }

  function path(current) {
    const nodes = [];
    for (let step = current; step; step = step.parent) nodes.push(step);
    return nodes.reverse();
  }

  function ply(current) {
    return path(current).length - 1;
  }

  function mainline(current) {
    const nodes = [current];
    for (let step = current; step.children.length; ) {
      step = step.children[0];
      nodes.push(step);
    }
    return nodes;
  }

  function isMainline(current) {
    for (let step = current; step.parent; step = step.parent) {
      if (step.parent.children[0] !== step) return false;
    }
    return true;
  }

  // Siblings let the page offer a choice wherever the record branches.
  function siblings(current) {
    return current.parent ? current.parent.children : [current];
  }

  function branchPoints(current) {
    return path(current).filter(function (step) { return step.children.length > 1; });
  }

  // Promotion makes a variation the main line without discarding the old one.
  function promote(current) {
    const parent = current.parent;
    if (!parent) return false;
    const index = parent.children.indexOf(current);
    if (index <= 0) return false;
    parent.children.splice(index, 1);
    parent.children.unshift(current);
    return true;
  }

  function remove(current) {
    const parent = current.parent;
    if (!parent) return false;
    const index = parent.children.indexOf(current);
    if (index < 0) return false;
    parent.children.splice(index, 1);
    return true;
  }

  function count(current) {
    let total = 1;
    for (const child of current.children) total += count(child);
    return total;
  }

  function find(current, predicate) {
    if (predicate(current)) return current;
    for (const child of current.children) {
      const hit = find(child, predicate);
      if (hit) return hit;
    }
    return null;
  }

  /* Rescore a whole record at a new komi.

     Komi belongs to the game rather than to a move, so it cannot be changed at
     one node and left alone at the others -- every cached position has to move
     together or the record scores two ways depending on where you stand in it.

     Unlike a radius change this invalidates nothing: komi does not touch
     legality, capture or settlement, so every move in the record stays legal
     and the tree keeps its exact shape.

     A fresh tree rather than an in-place rewrite, because callers hold node
     references to undo with. Mutating the cached positions would reach back
     through those references and rescore the history they were taken to
     preserve, so undo would not restore the old komi. The returned map carries
     old nodes to new ones, so a caller can keep its place. */
  function rescore(root, komi) {
    const map = new Map();
    function clone(source, parent) {
      const copy = node(source.move, model.update(source.position, { komi: komi }), parent);
      map.set(source, copy);
      for (const child of source.children) copy.children.push(clone(child, copy));
      return copy;
    }
    return { root: clone(root, null), map: map };
  }

  /* Rebuild a tree from a parsed record.

     Replaying is the only way to recover positions, because a record stores
     moves and the rules decide what each one produces. A move whose colour
     disagrees with the side to move is a corrupt record rather than something to
     paper over, so it is reported instead of silently applied. */
  function fromRecord(record, rules) {
    const start = create(record.setup);
    const rejected = [];
    function apply(parent, entry) {
      const expected = parent.position.toMove;
      if (entry.move.c !== expected) {
        rejected.push({ move: entry.move, reason: "out-of-turn", expected: expected });
        return;
      }
      const outcome = entry.move.pass
        ? pass(parent, rules)
        : play(parent, entry.move.x, entry.move.y, rules);
      if (!outcome.ok) {
        rejected.push({ move: entry.move, reason: outcome.reason, expected: expected });
        return;
      }
      for (const child of entry.children) apply(outcome.node, child);
    }
    for (const entry of record.variations) apply(start, entry);
    return { root: start, rejected: rejected };
  }

  root.VGO.gameTree = Object.freeze({
    create: create,
    fromRecord: fromRecord,
    rescore: rescore,
    play: play,
    pass: pass,
    path: path,
    ply: ply,
    mainline: mainline,
    isMainline: isMainline,
    siblings: siblings,
    branchPoints: branchPoints,
    promote: promote,
    remove: remove,
    count: count,
    find: find,
  });
})(globalThis);
