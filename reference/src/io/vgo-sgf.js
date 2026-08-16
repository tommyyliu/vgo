(function (root) {
  "use strict";

  const model = root.VGO.model;

  // Shortest string that parses back to the identical double, which is what
  // `String` on a number gives. Not a fixed decimal count: legality here turns
  // on differences of 1e-6 (SNAP_MARGIN) and 1e-7 (coordinateEpsilon), and
  // `toFixed(5)` moves a stone by up to 7.1e-6 -- enough to turn a legal
  // position into an overlapping one on reload, and enough that a saved game
  // could not be used to diagnose one.
  //
  // Two stalled games were undiagnosable for exactly this reason: every
  // apparent overlap in the saved position was smaller than the rounding that
  // produced it, so the file could not distinguish a real defect from its own
  // artefact. Ordinary coordinates stay short -- 0.5 prints as "0.5" -- and
  // only values that genuinely need the digits get them.
  function format(value) {
    return String(Number(value));
  }

  function serialize(position) {
    function points(color) {
      return position.stones
        .filter(function (stone) { return stone.c === color; })
        .map(function (stone) { return "[" + format(stone.x) + "," + format(stone.y) + "]"; })
        .join("");
    }
    // KM is always written, even at zero. Komi changes who won, so a record
    // that omits it is a record two readers can score differently -- exactly
    // the disagreement the field exists to prevent.
    let output = "(;FF[4]GM[VGO]SZ[1]RA[" + format(position.radius) +
      "]KM[" + format(position.komi) + "]PL[" + position.toMove + "]";
    const black = points("B"), white = points("W");
    if (black) output += "AB" + black;
    if (white) output += "AW" + white;
    return output + ")";
  }

  function parse(text, defaults) {
    if (typeof text !== "string" || !/GM\[VGO\]/.test(text)) {
      throw new Error("Not a VGO position.");
    }
    function values(property) {
      const match = new RegExp(property + "((?:\\[[^\\]]*\\])+)").exec(text);
      if (!match) return [];
      return Array.from(match[1].matchAll(/\[([^\]]*)\]/g), function (item) { return item[1]; });
    }
    function stones(property, color) {
      return values(property).map(function (value) {
        const pair = value.split(",");
        if (pair.length !== 2 || !Number.isFinite(Number(pair[0])) || !Number.isFinite(Number(pair[1]))) {
          throw new Error("Invalid " + property + " coordinate.");
        }
        return { x: Number(pair[0]), y: Number(pair[1]), c: color };
      });
    }
    const radiusMatch = /RA\[([-+0-9.eE]+)\]/.exec(text);
    const komiMatch = /KM\[([-+0-9.eE]+)\]/.exec(text);
    const playerMatch = /PL\[([BW])\]/.exec(text);
    const fallback = defaults || {};
    return model.createPosition({
      radius: radiusMatch ? Number(radiusMatch[1]) : fallback.radius,
      // A record written before KM existed scores at zero, which is what those
      // games were actually played at.
      komi: komiMatch ? Number(komiMatch[1]) : (fallback.komi || 0),
      stones: stones("AB", "B").concat(stones("AW", "W")),
      toMove: playerMatch ? playerMatch[1] : (fallback.toMove || "B"),
      passes: 0,
      phase: "playing",
    });
  }

  /* Game records.

     The position format above uses only SGF's setup properties, so it can say
     what a board looks like but not how it got there. A record adds SGF's own
     move nodes — `;B[x,y]`, `;W[x,y]`, an empty value for a pass — and its own
     variation syntax, where a branch is a parenthesised subtree.

     `serialize` and `parse` are unchanged, so every existing position still
     round-trips and a record degrades to its setup when read by `parse`.

     Structure cannot be recovered by matching properties one at a time, so this
     half needs a real parser:

       tree     := "(" node+ tree* ")"
       node     := ";" property*
       property := ident "[" text "]"+
  */
  function parseTree(text) {
    let index = 0;
    function skip() {
      while (index < text.length && /\s/.test(text[index])) index += 1;
    }
    function parseValues() {
      const values = [];
      skip();
      while (text[index] === "[") {
        index += 1;
        let value = "";
        while (index < text.length && text[index] !== "]") {
          if (text[index] === "\\") index += 1;         // SGF escape
          value += text[index];
          index += 1;
        }
        if (text[index] !== "]") throw new Error("Unterminated property value.");
        index += 1;
        values.push(value);
        skip();
      }
      return values;
    }
    function parseNode() {
      index += 1;                                        // ";"
      const properties = {};
      skip();
      while (index < text.length && /[A-Za-z]/.test(text[index])) {
        let identifier = "";
        while (index < text.length && /[A-Za-z]/.test(text[index])) {
          identifier += text[index];
          index += 1;
        }
        properties[identifier] = parseValues();
        skip();
      }
      return properties;
    }
    function parseSubtree() {
      skip();
      if (text[index] !== "(") throw new Error("Expected a game tree.");
      index += 1;
      const nodes = [];
      skip();
      while (text[index] === ";") { nodes.push(parseNode()); skip(); }
      if (!nodes.length) throw new Error("A game tree needs at least one node.");
      const children = [];
      while (text[index] === "(") { children.push(parseSubtree()); skip(); }
      if (text[index] !== ")") throw new Error("Unterminated game tree.");
      index += 1;
      return { nodes: nodes, children: children };
    }
    const tree = parseSubtree();
    skip();
    return tree;
  }

  function moveFromProperties(properties) {
    for (const color of ["B", "W"]) {
      if (!Object.prototype.hasOwnProperty.call(properties, color)) continue;
      const raw = properties[color][0];
      if (raw === undefined || raw === "") return { c: color, pass: true };
      const pair = String(raw).split(",");
      if (pair.length !== 2 || !Number.isFinite(Number(pair[0])) || !Number.isFinite(Number(pair[1]))) {
        throw new Error("Invalid " + color + " move coordinate.");
      }
      return { c: color, x: Number(pair[0]), y: Number(pair[1]), pass: false };
    }
    return null;
  }

  // A record parses to its setup position plus a plain tree of moves. Replaying
  // it is the game tree's job, so this module stays free of the rules.
  function parseRecord(text, defaults) {
    if (typeof text !== "string" || !/GM\[VGO\]/.test(text)) {
      throw new Error("Not a VGO record.");
    }
    const tree = parseTree(text);
    // Setup properties are read by the position parser, which ignores move
    // nodes, so a record and a plain position agree on their starting board.
    const setup = parse(text, defaults);

    // A subtree is a linear run of nodes with its branches hanging off the last
    // one. The root subtree's first node carries the setup, never a move.
    function convert(subtree, skipFirst) {
      const chain = [];
      const nodes = skipFirst ? subtree.nodes.slice(1) : subtree.nodes;
      for (const properties of nodes) {
        const move = moveFromProperties(properties);
        if (move) chain.push({ move: move, children: [] });
      }
      const branches = [];
      for (const child of subtree.children) {
        for (const head of convert(child, false)) branches.push(head);
      }
      if (!chain.length) return branches;
      for (let step = 0; step < chain.length - 1; step += 1) {
        chain[step].children = [chain[step + 1]];
      }
      chain[chain.length - 1].children = branches;
      return [chain[0]];
    }
    return { setup: setup, variations: convert(tree, true) };
  }

  function moveText(move) {
    if (move.pass) return ";" + move.c + "[]";
    return ";" + move.c + "[" + format(move.x) + "," + format(move.y) + "]";
  }

  // Emits a linear run inline and parenthesises only genuine branches, which is
  // what makes a mainline readable next to its variations.
  function serializeRecord(node) {
    function below(current) {
      let text = "";
      let step = current;
      for (;;) {
        if (!step.children.length) return text;
        if (step.children.length === 1) {
          step = step.children[0];
          text += moveText(step.move);
          continue;
        }
        for (const child of step.children) {
          text += "(" + moveText(child.move) + below(child) + ")";
        }
        return text;
      }
    }
    const setup = serialize(node.position);
    return setup.slice(0, setup.length - 1) + below(node) + ")";
  }

  root.VGO.sgf = Object.freeze({
    serialize: serialize,
    parse: parse,
    parseRecord: parseRecord,
    serializeRecord: serializeRecord,
  });
})(globalThis);
