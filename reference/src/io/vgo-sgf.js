(function (root) {
  "use strict";

  const model = root.VGO.model;

  function format(value) {
    return Number(value).toFixed(5);
  }

  function serialize(position) {
    function points(color) {
      return position.stones
        .filter(function (stone) { return stone.c === color; })
        .map(function (stone) { return "[" + format(stone.x) + "," + format(stone.y) + "]"; })
        .join("");
    }
    let output = "(;FF[4]GM[VGO]SZ[1]RA[" + format(position.radius) + "]PL[" + position.toMove + "]";
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
    const playerMatch = /PL\[([BW])\]/.exec(text);
    const fallback = defaults || {};
    return model.createPosition({
      radius: radiusMatch ? Number(radiusMatch[1]) : fallback.radius,
      stones: stones("AB", "B").concat(stones("AW", "W")),
      toMove: playerMatch ? playerMatch[1] : (fallback.toMove || "B"),
      passes: 0,
      phase: "playing",
    });
  }

  root.VGO.sgf = Object.freeze({ serialize: serialize, parse: parse });
})(globalThis);
