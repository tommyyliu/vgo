import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { checkTree, createReplay } from "../src/replay";
import { checkRecord } from "../src/check-record";
import { testRules } from "./wasm";
import { add, create, fromSgf, line, node, path, ply, promote, remove, sibling, toSgf, type GameTree } from "../src/tree";

const rules = testRules();
const dir = new URL("./fixtures/", import.meta.url).pathname;
const records = readdirSync(dir).filter((n) => n.endsWith(".sgf")).map((n) => readFileSync(join(dir, n), "utf8").trim());
const setup = { ruleset: "vgo" as const, radius: 1 / 18, komi: 0.1, stones: [], toMove: "B" as const };

function invariants(tree: GameTree) {
  for (const n of tree.nodes.values()) {
    for (const child of n.children) expect(node(tree, child).parent).toBe(n.id);
    if (n.parent === null) expect(n.id).toBe(tree.root);
    else expect(node(tree, n.parent).children).toContain(n.id);
  }
}

describe("records", () => {
  test("every fixture writes back byte for byte", () => {
    for (const text of records) {
      const { tree, problems } = fromSgf(text);
      expect(problems).toEqual([]);
      expect(toSgf(tree)).toBe(text);
      invariants(tree);
    }
  });

  test("replay through the tree reaches the same final position as the record", () => {
    for (const text of records) {
      const { tree } = fromSgf(text);
      const replay = createReplay(rules);
      const end = line(tree, tree.root).at(-1)!;
      expect(replay.position(tree, end).stones).toEqual(checkRecord(rules, text).steps.at(-1)!.after.stones);
      expect(ply(tree, end)).toBe(checkRecord(rules, text).moves);
    }
  });

  test("a variation survives the round trip", () => {
    const { tree } = fromSgf(records[0]!);
    const third = line(tree, tree.root)[3]!;
    const varied = add(tree, node(tree, third).parent!, { kind: "pass", color: node(tree, third).move!.color }).tree;
    const text = toSgf(varied);
    expect(text).toContain(")(");
    const back = fromSgf(text).tree;
    expect(toSgf(back)).toBe(text);
    expect(back.nodes.size).toBe(varied.nodes.size);
  });

  test("an illegal move in a loaded record is cut and reported", () => {
    const { tree } = fromSgf(records[0]!);
    const ids = line(tree, tree.root);
    const first = node(tree, ids[1]!).move!;
    // Replay the first move again as move 3: a stone on top of a stone.
    const stacked = add(tree, ids[2]!, { ...first }).tree;
    const checked = checkTree(createReplay(rules), stacked);
    expect(checked.problems).toEqual(["move 3 cut: illegal-placement"]);
    expect(checked.tree.nodes.size).toBe(tree.nodes.size);
  });
});

describe("edits", () => {
  test("playing an existing move goes to it", () => {
    const a = add(create(setup), 0, { kind: "place", x: 0.5, y: 0.5, color: "B" });
    const b = add(a.tree, 0, { kind: "place", x: 0.5, y: 0.5, color: "B" });
    expect(b.node).toBe(a.node);
    expect(b.tree).toBe(a.tree);
    const c = add(a.tree, 0, { kind: "place", x: 0.5, y: 0.50001, color: "B" });
    expect(c.node).not.toBe(a.node);
    expect(node(c.tree, 0).children).toEqual([a.node, c.node]);
  });

  test("edits leave the old tree as it was", () => {
    const a = add(create(setup), 0, { kind: "pass", color: "B" });
    const before = node(a.tree, 0).children;
    add(a.tree, 0, { kind: "place", x: 0.3, y: 0.3, color: "B" });
    expect(node(a.tree, 0).children).toBe(before);
  });

  test("remove drops a subtree; promote makes a branch the main line", () => {
    let tree = create(setup);
    const m1 = add(tree, 0, { kind: "place", x: 0.2, y: 0.2, color: "B" }); tree = m1.tree;
    const m2 = add(tree, m1.node, { kind: "place", x: 0.8, y: 0.8, color: "W" }); tree = m2.tree;
    const v2 = add(tree, m1.node, { kind: "place", x: 0.8, y: 0.2, color: "W" }); tree = v2.tree;
    const v3 = add(tree, v2.node, { kind: "pass", color: "B" }); tree = v3.tree;
    expect(line(tree, 0)).toEqual([0, m1.node, m2.node]);
    expect(sibling(tree, m2.node, 1)).toBe(v2.node);
    expect(sibling(tree, v2.node, 1)).toBe(v2.node);

    const promoted = promote(tree, v3.node);
    expect(line(promoted, 0)).toEqual([0, m1.node, v2.node, v3.node]);
    expect(path(promoted, v3.node)).toEqual([0, m1.node, v2.node, v3.node]);
    invariants(promoted);

    const pruned = remove(tree, v2.node);
    expect(pruned.nodes.has(v3.node)).toBe(false);
    expect(node(pruned, m1.node).children).toEqual([m2.node]);
    invariants(pruned);
    expect(() => remove(tree, 0)).toThrow();
  });

  test("a node's position is cached by id across edits", () => {
    const replay = createReplay(rules);
    const a = add(create(setup), 0, { kind: "place", x: 0.5, y: 0.5, color: "B" });
    const first = replay.step(a.tree, a.node);
    const b = add(a.tree, 0, { kind: "place", x: 0.2, y: 0.2, color: "B" });
    expect(replay.step(b.tree, a.node)).toBe(first);
  });
});
