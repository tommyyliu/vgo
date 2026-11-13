// The rules for tests, from the wasm `bun run build:wasm` makes. A missing build is the
// usual reason every test fails at once, so say so plainly.

import { existsSync, readFileSync } from "fs";
import { loadRules, type Rules } from "../src/rules";

export function testRules(): Rules {
  const path = new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url);
  if (!existsSync(path)) throw new Error("client/vendor/ has no wasm: run `bun run build:wasm` first");
  return loadRules(readFileSync(path));
}
