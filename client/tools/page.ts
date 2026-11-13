// Builds one self-contained page: pages/<name>.html with pages/<name>.ts bundled into it
// and the rules' wasm inlined, so it opens from disk or publishes as an artifact.
//
//   bun tools/page.ts <name> <out.html> [key=file ...]
//
// Each key=file is read as text and handed to the page as `data[key]` (see src/page.ts).

import { readFileSync, writeFileSync } from "fs";

const [name, out, ...pairs] = process.argv.slice(2);
if (!name || !out) throw new Error("usage: bun tools/page.ts <name> <out.html> [key=file ...]");

const result = await Bun.build({
  entrypoints: [new URL(`../pages/${name}.ts`, import.meta.url).pathname],
  target: "browser",
  format: "iife",
  minify: true,
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}
const script = await result.outputs[0]!.text();
const wasm = readFileSync(new URL("../vendor/vgo_wasm_bg.wasm", import.meta.url)).toString("base64");
const data: Record<string, string> = {};
for (const pair of pairs) {
  const at = pair.indexOf("=");
  if (at < 1) throw new Error(`expected key=file, got ${pair}`);
  data[pair.slice(0, at)] = readFileSync(pair.slice(at + 1), "utf8");
}

// "</script" inside inlined text would end the element early.
const safe = (text: string) => text.replace(/<\/script/gi, "<\\/script");
const page = readFileSync(new URL(`../pages/${name}.html`, import.meta.url), "utf8")
  .replace("<!--DATA-->", () =>
    `<script type="application/octet-stream" id="vgo-wasm">${wasm}</script>` +
    `<script type="application/json" id="vgo-data">${safe(JSON.stringify(data))}</script>`)
  .replace("<!--SCRIPT-->", () => `<script>${safe(script)}</script>`);
writeFileSync(out, page);
console.log(`${out}: ${(page.length / 1024).toFixed(0)} KB (wasm ${(wasm.length / 1024).toFixed(0)} KB as base64)`);
