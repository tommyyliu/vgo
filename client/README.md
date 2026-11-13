# Voronoi Go board

A board built on the Rust rules: `vgo-core` compiled to WebAssembly is the only
place the client learns a rule. Everything drawn on top is the client's own.

```bash
bun install                        # once
bun run build:wasm                 # crates/vgo-wasm -> client/vendor/
bun test                           # the rules door against recorded games
bun run check                      # types
bun tools/replay.ts <dir-or-sgf>   # replay records through the rules, report disagreements
bun tools/still.ts <sgf> <moves> <px> <out.svg>   # one position as SVG
bun tools/page.ts <name> <out.html> [key=file ...]   # pages/<name> as one self-contained file
```

Review 2, for example:
`bun tools/page.ts stepping dist/stepping.html record=../artifacts/old/main/experiments-2026-10-14/norm1-sim-r18-sgf/res-ab-game028-candB-natural.sgf`

`src/rules.ts` is the door; `crates/vgo-wasm/src/rules.rs` is the Rust half.
