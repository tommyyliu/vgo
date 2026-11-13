# Voronoi Go board

A board built on the Rust rules: `vgo-core` compiled to WebAssembly is the only
place the client learns a rule. Everything drawn on top is the client's own.

```bash
bun install                        # once
bun run build:wasm                 # crates/vgo-wasm -> client/vendor/
bun test                           # the rules door against recorded games
bun run check                      # types
bun tools/replay.ts <dir-or-sgf>   # replay records through the rules, report disagreements
```

`src/rules.ts` is the door; `crates/vgo-wasm/src/rules.rs` is the Rust half.
