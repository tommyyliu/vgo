# Voronoi Go app

The playable game, with its own JavaScript engine that also serves as a
behavioural oracle for the Rust engine. `web-bot/` adds a browser opponent to it. The rules and lemmas it implements are in [`spec/`](../spec/RULES.md).

## Contents

- [`ARCHITECTURE.md`](ARCHITECTURE.md): design of the JavaScript reference.
- [`src/`](src/): pure geometry and game modules.
- [`index.html`](index.html): interactive
  browser application.
- [`tests/`](tests/): engine and UI regression tests.

Run the complete reference suite from the repository root:

```powershell
.\js-app\tests\run-tests.ps1
```

Future engines should share conformance fixtures with this implementation. They
should not silently reinterpret numerical tolerances as game rules.
