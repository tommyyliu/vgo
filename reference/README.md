# JavaScript Reference

A self-contained JavaScript implementation of Voronoi Go, used as a behavioural
oracle. The rules and lemmas it implements are in [`spec/`](../spec/RULES.md).

## Contents

- [`ARCHITECTURE.md`](ARCHITECTURE.md): design of the JavaScript reference.
- [`src/`](src/): pure geometry and game modules.
- [`js-reference/voronoi_go.html`](js-reference/voronoi_go.html): interactive
  browser application.
- [`tests/`](tests/): engine and UI regression tests.

Run the complete reference suite from the repository root:

```powershell
.\reference\tests\run-tests.ps1
```

Future engines should share conformance fixtures with this implementation. They
should not silently reinterpret numerical tolerances as game rules.
