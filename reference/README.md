# JavaScript Reference

This directory is a self-contained behavioral reference for Voronoi Go. It is
kept intentionally independent from the future training implementation.

## Contents

- [`RULES.md`](RULES.md): normative game rules.
- [`AXIOMS.md`](AXIOMS.md): proved consequences used by implementations.
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
