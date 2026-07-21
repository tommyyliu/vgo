# Architecture

The reference app is intentionally split by authority. Approximate rendering
may describe the rules, but it cannot decide them.

## Dependency direction

```text
model + numeric policy
        |
        v
Voronoi geometry + legal-placement geometry
        |
        v
global position analysis
        |
        v
pure game transactions + serialization
        |
        v
reference UI and approximate contour rendering
```

Dependencies only point downward in this diagram. Engine modules do not read
the DOM, renderer settings, history, hover state, or contour depth.

## Position

`VGO.model.createPosition()` is the sole constructor for game positions. A
position contains only rule-relevant state:

```text
{ radius, stones, toMove, passes, phase }
```

The position, stone array, and stones are frozen. A change creates a new
position with `VGO.model.update()`. This makes history exact and makes partial
move mutation impossible.

A diagram can be invalid and still be rendered. `VGO.model.validate()` decides
whether it is playable; both placement and passing reject invalid diagrams.

## Analysis

`VGO.analysis.analyze(position)` is the one global interpretation of a
position. It returns validation, sourced Voronoi cells, positive-edge
adjacency, groups, the legal-set vertices, settled groups, survival witnesses,
and scores.

Voronoi edges retain the clipping constraint that created them. Board edges are
identified as board constraints and shared edges as stone bisectors. A
zero-length contact is discarded before adjacency is formed, so cells meeting
at only a vertex never connect groups.

Capture uses the finite cell-vertex criterion proved in `AXIOMS.md`. It does
not use the quadtree contour.

## Transactions

`VGO.game.place()` and `VGO.game.pass()` are pure functions. They return either
a rejected result or a new position, its analysis, and explicit events. A
placement performs exactly two simultaneous removal stages:

1. Analyze the provisional stone and remove all settled opponents.
2. Reanalyze and remove all settled mover groups.

A9 and A12 prove that neither stage needs a fixed-point loop.

The current rules object explicitly chooses removable self-capture,
unrestricted repetition, two-pass ending, and Voronoi-area scoring.

## Rendering

`VGO.settledContour.compute(position, depth)` is pure but approximate. It uses
the exact settled-point predicate and a Lipschitz-pruned quadtree to create SVG
loops. Its output is presentation data only. Topology diagnostics are retried
once at greater depth and then logged with the complete serialized position.

The HTML page owns only UI concerns: history, hover state, contour detail,
controls, SVG construction, and downloads.

## Numerical boundary

All tolerances live in `src/engine/numeric.js`. `rho` remains the geometric
cell-vertex distance from A4. `captureMargin` is a documented implementation
deadband, not a rule; replacing it with an error bound and certified fallback
remains ticket 001.

## Verification

Run all browser and engine checks from the repository root:

```powershell
.\reference\tests\run-tests.ps1
```

The engine fixtures call the production modules directly. The UI fixture loads
the real reference page and exercises conclusion, undo, setup restoration, and
invalid-diagram behavior.
