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

`VGO.settledContour.compute(position, knownVertices)` is pure and analytic. It
samples no field and iterates to no root. A15 removes the Voronoi partition: the
settled set is the union over stones of `{ x : ||x-s|| <= dist(x,L) }`, because a
non-nearest stone only makes the test harder. A16 makes each of those regions
star-shaped about its stone, so a two-dimensional level set becomes one scalar
equation per direction, and A17 solves that equation in closed form as a minimum
over the A6 candidate families. A candidate is admitted only once the center
realising it is verified legal, which is exactly what restricts a full circle or
line to the surviving part of `boundary L`, so `boundary L` is never built.

Loops are closed and simple by construction and carry no sampling depth, so
there is no contour topology to diagnose or repair. They overlap, so the union
is a nonzero-winding fill rather than an even-odd one. The output remains
presentation data only; capture does not use it.

The HTML page owns only UI concerns: history, hover state, controls, SVG
construction, and downloads.

The hovered move is previewed by running the real transaction and discarding
it. Because `VGO.game.place()` is pure, the preview shows exact capture,
self-capture, and post-move scores rather than a second, approximate rule
implementation in the page. The preview draws only boundaries the move would
move: a half-plane that binds nothing leaves its polygon's vertices untouched,
so unchanged edges compare equal and fall out of the diff. The transaction is
memoized on the snapped point and rendered at most once per animation frame.

The preview also shades what the move would settle, which the analytic contour
makes affordable at hover rate. Only regions whose area changes are drawn: in
open space a stone settles exactly its own radius-`r` disk by A16, so drawing
every region would ring the whole board without saying anything. What remains is
the move's own region and the neighbours it grows, which by A8 is the entire
effect of a non-capturing move.

## Records

`VGO.sgf.serialize` and `VGO.sgf.parse` still describe a single position, and a
game record is the same file with SGF's move nodes added: `;B[x,y]`, `;W[x,y]`,
an empty value for a pass, and a parenthesised subtree for a variation. A record
therefore still reads as a position through `parse`, which sees only its setup
properties. `parseRecord` needs a real parser rather than property matching,
because structure cannot be recovered one property at a time.

`VGO.gameTree` holds the record in memory. Positions stay immutable and are
cached on the node that produced them, which is sound because a node's position
is a pure function of the path that reaches it. The tree around them is mutable,
since adding a variation edits a record rather than creating a new game. The
first child is the main line and the rest are variations in the order they were
played; replaying an existing move navigates to it instead of duplicating it.

Replaying a record is the only way to recover its positions, so `fromRecord`
applies the rules move by move and reports anything it had to reject rather than
silently repairing it.

The page owns navigation. Undo reverses the last edit — a move, a load, a clear,
a radius change — and walking the tree is not an edit, so browsing a game never
fills the undo stack.

Coordinates are written with five decimals. That is cosmetic for a position and
cumulative for a record, since every replayed move is re-quantised; the game-tree
fixtures pin the current precision.

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
