# Core algorithms

What the engine computes and why it works, without the code. Each entry names the lemmas
in [`LEMMAS.md`](LEMMAS.md) it relies on and where it lives. `n` is the stone count.

## Rules

### Capture by cell vertices

Whether a group lives looks like a question about infinitely many board points. A
placement's advantage over an owner is linear in the board point, so it peaks at a cell
vertex, and one strictly challenged vertex keeps a group alive.

- **Relies on:** A2, A3, A5. Comparisons are exact: an interval check, then exact rationals.
- **Cost:** a few vertices per cell, stopping at the first witness per group.
- **Code:** `vgo-core/src/analysis.rs`, `numeric.rs`

### Legal space by enumerated features

The closest legal point to a query lies on one of a few feature types: the query itself,
a side of the inset, a point on an exclusion circle, or an intersection of those. So a
continuous set is searched through a finite candidate list.

- **Relies on:** A6, E7, N1.
- **Cost:** buckets keep each legality test near constant.
- **Code:** `vgo-core/src/legal_set/`

### Voronoi cells by sorted clipping

Clip the square against bisectors in order of distance from the stone, and stop once the
next stone is more than twice the polygon's circumradius away. A cell has about six
neighbours, so most stones are never clipped.

- **Relies on:** A2, E4, N3.
- **Cost:** about 8 clips per cell instead of `n - 1`.
- **Code:** `vgo-core/src/voronoi.rs`, `compute_cell`

### Cell reuse between positions

One new stone changes few cells. Each cell records the distance beyond which a new stone
cannot enter its clip sequence, and a child shares those cells with its parent.

- **Relies on:** E5, E8.
- **Cost:** recompute only the cells near the new stone.
- **Code:** `vgo-core/src/voronoi.rs`, `compute`; `model.rs`, `shared_cells`

## Rasterization

### Nearest and second-nearest stones by tile

The ridge and settled planes need each pixel's two nearest stones. A hierarchy of tiles
narrows the candidates, discarding stones that cannot be among the two nearest anywhere in
a tile. Discarding only selects candidates, so stored distances stay exact.

- **Relies on:** E1, N2.
- **Cost:** a few candidates per pixel instead of `n`.
- **Code:** `vgo-raster/src/packed.rs`, `sweep_tile`

### Stone planes by stamping

A stone plane only asks whether a pixel is inside some disc, so each stone stamps its own
disc and no per-colour distance is kept.

- **Relies on:** E2.
- **Code:** `vgo-raster/src/packed.rs`, `stamp_disc`

### Settled plane by distance comparison

A pixel is settled when its nearest stone is no farther than the legal set. An exact
distance transform of the sampled legal set gives two-sided bounds that decide most pixels,
and an exact legal-space test decides the narrow band between them.

- **Relies on:** A15, E3, N1. **Not proved exact: see O1.**
- **Cost:** linear in pixels, plus about 1,300 exact tests per 128-pixel big-board raster.
- **Code:** `vgo-raster/src/settled_bounds.rs`, `edt.rs`

## Search

### Progressive widening

A node's action set is grown as about a coefficient times the square root of its visits,
so search can work in a continuous action space. The coefficient is a tuned strength
parameter with no lemma behind it.

- **Code:** `vgo-search/src/mcts/`

### Coarse-to-fine proposals

The policy is a spatial map. Search pools a legal fine grid, draws a coarse region, then a
fine cell. A cell whose centre is illegal may still hold legal area, so cells near a
boundary are snapped to a legal point inside the cell.

- **Relies on:** A6, E6.
- **Cost:** snapping looks only at nearby stones and vertices.
- **Code:** `vgo-search/src/coarse_fine.rs`; `vgo-core/src/legal_set/nearest.rs`

## Learning

These are empirical choices, judged by arena play. Evidence is in
[`docs/OVERVIEW.md`](../docs/OVERVIEW.md).

- Bulk retraining from scratch on a recent window, with no promotion gate.
- Soft resignation, with thresholds calibrated from measured false positives.
- Bradley-Terry ratings over a match graph.
