# 005: Contour Topology Diagnostics

- Status: Done
- Priority: Medium
- Owner: Developer

## Problem

The adaptive settled-region renderer records `openChains` and `badVertices`, but
does not surface them. An open chain is omitted from SVG output, so a rendering
failure can silently remove settled shading.

This affects visualization only, not capture or move legality.

## Acceptance criteria

- Nonzero diagnostics emit a warning containing the VGO-SGF position, radius,
  contour depth, segment count, and vertex-degree summary.
- Open chains or degree-1 endpoints trigger a bounded retry at greater local or
  global depth.
- Exact degree-4 contacts are handled or reported as singular junctions rather
  than automatically treated as cracks.
- Tests cover board closure, exact tangency, symmetric saddles, a fully settled
  board, and narrow components.
- If real failures remain after retry, create a follow-up ticket for canonical
  edge IDs and directed half-edge stitching.

## Resolution

Dissolved rather than completed. The sampled renderer was replaced by the
analytic construction of A15-A17, which emits one closed, simple loop per stone
by direct construction. There is no marching-squares stitching, no sampling
depth, and therefore no open chain or bad vertex that could arise: both counters
are structurally zero and the retry path is gone.

The acceptance criteria are void with the mechanism that produced them. What
they were protecting — settled shading silently vanishing — is now impossible,
since a loop is emitted per stone with a radial boundary that A16 proves is
single-valued and at least `r`.

Degenerate contacts are covered instead by engine fixtures for exact tangency,
an isolated stone settling exactly its radius-`r` disk, and the star-shape and
radius-floor invariants. Exact hexagonal and square packings at spacing `2r`,
which make every contact tangent, are covered in
`experiments/settled-contour/`.
