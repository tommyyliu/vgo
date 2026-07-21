# 005: Contour Topology Diagnostics

- Status: In progress
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

## Progress

Diagnostics now include the VGO-SGF position, radius, depth, segment count,
open-chain count, bad-vertex count, and vertex-degree histogram. A failure is
retried once at greater depth before being logged. The fully settled board has
a permanent closure test; the remaining singular and narrow fixtures are still
open.
