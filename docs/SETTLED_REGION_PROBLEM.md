# The settled-region problem, stated in isolation

Self-contained statement for anyone optimising this without the surrounding
codebase. Current implementation: `crates/vgo-core/src/settled.rs` (per-stone
solve) and `crates/vgo-raster/src/lib.rs::settled_mask` (rasterisation).

## Setting

The board is the unit square `[0,1]²`. Stones are open discs of a fixed radius
`r` (production: `r = 0.0557142857142857`, about 1/18). A position is a list of
`n` stone centres, each Black or White. `n` is 28 at the median and 52 at the
maximum; the placement rule keeps centres at least `2r` apart.

**Legal set `L`** — where a new stone centre may go:

    L = { p in [0,1]²  :  r <= p.x <= 1-r,  r <= p.y <= 1-r,
                          ||p - s|| >= 2r  for every stone s }

**Settled set `S`** — board a future placement can no longer reach:

    S = { x  :  exists a stone s with  ||x - s|| <= dist(x, L) }

where `dist(x, L) = inf { ||x - p|| : p in L }`, and `dist(x, ∅) = +inf` so an
empty legal set makes everything settled.

Read plainly: a point is settled when some existing stone is already at least as
close to it as any legal future stone could ever be.

## What is actually needed

A boolean mask over a 128 x 128 grid, sampled at pixel centres
`x = (col + 0.5)/128`, `y = (row + 0.5)/128`. One mask per position. That is the
only output — the region's geometry is not needed, only its indicator on that
grid.

**Exactness.** This feeds a neural network as one input channel, so the mask
must match the current implementation on essentially every pixel; pixels within
floating-point epsilon of the boundary may differ. It must also stay consistent
with `reference/src/geometry/settled-contour.js`, which draws the same region as
a client-side overlay.

## Why it is worth optimising

Measured 2026-08-16 (`cargo bench -p vgo-raster`), time to build one 128 x 128
input tensor:

    stones | total raster | settled       | everything else
        14 |    0.164 ms  | 0.126 (76%)   | 0.039
        28 |    1.133 ms  | 1.040 (92%)   | 0.093
        52 |    4.616 ms  | 4.437 (96%)   | 0.178

In production this is **33% of all CPU** during self-play generation (13,270
core-seconds against 1,977 for every neural-network evaluation, over a 1258 s
shard on 32 cores). Cost grows roughly as `O(n²)`.

## Two known formulations

**Per-pixel (direct).** For each of the 16,384 pixels, evaluate `dist(x, L)` and
compare against the nearest stone. `dist(x, L)` is a minimum over a candidate
set that provably contains the nearest legal point:

  1. `x` itself, if `x ∈ L`
  2. for each stone `s`: `s + 2r·û` where `û = (x-s)/||x-s||` (the four axis
     directions when `x = s`)
  3. the four inset projections `(r, x.y)`, `(1-r, x.y)`, `(x.x, r)`,
     `(x.x, 1-r)`
  4. the legal-set vertices `V` (pairwise stone/stone, stone/edge and
     edge/edge intersections at separation `2r`, filtered to those in `L`)

Each candidate must be tested for membership in `L`, which is `O(n)`. So one
pixel costs `O((n + |V|) · n)`. **Measured at 42x the cost of the entire rest of
the raster** — this is the formulation the current code exists to avoid.

**Per-stone (current).** Three structural facts collapse it:

  - **A15** `S = ⋃_s R_s` with `R_s = { x : ||x-s|| <= dist(x,L) }`. A stone
    that is not the nearest only makes the test harder, so the Voronoi
    partition drops out and the stones can be handled independently.
  - **A16** Each `R_s` is **star-shaped about `s`**, so its boundary is a
    single-valued radial function `T(u)` of direction `u`. A 2-D level set
    becomes one scalar equation per direction.
  - **A17** `T(u)` has a **closed form**: a minimum over the same candidate
    families, and `t_c >= ||c - s||/2` by the triangle inequality, so candidates
    sorted by distance from `s` admit an exact early stop.

The implementation builds per-stone orderings (all stones by distance from `s`,
all vertices by distance from `s`), solves `T(u)` per direction, extracts a
polygon contour at 1/128 resolution, and scanline-fills it. Nothing samples a
field or iterates to a root.

The `O(n²)` comes from the per-stone setup: each of `n` stones sorts all `n`
stones and all `|V|` vertices by distance from its own origin.

## What a better algorithm would have to beat

Wall time to produce the 128 x 128 mask for a 28-stone position: **1.04 ms**,
single-threaded, on a Zen-class core. Anything that holds exactness and is
meaningfully below that is a win; a 5x would return ~25% of the machine.

**Tried, and it works** (2026-08-16, `crates/vgo-raster/src/edt.rs`,
`cargo run --release -p vgo-raster --example settled_edt`):

    stones | shipping | bounded-distance | speedup | exact tests | wrong (vs shipping)
        14 |  0.123ms |         0.225ms  |   0.5x  |    107      |  0  (1)
        28 |  1.004ms |         0.356ms  |   2.8x  |    134      |  0  (0)
        52 |  4.186ms |         0.579ms  |   7.2x  |    135      |  0  (2)

`settled(x) <=> D_S(x) <= D_L(x)` makes both sides distance transforms. Sample
the legal set by *stamping* each stone's 2r exclusion disc (O(n·r²·pixels), not
O(pixels·n)), take the exact separable Euclidean transform of it, and compare
against the distance to the nearest stone taken from the continuous
coordinates. Sampling makes `D_L` an overestimate, so it decides a pixel only
when the answer is outside a slack band; 0.8% of pixels fall inside it and get
the exact continuous test. Cost is O(pixels) and stays nearly flat as stones
accumulate, where the current implementation quadruples.

Two things that bit, both recorded in the code: the slack must be a **full**
cell diagonal, not half (a legal point near the set's boundary can sit in a cell
whose centre is outside it — half cost two wrong pixels at *eight* stones); and
oversampling must be **odd**, or no fine cell is centred on the output cell and
accuracy falls rather than rises.

Still not exact: a legal sliver narrower than a cell has no sampled centre at
all, and nothing in the slack bound covers that. It measured no worse than the
shipping implementation, which walks a contour at 1/128 and is wrong on 1-2
pixels of 16384 itself.

Ideas not yet tried, in no particular order:

- **Incremental reuse.** During tree search, consecutive positions differ by a
  single stone. `R_s` for a stone far from the change is unaffected, and `L`
  changes only near the new stone. Nothing currently exploits this — every
  position is computed from scratch.
- **Kill the per-stone sorts.** The `n` sorts of `n` stones are the `O(n² log n)`
  term. A single shared spatial structure (k-d tree, grid) might serve every
  stone's early-stop test without per-stone reordering.
- **Prune stones that cannot contribute.** `R_s` is contained in a disc around
  `s` whose radius is bounded by `dist(s, L)`; stones whose discs cannot reach
  any pixel of interest could be skipped entirely.
- **Coarse-to-fine.** Decide whole 8x8 pixel blocks by a conservative bound and
  only refine blocks the boundary crosses.
- **Per-pixel, on a GPU.** Done, in `crates/vgo-raster-cuda`: 13.9x per position
  at batch 32 in f32. But it does not pay off in self-play on this machine —
  rasterization is parallel across 64 actor threads and the GPU only wins where
  the CPU is weak (crossover around 9 cores). The distance-transform work above
  is the better lever on this hardware, and would port to the GPU too.

## Interfaces

    // crates/vgo-core/src/settled.rs
    SettledRegion::new(position, stone_index, vertices) -> SettledRegion
    region.contour_within_into(tolerance, &mut Vec<Point>)

    // crates/vgo-core/src/legal_set.rs
    legal_set::vertices(position) -> Vec<Point>          // V, computed once
    legal_set::distance(position, point, Some(&V)) -> f64 // dist(x, L)
    legal_set::contains(position, x, y) -> bool           // membership in L

    // crates/vgo-raster/src/lib.rs
    settled_mask(position, config) -> Vec<bool>           // the 128x128 output

`cargo bench -p vgo-raster` reports the split above and is the measurement any
change should move.
