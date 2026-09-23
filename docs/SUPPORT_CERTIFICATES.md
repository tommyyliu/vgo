# Stable support certificates

This is an opt-in experiment under `iteration-lab`, not a replacement for the
production settlement algorithm. It refines the earlier `(stone, territory
point, legal placement)` cache described in [the iteration lab](BOARD_ITERATION_LAB.md).
The [search integration](SUPPORT_SEARCH_INTEGRATION.md) now makes this available
to MCTS/self-play through an opt-in bounded worker-local cache.

## The stronger certificate

For our rules, a surviving stone `s` and a legal placement center `p` are enough:

```text
|s - p|² < 8r²  ⇒  s's Voronoi cell is contestable.
```

Let `m = (s+p)/2` and `d = |s-p|`. Every other stone `t` is at least `2r`
from both `s` (stone separation) and `p` (placement legality). The parallelogram
identity gives

```text
|t-m|² = (|t-s|² + |t-p|²)/2 - d²/4
       ≥ 4r² - d²/4
       > d²/4 = |s-m|².
```

Thus `s` uniquely owns `m`. A sufficiently small step from `m` toward `p`
stays in that cell and is strictly closer to `p` than to `s`. That is a
contestable territory point. Both centers lie in the convex board inset, so
this construction stays on the board. The cutoff is **strict**; equality is
not certified by this argument.

The consequence is stronger than choosing an inner witness: we do not need to
store a territory witness at all. The certificate survives any sequence of
insertions and removals while `s` survives and `p` remains legal. Cell vertices
may move freely. A group needs only one supported member; retaining support
edges per stone also handles group splits without pretending every child
inherits its parent's proof.

The implementation respects the existing tolerant legality predicate. It uses
a smaller separation bound, `a = 2r - 4*COORDINATE_EPSILON`, and requires an
outward-rounded squared-distance upper bound below a downward-rounded `2a²`.
The fast path is disabled for invalid setups, the website ruleset, and radii
comparable to the coordinate tolerance. This is a conservative sufficient
test, not a replacement for analytic settlement or an independent validation
of the engine's floating-point policy.

## Available and dormant points

The prototype constructs circle-circle and circle-inset-boundary intersections,
inset corners, and four cardinal points on each radius-`2r` exclusion circle.
Cardinal points cover circles with no intersections. Coordinates remain fixed;
we retain points even when stones currently block them.

Each point has a blocker count and each stone has a reverse list of points it
blocks. A point is available exactly when its count is zero. Removing a stone
decrements those counts, so captures can reactivate dormant proofs. A legal
point need not remain a vertex of the current legal-space boundary to be useful.

We tested retaining all support edges against up to four per stone, selected
greedily by availability and spatial separation. Two points at least `4r`
apart cannot both be blocked by one radius-`2r` open exclusion disk. However,
the four-point policy does not guarantee such separation, and losing all four
only triggers fallback; it never authorizes capture.

The number of candidate features is linear under the equal-radius packing
constraint: intersecting circle centers are within `4r`, with bounded neighbor
count. **This prototype's construction is still quadratic:** it scans all
stone pairs and all stone-point pairs. It also scans all points for an inserted
stone. A spatial index is a future optimization, not a measured property here.

No new candidate features are generated for the inserted stone. Existing
points can support it, but missed opportunities go through the unchanged
analytic fallback. The cache is prepared once per parent and reused for sibling
moves; it is not yet a continuously updated board with make/unmake.

## Negative facts have the opposite lifetime

An unchallengeable cell remains unchallengeable under insertion: its territory
shrinks and legal placement space shrinks. Removal can invalidate this fact by
expanding either set. Positive support certificates, by contrast, survive
removals as long as their supporting stone survives.

The experimental negative cache is deliberately more conservative than that
theorem. It reuses a parent cell's negative result only if its polygon values
are unchanged and **every parent stone survives**. Any parent-stone removal
invalidates the entire negative cache. This avoids relying on containment
inferences involving newly rounded polygon vertices.

Negative preparation currently builds parent analysis and explicitly checks
every cell. Its cost must be included in timings; it is not free knowledge.

## Reproduce and interpret the experiment

```sh
cargo test -p vgo-core -p vgo-search --lib --features vgo-core/iteration-lab
cargo run --release -p vgo-core --features iteration-lab --example support_lab -- 5
```

The benchmark reuses the iteration lab's deterministic played positions and
separate checkerboard stress setups. Each strategy first matches 920 complete
results across both rulesets, including refusals. Additional unit tests cover
tangencies, capture-driven point reactivation and negative invalidation, and
independent trajectories at three radii and four seeds (over 500 candidate
transitions per support strategy). These are differential tests, not a proof
that every floating-point degeneracy is covered.

Timing rotates five variants, discards the first round, and takes the median
of seven measured batches of five repetitions. Parent preparation is averaged
over five runs and amortized over the fixture's distinct sibling candidates,
not over timing repetitions. Complete move results, allocations, capture
stages, and final geometry remain included. Aggregate costs are weighted by
candidate count. This is neither neural-search throughput nor advancing-playout
performance.

### Results, 2026-09-11

Measured on an AMD Ryzen 9 9950X with Rust 1.97.1, release profile, using the
command above. Raw measurements are in
[`support-lab-2026-09-11.csv`](../diagnostics/support-lab-2026-09-11.csv).

| Strategy | Played positions, µs/move | Played speedup | Lattice stress speedup |
| --- | ---: | ---: | ---: |
| Baseline | 131.385 | 1.00× | 1.00× |
| Earlier ownership-witness cache | 101.758 | 1.29× | 0.96× |
| All support points | 90.645 | **1.45×** | 1.02× |
| Four support points per stone | 91.367 | 1.44× | 1.02× |
| All supports + eager negative cache | 109.124 | 1.20× | **4.27×** |

All figures include amortized preparation. The played aggregate contains 364
sibling moves; the lattice aggregate contains 42. The 256-stone lattice alone
improves from 15.577 ms to 3.566 ms per move with negatives: **4.37×** including
preparation. Its cached transition itself takes 1.559 ms, but preparing that
parent costs 32.117 ms, so quoting only the approximately 10× transition gain
would hide an important cost. These lattice diagrams are stress setups, not
evidence of the same gain in ordinary gameplay.

All support points certify 2,299 of 2,447 played-position resolution-stage group
checks (94%). The remaining checks use analytic fallback. Four-point pruning
has essentially the same played coverage and substantially fewer support
edges, but no clear speed advantage. On the 140-stone parent it reduces edges
from 2,913 to 560 without reducing the point/blocker graph. Small positions can
lose performance because preparation and bookkeeping exceed the saved search.

The eager negative variant improves the difficult stress workload but is slower
than supports alone on the played aggregate. This motivates testing lazy
negative collection next; its performance is not measured here. Production
defaults remain unchanged.

The CSV's `certified` and `groups` count group checks across resolution stages,
not unique groups. `reactivated` similarly counts available formerly blocked
points across stages. `eligible_negative_cells` counts reusable negative facts,
which may not all be consulted when a positive certificate already proves
their group alive.

## Engineering direction

1. Keep support edges attached to stable stone IDs and availability attached to
   fixed point IDs. Add local spatial queries for new blockers and support
   edges; use reverse blocker lists for removals.
2. Start with all support edges. Four-point pruning reduces edge storage but
   still pays candidate/blocker construction costs and may cause extra fallback.
   Revisit bounded representative sets only if memory or traversal profiles
   justify them.
3. Collect negative cell facts opportunistically during required fallback
   searches. Do not eagerly scan every parent cell on ordinary positions.
   Invalidate conservatively on removal before attempting finer dependencies.
4. Introduce reversible updates and incremental geometry as separately tested
   stages. Geometry and owned output construction still impose a floor on this
   prototype; certificate reuse alone does not remove it.

There is no constant-time or optimality claim. Large captures, topology changes,
group splits, and replacement searches can still require substantial work.
The promising result is a simpler proof with fewer invalidation causes, not
evidence that the whole iteration problem is solved.
