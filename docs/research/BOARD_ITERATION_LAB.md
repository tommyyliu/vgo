# Board-state iteration lab

This lab measures complete board transitions and explores algorithms derived
from the Vgo rules. It is independent of neural-network inference and does not
need training data, a GPU, new dependencies, or access to the website engine.

The follow-up [stable support certificate experiment](SUPPORT_CERTIFICATES.md)
removes ownership-witness tracking, retains dormant points across captures, and
compares full support sets, four-point sets, and cached negative cell facts.
The [MCTS memory lab](MCTS_MEMORY_LAB.md) measures retained tree payload, shares
policy logits, and tests compact reversible position history.
The [search integration follow-up](SUPPORT_SEARCH_INTEGRATION.md) connects
support certificates to an opt-in, bounded worker-local MCTS/self-play backend.

## Run it

From the repository root:

```sh
cargo test -p vgo-core -p vgo-search --features vgo-core/iteration-lab
cargo run --release -p vgo-core --features iteration-lab --example iteration_lab -- 10
cargo run --release -p vgo-search --example action_cost
```

The first benchmark accepts a repetition count and an optional fixture-name
substring, for example `-- 50 play-r1` for the radius-1/18 played positions.
Its CSV goes to stdout; correctness counts and aggregate timings go to stderr.
Run benchmarks sequentially, preferably on an otherwise idle machine. Increase
repetitions for tuning small differences. No CPU affinity or governor changes
are made by the lab.

The opt-in `iteration-lab` feature exposes a common strategy interface in
`crates/vgo-core/src/iteration_lab.rs`. Normal clients and search continue to
use the baseline settlement algorithm. Strategies are selected outside the
hot loop and compiled with constant parameters. The shared move transaction
still performs opponent capture, then global self-capture, and returns the
complete `MoveResult`, including final geometry, legal vertices, score, events,
pass state, and outcome. A faster implementation that omits those outputs
would need its own benchmark category.

## Fixtures and verification

The deterministic corpus starts from empty boards at radii 1/10, 1/18, and
1/38, plays up to 160 moves, and saves positions every 20 plies. Candidate
generation mixes seeded random legal centers with analytic legal-set vertices.
Three checkerboard lattice setups at 16, 64, and 256 stones separately stress
dense geometry and many independent groups. Setups are valid diagrams; they
need not be stable positions reachable by ordinary play.

Before timing, every prototype must match the baseline on the complete result
for both rulesets, including invalid-coordinate attempts. The initial corpus
contains 920 checks per prototype: 151 accepted capture moves, 16 self-capture
moves, 10 no-ops, and 158 refusals (including invalid coordinates). Separate
unit tests cover exact and near tangencies, large radii, first/second pass
state, and finished games. Comparison uses exact floating-point value equality
for the existing geometry, not an area tolerance.

This is differential validation against our engine, whose numeric policy is
retained. It does not independently prove agreement with ideal real arithmetic
or with the website. A future representation change must also be checked
against analytic degeneracies and the reference fixtures.

Timing uses seven batches per strategy with cyclically rotated execution
order and reports the median batch cost per move. Validation, geometry,
legal-set construction, and full analysis are also timed independently.
These component timings diagnose where work goes; they are not an additive
decomposition of a move, which may rebuild after captures. Timings include
allocation and destruction of the returned result.

## Review of the existing engine

- Voronoi cells use distance-ordered clipping with an exact geometric stopping
  bound, retained edge provenance, and shared scratch buffers. Every cell still
  constructs and sorts distances to all other stones.
- Legal-set vertices use spatial buckets above 31 stones, pruning candidate
  pairs and blocker checks. The bucket grid already existed for raster queries
  but was not reused by settlement's escape-witness queries.
- Settlement stops as soon as one cell supplies a witness for a group. Its
  ordinary witness search checks analytic candidates for legality before
  testing whether they can challenge the queried vertex.
- The move resolver already has two simultaneous removal stages, not an
  iterative removal loop. Reanalysis after enemy removal is necessary because
  that removal can revive friendly groups. It promotes the last settlement
  into final analysis when self-capture does not change the board.
- `place` has a debug-only validation at entry, but `Settlement::new` still
  validates each analyzed position. Validation has not actually disappeared
  from the release path. Removing it requires an explicit trusted-position
  API; `Position::new` can construct invalid diagrams.
- Search's `Action::try_apply` resolved accepted placements twice: once to
  check refusal and again through `apply`. This has been fixed to return the
  first result. Passes and rejected self-captures retain their semantics.

## Experiments and initial results

Measurements were taken on 2026-09-11, AMD Ryzen 9 9950X, Rust 1.97.1,
workspace release profile (thin LTO, one codegen unit), with the changes in
this lab on top of commit `4dd76e605737a8d26151690d78b31b3e2c17bfe2`.
The host was not isolated; these are local measurements, not confidence bounds.

The search-wrapper benchmark compares the previous implementation reproduced
in the example with the current wrapper in the same process. At 40 stones,
15 accepted candidates, it measured **269.7 to 136.9 microseconds per action,
1.97x faster**. This is the action-resolution path, not total search or training
throughput. See `diagnostics/action-cost-2026-09-11.csv`.

Four full-transition prototypes accompany the baseline:

| Strategy | Change | Observed tradeoff in exploratory runs |
| --- | --- | --- |
| WitnessFirst | Test existing legal vertices for a survival certificate before the full analytic search | Helps larger played positions; extra work can hurt lattices and some small boards. |
| Indexed | Keep the legal-set bucket index for every clearance query in settlement | Nearly halves the 256-stone lattice move cost, but can be slower around 40–60 stones. |
| QueryFirst | Test whether a feature candidate challenges the vertex before checking its legality, using the shared index | Saves irrelevant clearance work on large boards; robust distance tests cost more than early clearance rejection on many small boards. |
| PreparedPosition | Certify one witness per parent group, then reuse surviving certificates across sibling moves | Helps some expensive played positions substantially; setup cost and already-settled groups can erase the benefit. |

`PreparedPosition` is a first implementation of the certificate idea below.
It retains an immutable parent, certifies that cached points really belong to
their owners, and checks only the inserted stone for new blocking or ownership
changes. Each capture stage maps certificates through surviving owner identities
into the current groups. Missing or invalidated certificates fall back to the
baseline analytic search. The prototype still rebuilds Voronoi geometry and
legal vertices so it can return exactly the same full result.

The CSV reports cache construction, certificate count, cached move time, and
amortized cost including one construction per fixture's candidate batch. Cache
construction currently recomputes a full parent analysis and searches for its
witnesses; a future integration could retain those witnesses from the analysis
already performed for the parent. The benchmark charges the current cost in
full and does not assume that integration has happened.

The complete measured table is in `diagnostics/iteration-lab-2026-09-11.csv`.
In the final ten-repetition run, the 364 candidates from played positions
averaged 132.8 microseconds for the baseline and 102.4 for the certificate
prototype including amortized preparation: **1.30x** across that synthetic
played corpus. At 51 stones one saved position improved from 244.6 to 153.7
microseconds (1.59x); at 140 stones, 920.5 to 620.1 (1.48x). Several small
positions regressed because the cache was not used enough to repay its setup.

Do not interpret its combined average as an expected gameplay speedup: the
256-stone lattice dominates total time. Compare played positions and stress
positions separately. None of these experimental algorithms is promoted to
the production path based on this small corpus.

The 256-stone lattice measured 15.57 milliseconds per baseline move and 8.06
with the shared index (1.93x). The certificate cache regressed to 16.25
milliseconds including setup on this fixture, illustrating why its played-game
gain should not be generalized to every valid diagram.

## Design from the base rules: an incremental certificate engine

The most promising larger change is to avoid reconstructing facts that a move
does not invalidate. The rules ask an existential question for survival:

```text
group alive iff some x in its region and p in L satisfy |x-p| < d_S(x)
```

An alive group therefore needs only one certificate `(s, x, p)`: stone `s`
owns `x`, `p` is legal, and `p` is strictly closer to `x` than `s`. The point
`x` need not remain a polygon vertex. Vertex enumeration is a way to discover
a certificate, not a requirement on a certificate retained from a prior move.

For an insertion at `q`, a certificate survives if `p` remains legal and `q`
does not take ownership of `x`. Those are two distance comparisons against the
new stone; all other constraints were already satisfied. Retaining a valid
certificate proves survival without reconstructing the legal set or searching
every cell in the group. Losing a certificate does **not** prove settlement:
the group must search for a replacement before it can be captured.

The proposed state has four cooperating structures:

1. **Stable stone IDs, a spatial grid, and an undo log.** Insertions and removals
   update only affected buckets. Undo restores coordinates, topology, cached
   certificates, scores, and pass metadata; it does not approximately invert
   floating-point constructions. Search workers own separate mutable states.
2. **Incremental Voronoi/Delaunay topology.** Update the affected insertion or
   deletion cavity, its clipped cells, positive-length adjacency, and area
   deltas. This avoids an all-stone distance sort for every cell. Robust
   orientation/incircle decisions and explicit cocircular/point-contact cases
   are mandatory. Equal coordinates and setup validation remain separate.
3. **Survival certificates with reverse spatial indexes.** Index witness centers
   `p` so the new exclusion disk finds certificates it invalidates, including
   those belonging to geometrically remote groups. Index witness points `x`
   or associate them with changed cells to find ownership changes. After group
   splits, reassign certificates by their owning stone; each new component
   needs its own certificate. Merged groups may reuse any valid constituent
   certificate. A plain insertion-only union-find is insufficient for splits.
4. **On-demand exact replacement search.** Use the cell-vertex theorem and the
   line/circle candidate families only for groups without a surviving
   certificate. Maintain legal-boundary features incrementally, including
   isolated legal points, or build local query arrangements. Every negative
   answer must cover all feature families; a sampling miss is never death.

Resolve opponent captures in one batch, then update topology and find friendly
self-captures. Removal enlarges legal space and surviving cells, so a surviving
owner's valid certificate remains valid. Track removed identities to detect
no-op placements exactly. If an experiment begins from an unstable setup,
perform the full global checks required by the rules.

This design targets work proportional to changed geometry and invalidated
certificates on ordinary moves. It has no general constant-time guarantee:
Voronoi cavities, group splits, captures, and certificate invalidations can be
large. A legal-space grid is especially attractive because exclusion disks
have equal radii and stone centers have a packing constraint, but that does
not make group survival purely local.

The cached-certificate prototype atop the existing geometry is implemented in
this lab. The next stages are incremental cells and reversible make/unmake state.
At each stage, measure ordinary trajectories, sibling expansions from the same
parent, and deep play/undo separately. The current benchmark establishes the
rebuild baseline and certificate reuse; it does not yet measure incremental
geometry or reversible state. Keep a rebuild
fallback for degeneracies until the independent predicates and fixtures cover
them. The complete incremental engine remains a design, beyond the measured
certificate prototype.

## What would justify promotion

Expand to recorded self-play positions, multiple seeds, small/large radii,
near-full boards, forced legal points, group splits, remote self-captures, and
long make/unmake traces. Measure both same-parent branching and advancing
playouts. Preserve errors, state, groups, captures, and pass behavior exactly;
investigate every geometric disagreement rather than accepting matching total
area as sufficient. Only then choose workload-based thresholds or replace the
default transition algorithm. End-to-end search measurements must include
candidate generation and inference to establish the realized throughput gain.
