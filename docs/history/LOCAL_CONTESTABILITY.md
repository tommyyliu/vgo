# Local contestability: exact facts before witness graphs

Update, 2026-09-22: [a sharper square-board proof](LOCAL_CONTESTABILITY_SHARP_BOUND.md)
establishes the universal cutoff `(2 + 4sqrt(2))r`, approximately `7.656854r`,
including corners and simultaneous removal of distant stones. The older bounds
below remain sufficient; the new note records the improvement separately.

Exploration, 2026-09-12. The arguments below concern ideal Euclidean geometry
under our Vgo rules on a square board. They are not a proved floating-point
cutoff for the production engine. No production transition behavior changes.

## Individual status is monotone, not invariant

For a surviving stone s, write V(s) for its closed Voronoi cell clipped to the
board, and L for the legal-center set. Its cell is contestable exactly when
there exist v in V(s), p in L with |v-p| < |v-s|.

Insertion shrinks both V(s) and L. Therefore an existing non-contestable cell
cannot become contestable through insertion alone. Removal expands both sets:
a surviving contestable cell cannot become non-contestable through removal.
Neither statement depends on colors or group membership.

An individual cell CAN lose contestability without any capture: fill a hole in
the interior of a same-color cluster. A neighboring stone loses its local
liberty while the group survives through liberties at its outside edge. The
new differential test constructs this explicitly with 49 stones after filling.

## Finite witness reach

Let r be the stone radius, I=[r,1-r]^2 the legal-center inset, and
R=(2+sqrt(2))r. Assume valid centers and 0<r<=1/2.

If V(s) contains a point at distance at least R, convexity gives a point v in
V(s) at distance exactly R. Every stone is at least R from v. Project v onto I,
obtaining p. The displacement is at most sqrt(2)r, so every stone is at least
2r from p. Thus p is legal and |v-p| < |v-s|: it contests the cell. In fact
|p-s| <= R, since projection onto the convex inset cannot increase distance to
s, which is itself in the inset. The earlier 4.83r estimate is also valid but
unnecessarily loose for this branch.

Otherwise every point of V(s) is nearer than R. Any contesting p satisfies
|p-s| <= |p-v|+|v-s| < 2R. Thus every contestable cell admits a placement witness
within 2R=(4+2sqrt(2))r, approximately 6.83r, of its owner. A territory witness
can always be chosen within R. These constants need not be sharp.

## An exact locality theorem, not only a sufficient certificate

Set D=2R+2r=(6+2sqrt(2))r, approximately 8.83r. Keep s and all stones within D
of it, removing the rest, and retain the original board and radius. Then s's
individual contestability is unchanged in ideal arithmetic.

Proof: removal cannot destroy contestability. For the converse, let W be s's
cell in the reduced position.

* If W reaches radius R, choose v at R and its inset projection p as above.
  Omitted stones cannot beat s at v: their distance from v is greater than
  D-R > R. They cannot block p either: |p-s| <= R, so they are farther than
  D-R > 2r from p. Hence the constructed witness is valid on the original board.
* If W is wholly inside radius R, omitted stones cannot cut W, since they are
  farther than 2R from s. The cell is therefore unchanged. Any reduced-position
  contesting placement lies within 2R of s and cannot be blocked by an omitted
  stone farther than D=2R+2r away. It is also an original-position witness.

This means ordinary cell contestability can be evaluated on the reduced
position; an explicit circular territory clip is not necessary for this
theorem. The reduced Voronoi cell may have distant artificial extensions, but
the first branch supplies a nearby witness valid on the original board.

Consequence: inserting or removing a stone outside D of a surviving owner
cannot change that owner's status, even if its distant cell geometry changes.
Packing bounds the number of owners within D independently of global stone
count. This is not a claim that every move, geometry update, or capture is O(1).

## Infinite-plane formulation: a local packing problem

Translate the owner to the origin and scale lengths by r, so stone centers are
at least 2 apart. There are then no dimensional parameters. Let V be its
Voronoi cell. A proposed center p contests a territory point v exactly when

    |p-v|^2 < |v|^2  <=>  2 v.p > |p|^2.

For bounded polygon V the left side is linear in v, so it suffices to check
the polygon vertices. The region of centers capable of contesting the cell is

    F = union over vertices v of open_disk(v, |v|).

The exact question is whether F contains a point outside every open radius-2
exclusion disk centered on a stone. Tangent placement is legal; contesting a
point requires strict improvement. A placement center does NOT have to lie in
V itself. It may lie just outside and still take a small piece of the cell.

The existing vertex-based `escape_witness` solver already embodies this
inequality. The new opportunity is bounding and reusing its local inputs, not
claiming the vertex reduction is a newly invented solver.

In angular form, write p=t*u, |u|=1, and h_V(u)=max(v.u) over v in V. The center
can contest iff 0<t<2h_V(u). Its own stone requires t>=2. For another center b,
the forbidden radial interval is obtained from

    t^2 - 2t(b.u) + |b|^2 - 4 < 0.

When the quadratic has real roots, its interior interval is blocked; intersect
it with t>=0. Thus along each direction the question is whether any of
[2, 2h_V(u)) remains uncovered by those intervals. This is exact as a continuous
formulation; sampling a finite set of directions would not be an exact solver.

### Sharper planar locality: 6r

On the plane, if the cell reaches distance 2r from s, a point v there is itself
a legal center: every other stone is at least as far from v as s is. Playing at
v contests v. This includes unbounded cells.

Otherwise V is entirely within distance 2r of s. Any contesting center must
then be within 4r of s by the triangle inequality. Stones beyond 6r cannot
block such a center. They also cannot cut this small cell, being beyond 4r.

As in the square-board proof, this establishes invariance when ALL stones
beyond 6r are removed: if the reduced cell grows to 2r, its point at 2r is
still owned and legal in the original arrangement; otherwise its entire cell
and any qualifying placement are protected from the omitted stones. Thus
individual contestability on the plane is determined within 6r, not 8.83r.

For a bounded cell with circumradius rho<2r, only legal centers within 2rho
can contest it; blockers beyond 2rho+2r are irrelevant. This is an adaptive
reach AFTER establishing that cell, not a circular recipe for computing rho.

### Reintroducing the board

Replace V by the board-clipped cell and intersect F with the legal-center inset.
The disk-coverage formulation remains unchanged otherwise. The planar 6r
theorem cannot simply be used next to an edge, where its proposed centers may
fall outside the inset. The prior 8.83r square-board theorem remains the
conservative general bound. A sufficient condition for ignoring the edges
altogether is that the radius-4r disk around the owner lies in the inset (owner
at least 5r from every physical board edge). This is sufficient, not necessary.

The plane is therefore a useful mathematical core, but board clipping and
placement clipping are distinct constraints. Neither should be approximated by
adding ordinary virtual stones along an edge without a separate equivalence
proof. These planar deductions are not new measured performance results.

## One test for interiors, walls, and corners

The operational formulation need not construct the union of contesting disks.
For each vertex v of the board-clipped cell, ask whether

    distance(v, L) < distance(v, s),

where L is the legal-center inset minus all OPEN radius-2r exclusion disks.
One successful vertex suffices. For an unbounded plane cell, the radius-2r
escape argument settles the question first. This is the existing analytic
predicate with a bounded local input, not a different capture rule.

The closest legal center, when one exists, is either the query itself, a nearest
point on an exposed circular arc, a projection onto an exposed inset edge, or
a boundary junction (circle/circle, circle/edge, or inset corner). Degenerate
cases need care: a query at a circle center needs an available arc representative
or endpoint. The current solver already enumerates these analytic families.
An exact solver cannot replace them with angular or pixel samples: legal
centers at tangencies can be isolated points, not open gaps of positive area.

The territory board and center inset must remain separate constraints. Treating
the latter as four linear inequalities is simpler and exact; a row of ordinary
ghost stones would not itself implement a hard wall. No additional proof graph
is necessary just to accommodate edges.

### A unified proof with adaptive constants

Choose a radius R such that every point of the territory board within R of s
projects into the center inset by at most R-2r. The existing locality proof then
works with D=2R+2r: if the reduced cell reaches R, the projected point is legal;
otherwise every potential contesting center is within 2R.

The following choices give exact-arithmetic sufficient conditions. Distances
in the conditions are to the four PHYSICAL board edges, not the inset edges.

| Owner location | Sufficient condition | R | Keep stones within D |
| --- | --- | ---: | ---: |
| Interior | All four edge distances >=3r | 2r | 6r |
| Single-edge region | At least three edge distances >=4r | 3r | 8r |
| General / corner / narrow board | No extra condition | (2+sqrt(2))r | (6+2sqrt(2))r |

Use the interior case first. In that case the radius-2r neighborhood is inside
the inset, so no projection is necessary. For the single-edge case, the
radius-3r neighborhood can cross at most one inset edge; projecting a territory
point moves it at most r. In the general case, the maximum displacement is
sqrt(2)r. At equality, retain all points on the cutoff; production needs a
numerically conservative treatment, not these ideal inequalities verbatim.

IMPORTANT: the 3r interior condition permits the smaller STONE neighborhood
while keeping the board and inset in the predicate. It does not mean all board
constraints may be discarded. The earlier 5r sufficient condition concerns
ignoring board edges altogether, which is a different claim.

The differential test now checks both the general and adaptive neighborhoods:
432 comparisons on the same 216 owner queries, all matching the full-board
solver (314 positive, 118 negative). Adaptive classes contain 157 interior,
34 single-edge, and 25 general queries. Every comparison removes distant
stones. This checks reduced-input agreement, not timing or a floating-point
proof; it reuses the same oracle and fixture families described below.

## A simpler candidate architecture

Maintain stable stone identities and a three-state fact per stone:
`contestable`, `non-contestable`, or `unknown`. Unknown is not dead.

| Change | Known positive | Known negative |
| --- | --- | --- |
| Insert within D | Invalidate to unknown | Retain |
| Remove within D, owner survives | Retain | Invalidate to unknown |
| Insert/remove outside D | Retain | Retain |

New stones start unknown; removed stones lose their records. Passes preserve
the facts. Apply invalidation at every provisional capture stage and after final
self-removal; do not change the game's simultaneous-removal or stage ordering.

For each current group, any known-positive member proves survival. Otherwise
resolve unknown members only until one is positive. If every member is known
negative, the group is settled. Cache negatives and positives returned by those
required queries, rather than eagerly evaluating every member of every group.
This also avoids an expensive initial all-cell analysis solely to populate a
cache: existing group checks stop on their first witness, so many member facts
will legitimately remain unknown.

Group merges/splits do not invalidate stone facts by themselves, but group
membership must be current before aggregating them. Splits can expose large
components whose members are all unknown, so query work can still be nonlocal
in aggregate. Updating connectivity is a separate problem; Voronoi adjacency
is not restricted to the same radius merely because contestability is.

This fact cache needs only a small state per stone plus stable-ID and reversible
invalidation bookkeeping. It does not require retaining every potential legal
point, blocker count, support edge, or parent proof graph. Optional stable
support witnesses can still give cheap positive answers for unknown cells.
It does not reduce the dense-policy memory that dominates current MCTS trees.

## Evidence and limits

`iteration_lab/locality.rs` contains two new differential tests. The neighborhood
test checks 24 owners on each of nine fixtures (three radii, three occupancy
patterns; up to 576 stones). The initial general-bound run had 216 comparisons
agreeing with the full-board analytic cell predicate: 157 positive and 59
negative. The expanded general/adaptive results are reported above. Every query removes at
least one distant stone. A separate test verifies loss of individual liberty
without capture while the owning group remains alive.

```sh
cargo test -p vgo-core --features iteration-lab --lib iteration_lab::locality -- --nocapture
```

The tests use existing Voronoi construction and analytic escape queries, not an
independent exact-arithmetic oracle. They do not establish speed, full playout
parity, or numerical soundness near all tangencies and degeneracies. The test
cutoff includes a small outward slack; that is not a production error bound.

Before enabling this design: independently review the proof; establish a
conservative numerical contract (with fallback for uncertainty); exercise
irregular played positions, local insertions, removals, group splits, and undo;
then benchmark lazy three-state reuse against a duplicate-transition-free
baseline. Initially retain the existing geometry/group construction to isolate
the value of the fact cache before attempting incremental connectivity.
