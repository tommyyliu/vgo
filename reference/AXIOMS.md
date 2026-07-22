# Derived Properties and Implementation Axioms

Despite the filename, these are not additional game rules. They are theorems
derived from `RULES.md` and may be used as implementation invariants. Each claim
assumes a valid position, Euclidean distance, and exact real arithmetic unless
it explicitly says otherwise.

## A1. A stone's disk lies inside its cell

**Claim.** For every stone `s`, the closed radius-`r` disk centered at `s` lies
inside `V_s(S)` and inside the board.

**Proof.** The center rule places `s` at least `r` from every board edge, so its
disk is inside the board. For any point `x` with `||x-s|| <= r` and any other
stone `t`, the separation rule and triangle inequality give

```text
||x-t|| >= ||s-t|| - ||x-s|| >= 2r - r = r >= ||x-s||.
```

Thus `s` is a nearest stone to every point of its disk. This also proves that
every cell has nonempty interior.

## A2. Voronoi cells are compact convex polygons

**Claim.** Each `V_s(S)` is a compact convex polygon.

**Proof.** Squaring `||x-s|| <= ||x-t||` cancels `||x||^2` and produces a linear
half-plane constraint. A cell is the intersection of the compact square `D`
with one such closed half-plane for every other stone. A finite intersection of
closed convex half-planes is a compact convex polygon. A1 makes it nonempty and
full-dimensional.

## A3. A cell change has a vertex witness

**Claim.** A legal placement center `p` takes positive area from `V_s(S)` if and
only if it is strictly closer than `s` to at least one vertex of `V_s(S)`.

**Proof.** Define

```text
Delta(x) = ||x-s||^2 - ||x-p||^2
         = 2 x.(p-s) + ||s||^2 - ||p||^2.
```

`Delta` is affine, and `p` is strictly closer exactly where `Delta > 0`. An
affine function reaches its maximum over a compact convex polygon at a vertex.
Therefore a positive value anywhere implies a positive value at a vertex.
Conversely, strict positivity at a vertex persists on a sufficiently small
positive-area portion of the full-dimensional cell.

## A4. The meaning and lower bound of rho

For a cell vertex `v`, define

```text
rho(s,v) = ||v-s||.
```

**Claim.** A center `p` witnesses a change at `v` exactly when
`||p-v|| < rho(s,v)`, and every cell vertex has `rho(s,v) >= r`.

**Proof.** The first statement is A3's strict distance comparison at `v`. By A1,
the open radius-`r` disk around `s` is in the interior of the cell. A cell
vertex is on the cell boundary, so it cannot lie inside that open disk.

`rho` is therefore a geometric threshold, not a floating-point tolerance.

## A5. Finite capture criterion

Let `delta(v) = dist(v,L(S))`, with `delta(v) = Infinity` when `L(S)` is empty.

**Claim.** A group `G` is settled exactly when

```text
delta(v) >= rho(s,v)
```

for every vertex `v` of every cell belonging to a stone `s` in `G`.

**Proof.** By A3 and A4, a legal center changes `V_s(S)` exactly when it lies in
the open disk `B(v,rho(s,v))` for at least one cell vertex. Such a legal center
exists exactly when the minimum distance from `v` to `L(S)` is strictly less
than `rho(s,v)`. Negating that existential statement for every cell in the
group gives the claimed non-strict inequality.

This is the mathematical basis of `VGO.analysis.analyze()` and contains no
spatial sampling.

## A6. Closest legal centers occur on enumerated feature types

**Claim.** For a point `v` not already in `L(S)`, a closest point of `L(S)` is
one of the following:

1. A perpendicular projection onto an inset-board edge.
2. A radial projection onto a radius-`2r` exclusion circle.
3. An inset corner, circle-edge intersection, or circle-circle intersection.

Only candidates that satisfy every legal-placement constraint are retained.

**Proof.** `L(S)` is closed and bounded, so a closest point exists when it is
nonempty. A closest point not at a constraint junction lies in the relative
interior of one smooth boundary feature. First-order optimality makes the
point-to-boundary vector normal to that feature, giving a perpendicular line
projection or radial circle projection. If two or more independent constraints
are active, the point is a boundary intersection from item 3. These cases
exhaust the piecewise-line-and-circle boundary of `L(S)`.

For capture queries, A1 and A4 ensure a cell vertex is not a stone center, so
the required radial directions are defined.

## A7. How a move changes legal-placement space

**Claim.** Adding a legal stone at `q` changes the legal set by

```text
L(S union {q}) = L(S) minus B_open(q,2r).
```

Removing stones can only enlarge the legal set.

**Proof.** The new stone contributes exactly one new requirement,
`||p-q|| >= 2r`; every previous requirement remains. Removing a stone deletes
one requirement and cannot invalidate a previously legal center.

## A8. Settlement is monotone under addition and removal

Define the settled-point set

```text
Z(S) = { x in D : d_S(x) <= dist(x,L(S)) }.
```

**Claim.** Adding a stone can only enlarge `Z`; removing stones can only shrink
`Z`.

**Proof.** After adding a stone, `d_S(x)` can only decrease, while A7 says the
distance to the now-smaller legal set can only increase. Thus every previously
settled point remains settled. Removal reverses both inequalities: nearest-stone
distance can only increase and distance to legal space can only decrease. Hence
every previously unsettled point remains unsettled.

This is point-set monotonicity. It does not imply Voronoi-neighbor locality for
capture.

## A9. Removal cannot newly settle a surviving group

**Claim.** Removing any collection of stones cannot turn a previously
unsettled, fully surviving group into a settled group.

**Proof.** Choose an unsettled witness point `x` in the old region of the group.
If `x` belonged to surviving stone `s`, deleting other sites only removes
half-plane constraints from `V_s`, so `x` remains in `V_s`. By A8, `x` also
remains unsettled. If removal merges the group with another same-color group,
the merged group still contains this witness.

Consequences:

- Opponent captures can be found in one simultaneous batch.
- Rechecking for newly settled surviving opponent groups after removal is not
  necessary for correctness.
- Enemy removal may revive a mover group, so self-capture must be resolved after
  opponent removal rather than before it.

## A10. Territory locality does not imply capture locality

For a group `G`, define its witness-placement set from A5:

```text
W_G(S) = L(S) intersect
         union over s in G and vertices v of V_s of B_open(v,rho(s,v)).
```

**Claim.** If a new stone `q` takes positive area from an old cell `V_s`, then
the new cells of `q` and `s` share a positive-length boundary. Separately, if

```text
q is not in W_G(S),
W_G(S) is nonempty, and
W_G(S) is contained in B_open(q,2r),
```

then placing `q` settles `G` without `q` taking area from `G`.

**Proof.** For the first statement, within the old cell only `s` and `q` compete.
Both retain neighborhoods of their centers by A1, so any positive-area transfer
creates a separating portion of their perpendicular bisector. For the second
statement, `q` being outside `W_G(S)` means it does not challenge any cell of
`G`, by A3-A5. Its placement therefore does not take area from `G`. A7 removes
`B_open(q,2r)` from legal-placement space. The containment hypothesis removes
every member of the previously nonempty witness set, so A5 makes `G` settled.

Therefore only direct territory changes are Voronoi-local. Capture and
self-capture checks must be global unless a different, proved spatial index is
used.

## A11. The move transaction preserves stability

Call a position stable when it contains no settled group.

**Claim.** Starting from a stable position, every committed placement produced
by Rule 5 is stable.

**Proof.** Rule 5 removes every opponent group settled in the provisional
position. Every surviving opponent group was unsettled before that removal and
remains unsettled by A9. Rule 5 then removes every settled mover group. Each
surviving mover group was unsettled before that removal and remains unsettled by
A9. Removing mover stones also cannot settle a surviving opponent group. Thus
neither color has a settled group when the move commits. A pass changes no
geometry and also preserves stability, including when a second pass ends the
game.

This proof is why resolving only the group containing the new stone is
insufficient.

## A12. Self-capture needs only one simultaneous batch

**Claim.** After opponent capture resolution, removing every settled mover group
simultaneously cannot cause another surviving mover group to become settled.

**Proof.** Every mover group not selected for self-capture is unsettled before
the selected groups are removed. It survives that removal, so A9 says it remains
unsettled. Therefore self-capture does not require an order-dependent cascade or
fixed-point computation.

## A13. Score partitions the board

**Claim.** With at least one stone, the sum of all cell areas is the area of the
board, namely `1`.

**Proof.** Every board point has at least one nearest stone, so the closed cells
cover `D`. Distinct cell interiors are disjoint. Cell overlaps occur only on
distance-tie boundaries, which have zero area. Therefore cell areas add to the
area of `D`.

## A14. Passing preserves score

**Claim.** A pass cannot change either player's score, so the scores after two
consecutive passes equal the scores immediately before the first pass.

**Proof.** Passing changes neither the stone set nor the radius. Voronoi cells
and their areas are functions only of those values, so every cell and score is
unchanged by either pass.

## A15. The settled set needs no Voronoi partition

**Claim.** With at least one stone,

```text
Z(S) = union over s in S of R_s(S),
R_s(S) = { x in D : ||x-s|| <= dist(x,L(S)) }.
```

**Proof.** If `x` lies in `R_s` for any stone `s`, then
`dist(x,L(S)) >= ||x-s|| >= d_S(x)`, because `d_S` is the minimum over all
stones; so `x` lies in `Z(S)`. Conversely, if `x` lies in `Z(S)`, choose a
nearest stone `s`. Then `||x-s|| = d_S(x) <= dist(x,L(S))`, so `x` lies in
`R_s`. A non-nearest stone only makes the test harder, never easier, so the
union may be taken over every stone without first deciding cell ownership.

Consequences:

- Locating the settled region needs no cells, no adjacency, and no groups.
- The regions may overlap; their union, not their sum, is the settled set. Since
  each `x` in `Z(S)` lies in `R_s` for its own nearest stone, the intersections
  `R_s ∩ V_s` do partition `Z(S)`.
- This is a statement about the settled point set only. It does not make capture
  local, and it does not weaken A10.

## A16. Each per-stone settled region is star-shaped and contains the disk

**Claim.** Fix a stone `s` and a unit vector `u`, and write `x(t) = s + t u`.
Then `h(t) = dist(x(t),L(S)) - t` is non-increasing, `h(0) >= 2r`, and

```text
{ t >= 0 : x(t) in R_s(S) } = [0, T(u)]    with  T(u) >= r.
```

`T(u) = r` exactly when `s + 2r u` is a legal center. `T(u)` may be infinite,
and because `D` is convex a ray leaves `D` once and never re-enters, so
truncating `T(u)` at the exit distance changes nothing inside `D`.

**Proof.** `dist(.,L(S))` is 1-Lipschitz, so for `t2 > t1`,
`dist(x(t2),L) <= dist(x(t1),L) + (t2-t1)`, hence `h(t2) <= h(t1)`. The
separation requirement of `s` itself removes the open disk `B(s,2r)` from
`L(S)`, giving `h(0) = dist(s,L(S)) >= 2r`. Since `h` decreases at rate at most
`2`, `h(t) >= 2r - 2t`, which is positive for `t < r`; monotonicity then makes
`{ h >= 0 }` an interval containing `0`, so `T(u) >= r`. If `s + 2r u` is legal
then `dist(x(r),L) <= ||r u - 2r u|| = r`, so `h(r) <= 0` and `T(u) = r`.
Conversely if `T(u) = r`, the realising legal center `c` satisfies
`||c - x(r)|| = r` and hence `||c-s|| <= 2r`, while legality forces
`||c-s|| >= 2r`; equality puts `c` on the ray, so `c = s + 2r u`.

Consequence: `R_s(S)` is star-shaped about `s` with a single-valued radial
boundary. Locating a two-dimensional level set becomes one scalar equation per
direction.

## A17. The radial boundary is a minimum of closed-form feature roots

**Claim.** With `s`, `u`, `x(t)` and `T(u)` as in A16,

```text
T(u) = min over c in L(S) of t_c,
t_c = ||c-s||^2 / (2 u.(c-s))   when u.(c-s) > 0, and infinity otherwise.
```

The minimum is attained by a point of the A6 candidate family, and each family
member has a closed-form root:

```text
exclusion circle, centre q    t = (4r^2 - ||s-q||^2) / (2(u.(s-q) -/+ 2r))
inset edge line X             t = (X - s_x) / (1 -/+ u_x), likewise for Y
vertex v of L(S)              t = ||v-s||^2 / (2 u.(v-s))
```

For `q = s` the first reduces to `t = r`, recovering A16's disk. A candidate is
admissible only when the point realising it is a legal center.

**Proof.** `dist(x(t),L) = min over c in L of ||x(t)-c||`, so `h = min_c h_c`
with `h_c(t) = ||x(t)-c|| - t`, each non-increasing because distance to a point
is 1-Lipschitz. Expanding `||x(t)-c||^2 = t^2` gives
`||s-c||^2 + 2t u.(s-c) = 0`, hence the displayed `t_c`. Since every `h_c` is
non-increasing and `h(T) = 0`, we get `h_c(T) >= 0` for every `c`, so
`T <= t_c` always, with equality at a minimiser. At `t = T` we have
`dist(x(T),L) = T >= r > 0`, so `x(T)` is not in `L(S)` and A6 applies: a
closest point belongs to the enumerated family. Substituting the closed-form
distance to a circle or to a line and setting it equal to `t` produces the
displayed roots; in the circle case the `t^2` terms cancel and the root is
linear in `t`.

Consequences:

- The boundary is computed without sampling and without iteration.
- Admissibility is decided by a legality test on the realising point, which is
  exactly the restriction of a full circle or full line to the surviving part of
  `boundary L(S)`. The boundary of `L(S)` never has to be constructed, so no
  enclosure combinatorics among three or more stones is ever enumerated.
- Every admissible candidate yields an upper bound on `T(u)`, so a partial scan
  is always safe; and `t_c >= ||c-s||/2` by the triangle inequality, so visiting
  candidates in increasing distance from `s` permits an exact early stop.

## Numerical policy

The claims above use exact arithmetic. JavaScript `Number` arithmetic does not
make them literally exact. In particular, the exact A5 comparison is

```text
delta(v) < rho(s,v)    // group has an escape
```

The implementation compares squared distances in two stages. First it encloses
each subtraction, square, sum, and final signed margin with outward-rounded
binary64 intervals. A strictly positive lower bound proves an escape; a
nonpositive upper bound proves capture. If the interval contains zero, it
reconstructs every input binary64 value as an exact dyadic rational and compares
the two squared distances with arbitrary-size integer arithmetic.

Thus the fast path carries an explicit rounding-error bound, while the fallback
settles every comparison exactly for the coordinates produced by the geometry
engine. Equality is not an escape because A5 requires a strict inequality. No
numeric tolerance is added to `rho`, and the former fixed capture deadband is
not part of the implementation or rules.

Legal-center membership admits a center at distance `2r - coordinateEpsilon`, so
the radius computed from A17 satisfies `T(u) >= r - coordinateEpsilon/2` rather
than `T(u) >= r` exactly. This affects the shaded settled region only. It does
not enter capture, which is decided by A5 through the comparison above.

A16 and A17 assume every stone carries the same radius `r`, so that every
exclusion circle has radius `2r`. The closed forms would have to be rederived
for a variant with per-stone radii.
