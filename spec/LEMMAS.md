# Lemmas

Everything here follows from [`FORMAL.md`](FORMAL.md). IDs are stable, and code
cites them in brackets, such as `[E5]`, at the shortcut they justify.

| prefix | meaning |
|---|---|
| A | properties of the game, in exact real arithmetic |
| E | engine shortcuts: why a faster computation gives the same answer |
| N | floating-point facts those shortcuts depend on |
| O | open: claimed or hoped for, and not proved |

Every lemma has a status. **Proved** has a complete written proof.
**Conditional** has one under assumptions it states. **Open** has none. Tests and
game comparisons can refute a lemma, and they never change its status to Proved.
None of these proofs has been machine-checked; see [`lean/`](lean/README.md).

A1 to A17 have status Proved, for a valid position in exact arithmetic. The E and N
lemmas follow them, and the [open problems](#open-problems) are at the end.

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

The claims A1 to A17 use exact arithmetic. JavaScript `Number` arithmetic does not
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

## Engine lemmas

Notation: `n` is the stone count, `|x|` is Euclidean length, and `u = 2^-53` is unit
roundoff. A rounded operation returns `x(1 + t)` with `|t| <= u`, absent overflow and
underflow, and `FMA(a, b, c)` rounds `ab + c` once. Rounding is monotone: `x <= y`
implies `rnd(x) <= rnd(y)`.

### E1. Tile candidates

*Status: Proved.* Used by `sweep_tile`.

**Claim.** Let `T` be a set of points, `c` a point and `h` a bound with
`|p - c| <= h` for all `p` in `T`. Let `K` be a set of stones containing distinct
`a` and `b`, and let `d = max(|a - c|, |b - c|)`. Let `R` be any subset of
`{s in K : |s - c| >= d + 2h}`. Then for every `p` in `T`, the two smallest
distances from `p` to `K` equal those from `p` to `K - R`.

**Proof.** Fix `p`. By the triangle inequality `|a - p| <= d + h` and
`|b - p| <= d + h`. For `s` in `R`, `|s - p| >= |s - c| - h >= d + h`. Since `a` and
`b` are not in `R` (each is at most `d` from `c`, and if `h = 0` and `d = 0` then `a`
and `b` coincide, contrary to distinctness of stones), `K - R` holds two stones that
are no farther from `p` than any stone of `R`. So removing `R` cannot change the two
smallest values. QED

**Hierarchy.** Applying this at each tile, starting from all stones, gives by
induction that a tile's list holds each of its pixels' two smallest values. Any two
stones may serve as `a` and `b`, so a child needs only its parent's list.

### E2. Disc stamping

*Status: Proved.* Used by `stamp_disc`.

**Claim.** For a finite set `C` of stones and computed squares `q(p, s)`,
`min over s in C of q(p, s) <= r^2` holds exactly when `q(p, s) <= r^2` for some `s`
in `C`. With the same `q`, stamping sets the same bits as keeping the minimum.

**Proof.** A minimum over a finite set is attained by an element. QED

The stamped range must contain every pixel that passes the test. A passing pixel has
`|x - s.x| <= r(1 + 4u)`, so its column is within `(s.x +- r) * width - 1/2` up to
rounding. The range is widened by one pixel on each side, which covers that.

### E3. Binary distance transform

*Status: Conditional on `width + height < 65535`.* Used by
`squared_distance_transform_into` and `envelope_row`.

**Claim.** Given a mask on a `w` by `h` grid with at least one true cell, the transform
returns the exact squared distance from every cell to the nearest true cell. On an
all-false mask it returns a constant larger than any squared distance on the grid.

**Proof.** Let `D(x, y)` be the squared distance to the nearest true cell. Grouping true
cells by column gives `D(x, y) = min over c of (x - c)^2 + G(c, y)`, where `G(c, y)` is
the squared distance within column `c`, or infinite when `c` has no true cell.

*Columns.* The forward pass sets each cell to 0 if true, otherwise one more than the
cell above, and so holds the distance to the nearest true cell at or above it. The
backward pass does the same from below and takes the minimum. Values are capped at
`m = w + h + 1`. A column with no true cell holds `m`, and `m^2 > w^2 + h^2 >= D`, so
such a column cannot supply the minimum when a true cell exists.

*Rows.* Fix a row and write `P_c(x) = (x - c)^2 + G_c`. For `a < b`, `P_b - P_a` is
linear and strictly decreasing in `x`, so `{x : P_b(x) < P_a(x)}` is an up-set. Keep a
stack of parabolas `s_0 < s_1 < ...` with starts `t_0 < t_1 < ...`, with the invariant
that among columns seen so far `P_{s_i}` is minimal on integers in
`[t_i, t_{i+1})`, the last interval ending at `w`.

To add column `u`: if `P_u(t) < P_s(t)` at the top's start `t`, the up-set property means
`u` beats the top on its whole interval, so pop it. If the bottom is popped this way, `u`
replaces it at start 0. Otherwise let `a` be the surviving top. Expanding,
`P_u(x) < P_a(x)` is equivalent to `x > N / (2(u - a))` with
`N = u^2 - a^2 + G_u - G_a`, so the first integer where `u` wins is
`floor(N / (2(u - a))) + 1`, computed by `div_euclid` with a positive divisor. Because
`a` was not beaten at its own start, this exceeds that start. Push `u` there if it is
below `w`. This restores the invariant, and reading the stack back gives the minimum at
every `x`.

*Exactness.* `envelope_row` works in `i32`, and the transform asserts `m < 2^14`. Every
height is at most `w^2 + m^2 < 2^29`, and `N` is a sum of two squares below `2^28` and two
values below `2^28`, so nothing overflows. Conversion to `f64` is exact. QED

Since the values are exact integers, any correct transform produces the same bits.

`envelope_row` skips columns holding `m^2`, which is what the first half shows they cannot
supply the minimum. Some column in every row holds a smaller value, because a true cell
exists and its column's distance is below `m` in every row, so a skipped first column is
beaten when that one arrives. The stack keeps each parabola's value at its own start
instead of recomputing it, and the output writes each interval as a run: neither changes
what is computed.

### E4. Voronoi cutoff

*Status: Proved in exact arithmetic.* Used by `compute_cell`. The argument with
measurements is in [`VORONOI_CUTOFF.md`](../docs/history/VORONOI_CUTOFF.md).

**Claim.** Let `P` be a polygon containing `s`, with circumradius `R` about `s`. If
`|o - s| > 2R`, the bisector of `s` and `o` does not cut `P`. If stones are visited by
non-decreasing distance, every later stone also satisfies this.

**Proof.** For `p` in `P`, `|p - o| >= |o - s| - |p - s| > 2R - R >= |p - s|`, so `P` is
strictly on `s`'s side. Clipping only shrinks `P`, so `R` does not increase, while
distance does not decrease. QED

### E5. Cell reuse

*Status: Proved.* Used by `compute`. This is a statement about the program, so it holds
for floating-point execution, not only in exact arithmetic.

**Definitions.** For stone `s` of ordered list `S`, `compute_cell` is a deterministic
program. Let `o_1, ..., o_k` be the stones it clips against, `d_j` their computed
distances with `d_0 = 0`, `P_j` the polygon after `j` clips, and `stop_j` the computed
value `2 R(P_j) + eps`. Let `reach(S, s)` be infinite if `P_k` is empty, otherwise
`max(d_k, stop_k)`.

**Claim.** Let `S'` be `S` with stone `p` appended, and let `D` be the computed distance
from `s` to `p`. If `D > reach(S, s)`, then the program on `S'` clips against exactly
`o_1, ..., o_k`, produces a bit-identical cell, and has the same `reach`.

**Proof.** A stone's sort key depends only on its distance from `s` and its index, and
appending leaves both unchanged for stones of `S`. The key of `p` sorts after `o_k`
because `D > d_k` (a NaN fails that test, so the claim does not apply). So the first `k`
visits are the same stones. Each clip depends only on the polygon and the two stones
involved, so by induction `P_j` and `stop_j` agree bit for bit, and the cutoff test at
each of those visits has the same outcome.

At visit `k + 1` the program on `S` either has no stones left or stops at `o_{k+1}` with
`d_{k+1} > stop_k`. On `S'` the visited stone is `o_{k+1}` or `p`, and both exceed
`stop_k`, so it stops. The sequence is therefore unchanged. Everything after the loop is
a function of `P_k`. Finite `reach` means `P_k` is non-empty, so the empty-polygon exit
is not involved. QED

**Chain.** Since `reach` is preserved, induction extends this across any number of
appended stones, so a cell reused for generations equals a fresh build. Removing stones
shifts indices, so the parent link is cleared by `with_stones` and only `place` sets it,
and only for an append.

Not cutting the final polygon would not be sufficient. A stone can cut an intermediate
polygon and be clipped away later, which gives the same shape with different rounding.

### E6. Bounded snapping

*Status: Conditional on every stone lying in the inset `C` of `FORMAL.md`, up to
`COORDINATE_EPSILON`.* Used by `nearest_bounded`. This holds for positions reached by
`place`, but not for hand-built positions with stones far outside `C`.

Write `m = SNAP_MARGIN = 1e-6`, `e = COORDINATE_EPSILON = 1e-7`, `w = 2r + m`, and let
`Pi` be coordinate-wise clamping to the box used by `nearest_from`.

**Claim.** Let `F` be the point the unbounded search returns and `B` the bounded result
for `reach = rho`, where the bounded search uses padding `64m`. If `F` is within `rho`
of the query `q` then `B = F`. Otherwise `B` is either illegal or farther than `rho`
from `q`.

**Proof.**

*Movement.* `clear_by_margin` makes at most 4 passes, each moving the point directly away
from its nearest stone, to distance `2r + m`. A legal candidate starts at least `2r - e`
from every stone. If earlier passes moved it by `M`, it is at least `2r - e - M` from
every stone, so the next pass moves at most `m + e + M`. From `M = 0` that gives at most
`1, 3, 7, 15` times `m + e`, so the total is at most `15(m + e) < 17m`. Call this bound
`M*`.

*Coverage.* `Pi` is a projection onto a convex box, so it does not increase distances, and
a stone in `C` is within `sqrt(2)(m + e)` of the box. A stone's candidate is
`Pi(stone + w * v)` for a unit vector `v`, so it is within `w + sqrt(2)(m + e) < w + 2m` of
the stone. If that candidate is within `rho + M*` of `q`, the stone is
within `rho + M* + w + 2m < rho + 64m + w` of `q`, and `for_each_near` reports it, as does
the scan of squared distances used below the bucket threshold. The vertex grid reports
every vertex within `rho + 64m`, and the other candidates (the clamped query and the four
sides) are considered directly. So every candidate within `rho + M*` is in the bounded
list.

*Selection.* The unbounded search returns the legal candidate of least key, moved by
`clear_by_margin`. The bounded search rejects any exact tie in its list by falling back
to the unbounded search, so otherwise its keys are distinct.

If the unmoved winner `f` is within `rho + M*`, it is in the list, and it is the least
legal one there because the list is a subset of all candidates. Then `B = F`.

If `f` is beyond `rho + M*`, then `F` is beyond `rho`. Every legal candidate is at least
as far as `f`, so `B`, moved by at most `M*`, is also beyond `rho` or is reported illegal.

Finally, if `F` is within `rho` then `f` is within `rho + M*`, which is the first case. QED

A snapped point is kept only if it lies inside its cell, which is within half a diagonal
of the cell's centre, so that half diagonal is a valid `rho`. Rounding in that check is
far smaller than the slack in `64m`.

### E7. Threshold pruning and test order

*Status: Proved in exact arithmetic.* Used by `visit_candidates_within`.

**Claim.** (a) A stone at distance `a` from `q` with `|a - 2r| >= t` has its projected
candidate at distance at least `t` from `q`. (b) For pure predicates `A` and `B`, testing
`A` before `B` gives the same `exists c : A(c) and B(c)` as the opposite order.

**Proof.** (a) The candidate lies on the ray from the stone through `q` at distance `2r`
from the stone, so it is `|a - 2r| >= t` from `q`. When `a` is too small to give a
direction, all four candidates are `2r` from the stone and the same bound holds up to `a`.
(b) Conjunction of pure predicates is commutative. QED

The code tests `a^2` against the squares of `(2r + t)(1 + 1e-12)` and
`max(2r - t, 0)(1 - 1e-12)`, skipping only outside them. On a unit board rounding is near
`1e-16`, so a skipped stone's candidate cannot pass the comparison it was spared.

Part (b) does not cover a visitor that records what it sees, so `escape_witness` keeps
legality first.

### E8. Derived data on a position

*Status: Proved.* Used by `Position::legal_index`, `shared_cells` and
`adopt_legal_index`.

**Claim.** A cached value computed from `(r, stones)` by a deterministic function equals a
fresh computation, provided it is discarded whenever those inputs change.

**Proof.** A deterministic function of equal inputs gives equal results. Stones change only
through `with_stones` and the constructors, which all reset the
caches, and `adopt_legal_index` is called only when no self-capture changed the stones. QED

### E9. A pixel inside a stone's disk is settled

*Status: Proved, with rounding bounded by the margin.* Used by `classify_row`.

**Claim.** Let `e` be `COORDINATE_EPSILON`. If a pixel `p` has nearest stone `s` and its
computed squared distance `q` satisfies `q <= (r - e)^2`, the three-case classification
returns settled, so it may return settled without reading the sampled field.

**Proof.** Write `d = |p - s|`, so `d <= r - e` up to relative rounding near `1e-16`.
Legal-set membership requires every stone at distance at least `2r - e`. Each legal-set
vertex meets that too: other stones by the same test, and the one or two it was built on
because it lies on their circles of radius `2r`, up to rounding, or farther when a
discriminant is clamped to zero. So every candidate `c` the exact test can accept has
`|c - s| >= 2r - e - 1e-15`, and by the triangle inequality

`|c - p| >= |c - s| - d >= 2r - e - 1e-15 - (r - e) = r - 1e-15 > d`,

since `d <= r - e` and `e = 1e-7`. The exact test therefore accepts nothing and says
settled, and so does the vertex check that precedes it.

The cheap "certainly unsettled" case cannot fire first. It needs `d > D + h/sqrt(2)`, where
`D` is the distance to the nearest sampled cell centre and `h` the spacing. A sampled cell
is either a legal centre, so `D >= r - 1e-15`, or holds a vertex, whose centre is within
`h/sqrt(2)` of it, so `D + h/sqrt(2) >= r - 1e-15`. Either way `D + h/sqrt(2) > d`. The
remaining case is settled, so all paths agree. QED

This holds whether or not [O1] does, because it depends only on the candidates being clear
of stones, not on the sampled grid seeing every legal point.

### E10. The vertex grid replaces the exact test's vertex scan

*Status: Proved.* Used by `VertexGrid::any_closer` and `none_closer_than_except_vertices`.

**Claim.** Let `A` be whether some legal-set vertex passes `length_below(v - p, d)`, and `B`
whether the exact test accepts a non-vertex candidate. The exact test's answer is
`not (A or B)`, and `any_closer` computes `A`.

**Proof.** The exact test accepts a vertex exactly when that comparison passes, and it
tests no other property of a vertex, so its answer is `not (A or B)`. `any_closer` applies
the same comparison, to the vertices in the buckets within `d(1 + 1e-9) + 1e-12` of `p`. By
[N1] a passing vertex has computed length below `d`, so its true distance is within that
reach and its bucket is read. Reading more vertices cannot add a false positive, because
each still has to pass the comparison. QED

`has_escape` reads vertices the same way, under [E7]: its comparison passes only within
`reach`, and it asks whether any candidate passes, so the order they are read in is
immaterial. The stone buckets return the same answer as the full scan for the same reason.

### E11. Validity survives a legal move

*Status: Proved.* Used by `Position::mark_valid` and `place`.

**Claim.** If a position validates and `legal_set::contains` accepts `p`, the stones with
`p` appended validate, and so does any subset of them.

**Proof.** `validate` checks the radius, and each stone and each pair of stones on their
own, so a subset of a valid list passes and only `p` needs checking. `contains` requires
`p` finite and applies the same inset expressions. For each earlier stone it requires the
computed `q = FMA(dx, dx, dy^2) >= rnd(m^2)` with `m = 2r - e`, and `validate` tests
`rnd(sqrt(q)) < m` on the same `q`. Rounding is monotone, and `rnd(sqrt(rnd(m^2))) = m` for
binary64 without overflow or underflow, so `rnd(sqrt(q)) >= m` and no issue is raised. QED

### E13. Legal vertices of a position extended by one stone

*Status: Proved.* Used by `vertices::extend`.

**Claim.** Let `S` be a stone list and `S' = S + t`, with `t` appended last. If the
enumeration of `S` dropped no candidate as a duplicate, and no new candidate below shares a
key with a kept vertex, then the following equals the enumeration of `S'`, in order: take
`S`'s list, keep the vertices clear of `t`, and insert each admissible candidate built from
`t` at the end of its section.

**Proof.** A candidate of `S'` is built from the walls, from corners, or from one or two
stones. Its coordinates depend only on the stones that built it, so a candidate not
involving `t` has the same coordinates under `S` and `S'`, and the arithmetic is unchanged
because `t` is `second` in every pair, as it is in the full loop.

*Admission.* A candidate is admissible when it is finite, in the inset, and clear of every
stone it was not built on. Under `S'` that is the same test plus clearance from `t`, which
is the comparison `extend` applies. A candidate rejected under `S` stays rejected.

*Duplicates.* The enumeration keeps the first of any candidates sharing a key. With no drop
under `S`, every admissible candidate of `S` is in its list, so the filtered list holds
every admissible candidate of `S'` not involving `t`, with distinct keys. If no new one
shares a key with those or with another new one, nothing is dropped under `S'` either.
Otherwise `extend` declines and the full enumeration runs.

*Order.* The enumeration visits corners, then walls by stone, then pairs by `first` and
`second`. `t` has the highest index, so its wall candidates end the wall section, and each
pair `(f, t)` ends the section for `f`. QED

### E14. A clip that cuts nothing changes nothing

*Status: Proved.* Used by `clip_half_plane` and `normalize_polygon`.

**Claim.** (a) If every vertex of a polygon is on the kept side, and the polygon came from
`normalize_polygon`, and no two neighbouring vertices are within `EDGE_EPSILON`, then the
clip and the normalization after it leave the polygon unchanged. (b) A vertex with
`|cross| > c * max(L, 1) * (1 + 1e-9)`, where `L` is the sum of the absolute components of
both edges, is not removed as collinear.

**Proof.** (a) With no vertex outside there is no crossing, and each vertex is copied with
its source, so the clip's output is the polygon. Normalization first merges coincident
neighbours, of which there are none. Its collinearity pass is a deterministic function of
the polygon, and `normalize_polygon` returns only when that pass changed nothing or fewer
than three vertices remain, so it changes nothing here either. The spike case, where
removing a collinear vertex brings two vertices together with no pass to merge them, is
exactly what the coincidence test excludes. Since the polygon is unchanged, so is its
circumradius, which may therefore be kept.

(b) The test removes `b` when `|cross| <= c * max(s, 1)` with `s = |ab| + |bc|`. A length is
at most the sum of its components' magnitudes, so `s <= L` up to relative rounding of a few
`u`, far below `1e-9`. QED

## Floating-point lemmas

### E15. A nearest field extends over new stones

*Status: Proved.* Used by `nearest_field`.

**Claim.** Let `S` be a list of stones and `A` a prefix of it. For a pixel `p`, let
`(n, m)` be the two smallest computed squared distances from `p` to `A`, both
`+inf` when `A` is empty. Folding each stone of `S - A` into `(n, m)` by

```text
v = the computed squared distance from p to the stone
(n, m) := (min(n, v), min(m, max(n, v)))
```

gives the two smallest computed squared distances from `p` to `S`, bit for bit.

**Proof.** The update keeps `n <= m` and, by induction on the values folded, `(n, m)`
are the two smallest of the multiset seen so far: a value below `n` becomes the new
smallest and pushes `n` to second, a value between them replaces `m`, and a larger
one changes nothing. So the result depends only on the multiset, not on the order.
A computed distance is a pure function of the pixel and the stone's coordinates, no
value is NaN, and the sweep and the fold evaluate it through one function, so a
stone contributes the same bits whichever path computes it. `A` being a prefix,
compared by the exact bits of each coordinate, makes `A` plus `S - A` exactly the
multiset of `S`. QED

**Capture.** Removing a stone from the middle of the list ends the shared prefix
there, so `A` is not a prefix of the new list and the fold does not apply. Nothing
can be subtracted from a field, which is why the test is a prefix and not a subset
of stones. Such a position is swept instead, and by E1 a sweep gives these same two
smallest values, so the paths agree.

### E16. Far stones leave an exact answer alone

*Status: Proved.* Used by `classify_row` and `KnownAnswers`.

**Claim.** Let `S` be a list of stones and `S'` the same list with stones `T` after it.
Fix a point `p` and a threshold `d`, and let the margin be `m = 1e-5`. If every `t` in `T`
has `|t - p| >= d + 2r + m`, then the exact test gives the same answer for `S'` as for `S`.
The exact test here is the vertex check of [E10] together with
`none_closer_than_except_vertices`, which asks whether any candidate `c` has
`length_below(p - c, d)` and is legal.

**Proof.** The test reads only the stones and the radius. Call a candidate *near* if it
passes `length_below(p - c, d)`, so by [N1] it is within `d` of `p`. It is enough to show
the two lists have the same near candidates, with the same legality.

*New candidates are not near.* A candidate that `S'` has and `S` lacks comes from some `t`:
its projection onto `t`'s exclusion circle, or a vertex on that circle ([E13]). Either way
it is within `2r` of `t` up to rounding and `COORDINATE_EPSILON`, far below `m`, so it is
at least `|t - p| - 2r - m/2 >= d + m/2` from `p`. It is not near.

*Old candidates keep their legality.* If `c` is near, then
`|c - t| >= |t - p| - d >= 2r + m`, so `t` does not exclude it. Clearance from the rest of
`S` is the same test in both lists, so `c` is legal for `S'` exactly when it is legal for `S`.

*Old near vertices are still there, unchanged.* The point `p`, its wall and stone
projections and the vertices are computed from the coordinates alone, so a candidate both
lists produce has the same bits. A vertex of `S` near `p` is legal for `S'` by the last step,
so `S'` enumerates it too. De-duplication cannot replace it, since it merges only points
within about `COORDINATE_EPSILON`, and every new vertex is at least `m` from any near point.

The buckets return what a full scan returns ([E7], [E10]), and the tests are pure, so the
order of visiting has no effect. So the same near candidates pass, and the answers agree. QED

**Use.** A position extends the anchor's stones `S` when they are a prefix of its list, as
in [E15]. Take `d` to be the pixel's nearest-stone distance. A stone that far away is not
the nearest, so `d` has the same bits in the leaf and in the anchor's own field. Two leaves
that each satisfy the claim at `p` therefore both give the answer for `S`, so one can save it
and the other read it. The anchor is keyed by its stones, grid and radius, so an answer never
leaves the board it was computed for.

This covers only the exact test. The two cheap cases run for every pixel of every leaf,
so a pixel the unproved floor of [O1] decides is decided as it was, and the plane does not
depend on what the thread rendered earlier.

### N1. Length comparison

*Status: Proved for `t < 2^511`.* Used by `length_below`. Outside that range the shortcut
can differ near `s = 2^1024`, which no coordinate on a unit board reaches.

**Claim.** Let `s` be the computed `dx^2 + dy^2`, and `length = rnd(sqrt(s))`. Then
`length_below(dx, dy, t)` equals `length < t`.

**Proof.** Both sides use the same `s`, so only the root and the limit can round. If `t`
is NaN or below `1e-140` the shortcut is skipped and the direct comparison runs. If `s` is
NaN both shortcut tests are false, and the direct comparison runs.

Otherwise let `L = rnd(t^2) = t^2 (1 + a)` with `|a| <= u`, which neither overflows nor
underflows in this range. Let `e = 8u`.

If `s > rnd(L(1 + e))` then `s >= t^2 (1 - u)^2 (1 + 8u) >= t^2 (1 + 5u)`, so
`sqrt(s) >= t(1 + 2.4u)` and `length >= t(1 + 2.4u)(1 - u) > t`. The direct comparison is
false, matching the early return. An infinite `s` also lands here.

If `s < rnd(L(1 - e))` then `s <= t^2 (1 + u)^2 (1 - 8u) <= t^2 (1 - 5u)`, so
`sqrt(s) <= t(1 - 2.5u)` and `length <= t(1 - 2.5u)(1 + u) < t`. The direct comparison is
true, matching the early return.

In every other case the function evaluates the direct comparison. QED

### N2. Slack in the tile bound

*Status: Conditional on a playable position with `n >= 2` and `r >= 1e-4`.* Used by
`sweep_tile`. The smallest stone used is `r = 1/50`.

**Claim.** The factor `1 + 1e-9` on the squared limit is enough for E1 to hold for the
values the program stores: at every pixel of the tile, a discarded stone's stored square
is greater than those of the two stones that defined the limit.

**Proof.** All coordinates lie in `[0, 1]`, and computed distances are at most 2. The
computed centre `c'` and half-diagonal `h'` satisfy: every pixel is within `h' + 8u` of
`c'`, because each half-width is computed to within `4u`. The computed `d'` and the true
distance `d` from `c'` to the defining pair satisfy `|d' - d| <= 8u`.

A discarded stone has computed squared distance above `(1 + 1e-9)(d' + 2h')^2`, up to
relative error `8u`. So its true distance from `c'` is at least
`(1 + 4.9e-10)(d + 2h') - 16u`.

For a pixel `p`, it is at least `(1 + 4.9e-10)(d + 2h') - 16u - h' - 8u`, which is at least
`(1 + 4.9e-10)(d + h') - 24u`. The defining stones are within `d + h' + 8u` of `p`.

Stones are at least `2r - 1e-7` apart, so one of the pair is at least `r - 1e-7` from `c'`,
and `d >= 9.9e-5`. Thus `24u + 8u < 3.6e-15` is below `4e-11 (d + h')`, leaving a relative gap
above `4.5e-10` between the discarded stone and the defining pair at `p`. A stored square has
relative error below `4u`, about `4.5e-16`, which cannot close that gap. QED

### N3. Integer order keys

*Status: Proved.* Used by `order_key`, `total_order_key` and `sort_next_prefix`.

**Claim.** The transform `b xor (asr(b, 63) lsr 1)` on a value's bits orders signed
integers exactly as `f64::total_cmp` orders values. Adding the index in the low bits gives
a strict total order on `(value, index)`. Partial sorting by repeated selection then
visits exactly the sequence a full sort gives.

**Proof.** This is the transform `total_cmp` is defined by. Keys are unique, since indices
are, so a strict total order has exactly one sorted sequence, and each selection step
places the correct block in order. A minimum over keys ignores ties only among equal
values, so the callers that depend on the first of equals keep an explicit position scan.
QED

## Open problems

### O1. The settled plane

*Status: Open.* The raster's `settled` plane is decided in three cases: two distance bounds
from the sampled legal set and an exact fallback. The upper bound assumes every legal point
has a sampled cell within a known distance, and that fails for a sliver of legal space
narrower than a cell. Marking the legal set's vertices covers every bounded component, and
the tests agree with the definition on every fixture tried, but no proof shows the plane
equals the definition on every position. `settled_bounds.rs` documents the failure mode.

This affects model input, not the rules: capture uses A5.

To close it, either prove that vertex marks plus the slack suffice, or bound the error.

### O2. Geometry against the real numbers

*Status: Open.* A1 to A17 are real-number theorems. E5 shows reuse matches the existing
floating-point program, which is a different claim from matching the mathematics. Distance
comparisons in capture are exact by construction (see the numerical policy above), but
cell polygons and legal vertices are computed in floating point with tolerances, and no
written bound relates them to the true cells and legal set.

To close it, state a tolerance `t` and prove every computed vertex is within `t` of a true
one, or list which decisions depend on these values and show that they are robust.
