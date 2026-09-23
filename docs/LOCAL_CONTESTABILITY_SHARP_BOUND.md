# A universal locality bound of (2 + 4 sqrt(2))r

2026-09-22. Exact Euclidean geometry, square territory board, square inset for
legal centers, and the rules in `reference/RULES.md`. This improves the general
8.828427r bound in [the earlier note](LOCAL_CONTESTABILITY.md).

**Theorem.** Keep an owner stone s and every stone whose center is within

    D = (2 + 4 sqrt(2))r = 7.656854249492381... r

of s, and remove any or all more distant stones. The individual contestability
of s's board-clipped Voronoi cell is unchanged. The board and legal-center inset
must remain in the query.

Consequently, insertion or removal beyond D cannot change this individual
status. This is not a bound on group connectivity, capture cascades, or changes
to every part of the Voronoi cell. Nor is it a floating-point error guarantee.

## Normalize and define the projection test

Scale so r = 1. Let B be the square board, I its inset by 1, and W the owner's
cell after distant stones have been removed. For v in B write

    rho(v) = |v-s|,
    q(v) = projection of v onto I,
    delta(v) = |v-q(v)|.

Always delta <= sqrt(2). Each retained stone a satisfies |a-v| >= rho(v)
when v is in W. Therefore

    |a-q(v)| >= rho(v) - delta(v).

If rho(v) - delta(v) >= 2, the projection is a legal placement in the reduced
position and strictly contests v, since delta(v) < rho(v).

## Case 1: the projection test succeeds somewhere in W

The function rho-delta is continuous and is zero at s. Along the segment from
s to a successful v (which lies in the convex cell W), choose w with

    rho(w) - delta(w) = 2.

Put rho = rho(w), delta = delta(w), and q = q(w). Then

    rho = 2 + delta <= 2 + sqrt(2).

The projection q is legal against all retained stones. Orthogonal projection
onto the convex inset fixes s and cannot increase its distance, so

    |q-s| <= rho.

For any omitted stone a, |a-s| > D. The two useful numerical inequalities are

    D > 2(2 + sqrt(2)),
    D > (2 + sqrt(2)) + 2.

They give

    |a-w| >= |a-s| - rho > rho,
    |a-q| >= |a-s| - |q-s| > 2.

Thus w is still owned by s in the full position, and q remains legal there.
Since |q-w| = delta < rho, the full cell is contestable.

This case explicitly handles a reduced cell that acquired new territory when
the distant stones were removed, including territory near a corner.

## Case 2: the projection test fails everywhere in W

For every v in W,

    rho(v) < 2 + delta(v) <= 2 + sqrt(2).

Since D > 2(2 + sqrt(2)), an omitted stone cannot cut W at all: its distance
from any v in W is greater than rho(v). Hence the full and reduced cells agree.

Suppose the reduced cell is contestable. Choose a territory witness v in W and
a legal placement p in I with |p-v| < rho = |s-v|. Both s and p belong to the
disk centered at v of radius rho, intersected with I. Bound their separation
according to the location of v relative to I.

### v is inside I

Here delta = 0 and rho < 2, so |p-s| < 2 rho < 4.

### v is outside I in exactly one coordinate

Here 0 < delta <= 1. The inset is contained in a half-plane whose boundary is
delta away from v. Let q be the perpendicular projection onto that boundary.
For any x in the intersection of this half-plane and the radius-rho disk,

    |x-q|^2 <= rho^2 - delta^2.

Indeed, expand |x-v|^2 and use (x-q).(q-v) >= 0. Thus s and p lie in a disk
of radius sqrt(rho^2-delta^2) about q, giving

    |p-s| <= 2 sqrt(rho^2 - delta^2)
           < 2 sqrt((2+delta)^2 - delta^2)
           = 4 sqrt(1+delta)
           <= 4 sqrt(2).

Additional inset edges can only restrict this intersection further.

### v is outside I in both coordinates

After reflection and translation, all vectors x-v for x in I have nonnegative
coordinates. In particular (s-v).(p-v) >= 0, so

    |p-s|^2 <= |s-v|^2 + |p-v|^2 < 2 rho^2.

Consequently

    |p-s| < sqrt(2)(2 + sqrt(2)) = 2 + 2 sqrt(2) < 4 sqrt(2).

Corners therefore give a smaller separation bound than a straight edge.
These cases also cover narrow boards; clipping by further boundaries never
weakens an inequality.

### The witness survives every omitted stone

In every case |p-s| < 4 sqrt(2). Hence, for each omitted stone a,

    |a-p| >= |a-s| - |p-s| > D - 4 sqrt(2) = 2.

The same placement remains legal in the full position, where the cell is
unchanged. It still contests v.

## Finish and scope of the result

Both cases show that reduced-position contestability implies full-position
contestability. The reverse implication follows from monotonicity: removing
stones expands both the owner's cell and the legal-placement set. This proves
the theorem, including simultaneous removal of all stones beyond D.

The proof preserves the strict contest inequality and permits legal tangency
at separation 2. Keep stones at exactly D when applying the stated cutoff.

The constant is an upper bound, not a proof of optimality. Taking the reported
7.484r construction as given leaves a gap of about 0.173r. Claude's scratch
write-up already identifies the same constant for straight-edge blocking; the
additional result here is the full square-board removal-invariance proof,
including corners and distant stones that could cut the reduced cell.

## Differential check

The existing `iteration_lab/locality.rs` probe now also evaluates this cutoff.
On its nine fixture families and 216 owner queries, all 216 new-cutoff results
agree with the full-board analytic predicate (157 contestable, 59 not). Every
query removes at least one distant stone. Including the original and adaptive
cutoffs, the run reports 648 matching comparisons. Both locality tests pass:

```sh
cargo test -p vgo-core --features iteration-lab --lib iteration_lab::locality -- --nocapture
```

These fixtures reuse the existing floating-point solver. They are a regression
check, not an independent proof or a production rounding contract.
