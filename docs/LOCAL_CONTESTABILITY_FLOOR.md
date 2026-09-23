# An exact five-stone locality floor: 7.4918r

2026-09-22. This is an explicit square-board configuration in which removing a
stone at distance **7.4918r** changes another stone's individual cell from
non-contestable to contestable. The distant stone does not change the owner's
Voronoi cell; it blocks the last available placement. The construction uses
the two-wall-stone edge geometry described in Claude's earlier scratch work,
with a guard stone added and a full-cell exact certificate.

All decimals defining the example below are exact terminating decimals.

## Coordinates

Measure lengths in stone radii, so r = 1. Take the territory board [0,12]^2;
legal centers must lie in [1,11]^2.

| Stone | x/r | y/r | Role |
| --- | ---: | ---: | --- |
| s | 1 | 1 | Owner whose individual status changes |
| g | 1 | 3 | Guard closing the upper part of s's cell |
| b1 | 3.2371 | 2.8783 | First wall stone |
| b2 | 5.2070 | 2.5322 | Second wall stone |
| t | 8.4918 | 1 | Distant decisive blocker |

The minimum center separation is exactly 2, attained by s and g. In
particular, |b1-b2|^2 = 4.00029122 > 4. All centers are in the inset.

To use the standard unit board, divide every coordinate by 12 and set r=1/12.
Colors do not affect this individual-cell claim. To make it an actual capture,
make s Black and g, b1, b2, t White. Start with t absent and White to move.
The four-stone starting position has no settled groups. Playing t is legal and
captures exactly s; the engine regression checks this complete move. The
five-stone diagram represents the provisional state before capture resolution.

## What changes when t is removed

The owner's cell is identical in the two positions. Its vertices are

    (0, 0)
    v = (3.746687072549282..., 0)
    w = (2.067459501139868..., 2)
    (0, 2).

The nonintegral coordinates have the exact definitions

    v.x = (|b1|^2 - |s|^2) / (2(b1.x-s.x)),
    w.x = v.x - 2(b1.y-s.y)/(b1.x-s.x).

With t present, every legal-center candidate capable of contesting this cell
is blocked. After removing t, the exact placement

    p = (6.4928, 1)

is legal and strictly contests v:

    min over a in {s,g,b1,b2} of |p-a|^2 = 4.00091848 > 4,
    |p-t|^2 = 3.996001 < 4,
    |v-s|^2 - |v-p|^2 = 73490231 / 23303125000 > 0.

Since t and s share their y-coordinate,

    |t-s| = 8.4918 - 1 = 7.4918.

Reversing the removal also gives a legal insertion that changes the owner's
individual status, before any group capture resolution.

## Exact certificate that the whole cell is non-contestable before removal

For fixed p, the difference |v-s|^2 - |v-p|^2 is affine in v. A placement can
therefore contest some point of the convex polygon exactly when it contests
at least one of its vertices. The candidate region is the union of the four
open disks centered at the cell vertices, each with radius equal to that
vertex's distance from s, intersected with the legal-center inset.

That entire candidate region is contained in the rational rectangle

    Q = [1, 6.4934] x [1, 3.463].

The x bound uses y >= 1 when clipping the disks about the bottom vertices;
using their full disk widths would be unnecessarily loose. For every cell
vertex u, write rho^2 = |u-s|^2. The exact checker verifies

    6.4934 >= u.x,
    (6.4934-u.x)^2 + max(1-u.y, 0)^2 >= rho^2,
    3.463 >= u.y,
    (3.463-u.y)^2 >= rho^2.

Next partition Q into its five nearest-stone Voronoi regions. On each polygon,
the squared distance to its owning stone is convex and has its maximum at
some vertex. All polygon vertices are rational because the clipping boundaries
are rational lines. The maximum squared distances are:

| Nearest stone | Maximum squared distance on its part of Q |
| --- | ---: |
| s | 3.636928521397 or less |
| g | 2.139469786574 or less |
| b1 | 3.997712043834 or less |
| b2 | 3.998994145074 or less |
| t | 3.998994145074 or less |

Every maximum is strictly less than 4. Thus the five open radius-2 exclusion
disks cover all of Q, and in particular cover every possible contester. This
proves the owner is non-contestable with t present, including isolated legal
tangencies that a sampled picture could miss.

## Reproduce

[The standalone certificate](../diagnostics/check_locality_floor.py) uses only
Python's standard library and `Fraction`; all decision comparisons are exact.
Only the printed summaries are converted to floating point.

```sh
python diagnostics/check_locality_floor.py
cargo test -p vgo-core --features iteration-lab --lib iteration_lab::locality -- --nocapture
```

Both pass. The Rust regression independently checks the full and reduced
positions with the existing analytic cell predicate, legality of p, and the
actual White placement at t capturing just s.

Together with [the universal upper bound](LOCAL_CONTESTABILITY_SHARP_BOUND.md),
this gives

    7.4918r <= optimal universal locality radius <= 7.656854249492381... r.

This construction establishes a floor, not an optimality claim.
