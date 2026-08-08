# Voronoi cells by sorted insertion with a distance cutoff

`voronoi::compute` builds each cell by clipping the unit square against the
bisector to every other stone: `n` cells against `n - 1` bisectors, so the cost
is quadratic in the stone count and stays quadratic however cheap one clip
becomes. Measured on this box, `Analysis::new` runs 94 us at 35 stones and
17,146 us at 500 -- a 14x board costing 182x.

Almost all of that work does nothing. A cell has about six Voronoi neighbours
regardless of how many stones are on the board (Euler's formula bounds the mean
degree of a planar graph below six), so at 500 stones roughly 494 of the 499
clips per cell leave the polygon unchanged.

This document records the rule that skips them, the claims it rests on, and
what has and has not been proved.

## The rule

Process the other stones in increasing distance from the cell's own stone,
clipping as you go, and stop at the first candidate whose distance exceeds twice
the current polygon's circumradius.

```
poly = unit square clipped by the board constraints
for (d, other) in stones sorted by distance from s:
    R = max over vertices v of poly of |v - s|
    if d > 2R: stop
    poly = clip(poly, bisector(s, other))
```

## Claim 1 (soundness of the cutoff)

*If `|o - s| > 2R`, where `R` is the circumradius of the current polygon, then
the bisector of `s` and `o` does not intersect that polygon.*

For any point `p` of the polygon, `|p - s| <= R` by definition of `R`, and

```
|p - o| >= |o - s| - |p - s| > 2R - R = R >= |p - s|
```

so every point of the polygon is strictly closer to `s` than to `o`, and the
half-plane `{x : |x - s| <= |x - o|}` contains the polygon entirely. Clipping by
it is a no-op.

## Claim 2 (soundness of stopping, not just skipping)

*If the candidates are sorted by distance and the cutoff fires at `o`, then no
later candidate can contribute either.*

Clipping only removes area, so the polygon shrinks monotonically and `R` is
non-increasing. Sorting makes `d` non-decreasing. So once `d > 2R` holds it
continues to hold for every later candidate, and the loop may stop rather than
merely skip `o`.

Both claims are what makes the result identical to clipping against all `n - 1`
bisectors rather than an approximation of it.

## What is measured

Simulating the loop over random boards, with the polygon actually maintained
(so `R` falls progressively rather than jumping to its final value):

| stones | packing | mean clips | max | of `n-1` |
| -----: | ------- | ---------: | --: | -------: |
|     35 | random  |       7.40 |  17 |       34 |
|     35 | tangent |       9.80 |  30 |       34 |
|    140 | random  |       7.85 |  27 |      139 |
|    140 | tangent |      10.43 |  45 |      139 |
|    500 | random  |       8.17 |  18 |      499 |
|    500 | tangent |      11.49 |  49 |      499 |

The mean is flat in `n` -- 7.40 at 35 stones and 8.17 at 500 -- which is the
whole point: the per-cell work stops depending on the board size, so the build
goes from `O(n^2)` to `O(n k)` with `k` around 8, or around 11 when stones are
packed at exactly `2r` (tangent, the degenerate case real games produce).

A variant using each cell's *final* circumradius as the bound, which is a lower
bound on the achievable count, gives 7.35 / 9.81 / 7.84 / 10.46 / 8.20 / 11.50.
The realistic loop is within 1% of it, so maintaining `R` progressively costs
essentially nothing against knowing it in advance.

## What is not yet established

- **Ordering.** The claims need candidates sorted by distance, and sorting all
  `n - 1` per cell is `O(n log n)`, worse than the clipping it saves. The
  intended fix is a uniform grid walked outward ring by ring, which enumerates
  in approximately increasing distance. Claim 2 needs *exact* ordering, so
  either the enumeration must be exactly sorted within its guarantees or the
  stopping rule must be weakened to skipping. **This is the open design
  question.**

- **Floating point in the acceptance test.** `d > 2R` is compared in `f64`. If
  `R` is high by an ulp the loop stops one candidate early and the cell is
  wrong -- silently, and in perhaps one position in millions. Comparing with a
  margin (`d > 2R + eps`) turns the failure mode into "a few extra clips"
  instead of "a wrong cell", and costs almost nothing given the counts above.

- **Degenerate cells.** A polygon that clips to empty, or to fewer than three
  vertices, has no meaningful circumradius. The loop must define `R` for those
  cases rather than reading a maximum over an empty vertex set.

- **The bound is not tight.** `2R` is conservative: it admits stones whose
  bisector misses the polygon. That is deliberate -- a tight test would cost
  more than the clip it avoids -- but it means the measured counts above are of
  the rule, not of the true neighbour count (about 5).

## Status

Not implemented. The measurements come from a simulation of the rule against
`Analysis::new`'s existing output, not from a working implementation.
