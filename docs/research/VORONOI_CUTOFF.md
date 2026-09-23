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
  `n - 1` per cell is `O(n log n)`, worse than the clipping it saves. This is
  still the open question; a grid walk was tried and did not pay (below).

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

## A grid walk was tried and reverted

Replacing the per-cell sort with a uniform grid walked outward ring by ring is
the obvious way to drop the `O(n log n)`. It was implemented, made correct, and
then reverted for being slower where it matters:

| stones | per-cell sort | grid ring walk |
| -----: | ------------: | -------------: |
|     35 |       60.5 us |      108.5 us  |
|    140 |      590.8 us |      617.2 us  |
|    500 |     7012.7 us |     6251.9 us  |

It only wins past ~300 stones, and production plays at 35.

Why it lost: a ring is dismissed only once `(m - 1) * side` exceeds twice the
circumradius, and with cells of `8r` that takes two rings -- a 5x5 block of
cells. That gathers far more candidates than the ~8 the sorted walk clips, and
each ring still sorts its own members to keep the clip sequence a function of
the position rather than of bucket insertion order. The sort never went away;
it moved and multiplied.

Three defects were found and fixed along the way, all worth knowing if this is
attempted again:

- The ring membership test compared against `column - ring` in `usize`, which
  underflows at the low edge and silently dropped cells. Chebyshev distance
  (`c.abs_diff(column).max(r.abs_diff(row))`) is the version that works.
- `dimension` was `floor(1 / side)`, leaving the last cell oversized. The ring
  bound assumes every cell is exactly `side` wide, so stones clamped into that
  cell were nearer than the bound claimed. `ceil` fixes it.
- Rings 0 and 1 both touch the centre cell, so the walk cannot stop before
  finishing ring 1 however small the circumradius is.

The grid also cannot preserve bit-identity: its clip order differs from the
sorted walk's, and clipping is not associative in `f64`. Measured over 240
boards the polygons agreed to 8.9e-16 (4 ulps) and areas to 1.4e-16, against a
~3 ulp baseline the existing code already has from permuting stone order. A
120-sample shard against a real model was unchanged, so the perturbation is far
below anything that moves a discrete decision -- but "identical geometry" stops
being available as a test.

## The collinearity square roots were not the cost

`normalize_polygon` samples at ~18% of self-play CPU and runs on every vertex
of every intermediate polygon, once per clip. Its collinearity predicate is

```text
|cross| <= COLLINEAR_EPSILON * max(|ab| + |bc|, 1)
```

which reads as two square roots per vertex, about nine per call. Instrumenting
the real clip sequence showed those square roots never change an outcome: over
46,424 calls on random boards and 69,368 on tangent-packed ones, the pass
removed **zero** vertices and never needed a second iteration. The clipper does
not produce collinear vertices at this tolerance, so every square root was
proving a negative.

Replacing them with a one-sided squared-length rejection -- sound by
`(|ab| + |bc|)^2 <= 2 (|ab|^2 + |bc|^2)`, falling through to the exact form when
it cannot decide -- measured **no improvement**: 2790 us against a 2800 us
baseline on `mcts/32-sim/spatial`, inside the noise. Reverted.

The lesson is that `sqrt` is one instruction and the compiler already knew it.
The 18% is the polygon traversal, not the arithmetic inside it: bounds checks,
the modular indexing for the cyclic neighbours, and the `Vec` allocated per
pass. An optimization here has to remove the walk, not the square roots.

## Status

The cutoff is implemented and in `compute`. Candidates are still sorted per
cell, so the build is `O(n^2 log n)` in the worst case and `O(n k)` in the
clipping that dominates it; the sort is what to attack next, but not with a
uniform grid at these board sizes.
