# Settled Contour — optimization sandbox

Isolated bench for one calculation: the settled (uninteractable) region of a
position. Nothing here is imported by `reference/`; the reference modules are
loaded read-only as the engine and as the correctness oracle.

```powershell
.\experiments\settled-contour\run.ps1
.\experiments\settled-contour\run.ps1 -Candidates v4-pruned-fine -Cases grown-40 -Side 500
```

## Contract

A candidate is a function `position -> svg path` (or `{ d, fillRule }`). It is
scored by rasterising that path with `Path2D.isPointInPath` against the exact
predicate, so a polygon candidate and an analytic candidate are judged by the
same rule and no candidate can be graded against another approximation.

The oracle is `g(x) = dist(x,L) - dist(x, nearest stone)`, settled where `g > 0`
— the same field `settled-contour.js` samples, evaluated through the shipping
`legalSet.distance`. Because `g` is 2-Lipschitz, `|g|/2` at a disagreeing sample
is a lower bound on that sample's distance from the true boundary, which
separates boundary jitter from real geometric error.

## Status

The winner has landed in `reference/src/geometry/settled-contour.js`, and the
axioms it rests on are A15-A17. `v0-shipping` now loads that module, so re-running
the bench validates the real implementation rather than a copy of it: 16 cases,
oracle grid 400x400, **6 mismatches out of 2.56M samples, worst error 4.0e-6
board units**.

The `v0-quadtree` figures below are the sampled renderer it replaced, kept as the
historical baseline.

## Standings

Ten cases, total wall time, oracle grid 200x200:

| candidate | total | mismatches | worst error | vs baseline |
| --- | --- | --- | --- | --- |
| `v0-quadtree-d10` (shipping) | 6257.80ms | 0 | 0 | 1x |
| `v0-quadtree-d8` | 1523.00ms | 2 | 2.2e-4 | 4x |
| `v1-star-128` | 935.90ms | 6 | 5.3e-4 | 6.7x |
| `v3-adaptive-fine` | 48.50ms | 0 | 0 | 129x |
| `v4-pruned-fine` | 37.70ms | 0 | 0 | **166x** |
| `v4-pruned` | 15.30ms | 4 | 5.9e-5 | **409x** |

At a 500x500 oracle (1.5M samples per case) `v4-pruned-fine` disagrees on 3
samples with a worst error of 1.4e-6 board units — 0.001px on a 700px board, and
the same fidelity as the depth-10 quadtree, which itself disagrees on 2 samples
at that resolution.

### The speedup is not uniform

Quadtree cost scales with the *length of the settled boundary*; sweep cost scales
with *stones x rays x features*. Their worst cases are opposite, so the corpus
must cover both regimes or the headline number is a lie:

| regime | case | v0-d10 | v4-pruned | ratio |
| --- | --- | --- | --- | --- |
| sparse, small settled set | `tiny-r` (30 stones) | 1241.60ms | 1.40ms | 890x |
| typical midgame | `grown-24` | 2749.80ms | 2.30ms | 1200x |
| dense, large settled set | `grown-40` | 2534.20ms | 10.30ms | 246x |
| board fully settled (L empty) | `hex-packed` (68) | 41.50ms | 0.10ms | 415x |
| **L nonempty but minuscule** | **`hex-gap` (67)** | **429.30ms** | **76.60ms** | **5.6x** |

`hex-gap` is the adversary: packing with a single interior stone removed. No
empty-L fast path applies, `T` stays board-scale on nearly every ray, so the
distance pruning never fires and each ray scans every stone. At the fine
tolerance that case is only 1.8x faster than the quadtree. It is still correct
and still a win, but the honest range is **1.8x to 1200x**, not a flat 166x.

Before the empty-L fast path, `hex-packed` was 285ms — 7x *slower* than the
baseline. Dense packing was simply missing from the first corpus.

## What each step exploited

**v1 — the region is star-shaped.** `Settled = union over stones of
`R_s = { x : dist(x,L) >= |x-s| }`, with no Voronoi clipping, because a
non-nearest stone only makes the test harder (`|x-s| >= |x-s(x)|`). Along any ray
from `s`, `h(t) = dist(x(t),L) - t` is non-increasing (both terms 1-Lipschitz)
and `h(0) = dist(s,L) >= 2r > 0`. So each `R_s` has a single-valued radial
boundary `T(theta)`, and `T >= r` always. One monotone root find per ray replaces
the entire quadtree. 6.7x.

**v3 — the root is closed form.** `h` is the lower envelope of `h_f` over the
boundary features of `L`, each non-increasing, so `T = min_f T_f`. By A6 the
features are only three kinds, and each root is closed form — the `t^2` terms
cancel in the circle case, leaving a linear solve:

```text
exclusion circle q   |x(t)-q| = dia -/+ t   t = (dia^2 - |s-q|^2) / (2(u.(s-q) +/- dia))
inset edge X         |X - x(t).x| = t       t = (X - s.x) / (1 +/- u.x)
vertex v of L        |v - x(t)| = t         t = |v-s|^2 / (2 u.(v-s))
```

For `q = s` the first yields `t = dia/2 = r` exactly — the open-space disk,
recovered analytically. Each candidate is accepted only once its realising point
is verified legal, which trims features to the true boundary of `L` without ever
constructing that boundary. Angles are then subdivided until the chord is within
tolerance of the curve. 129x.

**v4 — exact distance pruning.** `|c-s| <= |c-x(t)| + |x(t)-s| = 2t`, so every
candidate obeys `t >= |c-s|/2`. Visiting candidates in increasing distance from
the stone lets the scan stop as soon as `|c-s|/2 >= best`. No reach heuristic, no
spatial index, no approximation. The stone's own circle sorts first and yields
`t = r` immediately whenever the radial escape is open, so in open play the scan
ends after a few candidates regardless of board size. 166x.

## Two corrections found by the bench

`T(theta)` is genuinely **infinite** for rays leaving the board: `dist(.,L)` then
grows as fast as `t` and `h` never crosses zero. That is not an empty legal set.
An early candidate inferred "L is empty" and painted the whole board — caught
immediately as false positives equal to `1 - settled fraction`. Capping `T` at
the ray's board-exit distance plus a margin fixes it and makes the fully sealed
board (`L` empty, every ray infinite) fall out for free.

Uniform angular sampling cannot follow a radial function that jumps from `~r` to
off-board between neighbours; the chord cuts across the board. Adaptive
flattening against the true curve removes those errors.

## Not yet exploited

A8 proves the settled set only **grows** when a stone is added, and A7 gives the
increment exactly: `L` loses precisely `B_open(q,2r)`. A move preview therefore
never needs a full recompute, only the growth. That is the remaining win for the
hover-preview use case, on top of the 166x here.

Also unexploited: the boundary pieces are circular arcs, parabolas and bisector
lines, so they could be emitted as exact SVG arc and quadratic segments instead
of flattened polylines.

## Files

- `run.html` / `run.ps1` — runner; `?candidates=`, `?cases=`, `?side=`, `?budget=`
- `harness/corpus.js` — ten deterministic cases, seeded, including degenerate ones
- `harness/oracle.js` — exact predicate, Path2D scoring, Lipschitz error bound
- `harness/bench.js` — timing, reporting, candidate registry
- `candidates/` — the ladder above
- `debug.html?case=corner` — per-stone `T(theta)` dump against a bisection reference
