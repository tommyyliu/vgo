# Human 002: Gameplay Variants Worth Trying

- Status: Open
- Priority: Exploratory
- Owner: Human + Developer

## Purpose

A running list of rule variants that might make the game better to play. This is
a parking lot, not a plan. Nothing here is committed, and an idea can sit unread
for a long time without going stale.

Each entry records what the variant changes, roughly what it would cost to build,
and which axioms it threatens — because in this codebase the second and third
questions are usually decided together. An idea that keeps the geometry exact is
cheap; an idea that forces sampling into the rules is not, however small it looks
in a rulebook.

Add ideas freely. The judgement of whether a variant is *fun* belongs to a human
and no amount of analysis substitutes for playing it.

## Cost vocabulary

- **Free** — no new mathematics; existing machinery is already parameterised for it.
- **Bounded** — new derivations, but the exactness commitment in
  [`ARCHITECTURE.md`](../../reference/ARCHITECTURE.md) survives.
- **Structural** — breaks an axiom the engine is built on, or forces approximation
  into the rules rather than the rendering.

---

## 1. Terrain

Three genuinely different games hide behind one word. Analysed 2026-07-22.

**1a. No-build zones.** Polygonal regions where a stone may not be placed.
Influence flows over them normally.

- Cost: **free**. This changes only the legal-placement set `L`, and A15/A16/A17
  are already parameterised by `L`. The board edge is *already* a no-build
  boundary: A17's inset-edge feature generalises to any polygon edge and its
  corner feature to any polygon vertex.
- Gives asymmetric openings and natural walls. Does not give the feeling of a
  mountain blocking sight.

**1b. Impassable obstacles.** Influence measured by geodesic distance around
polygonal holes. This is the "there's a mountain, go around it" version.

- Cost: **bounded, but the largest bounded item on this list.** Exactly
  computable — geodesics bend only at obstacle vertices and bisectors are
  hyperbolic arcs. A1 survives (`d_geo >= d_euc`), A15 survives verbatim, A16
  survives in geodesic form because `d(.,L)` is 1-Lipschitz in any metric.
  A2 dies: cells stop being convex polygons. **A5 needs a new proof**, and that
  is the real risk — capture is currently finite *because* cells are polygons.
- New wrinkle: a cut locus forms behind an obstacle where geodesics arrive from
  two sides, so a settled region stops being a simple closed curve traced by
  initial direction. The per-stone sweep assumes it is.

**1c. Graded cost.** Regions that cost more per unit to cross.

- Cost: **structural, and provably so.** This is the weighted region problem;
  shortest-path length is not computable from the rationals by finitely many
  arithmetic operations and `k`-th roots, with as few as two distinct weights.
  Capture would become an epsilon-approximation with Steiner points, which is
  exactly the property `ARCHITECTURE.md` refuses to give up.
- Recommendation: do not pursue against the exact engine. Note that the raster
  and self-play path could support it today, since multi-source fast marching on
  a grid is routine; that only matters if the reference stops being the oracle.

## 2. Per-stone radius

Let a player choose a larger or smaller stone, perhaps at a cost in stones.

- Cost: **bounded**. Under a power (Laguerre) distance `||x-s||^2 - r_s^2` the
  cells stay convex polygons, so A2 and the exact area machinery survive. Under
  plain nearest-centre they also stay polygonal but A1 can fail, letting a large
  stone lose the ground under itself.
- Already flagged in the numerical policy: A16 and A17 assume one shared radius,
  so the closed forms need rederiving.

## 3. Only settled area scores

Score the settled region rather than the whole Voronoi area, so territory counts
only once no legal move can take it back.

- Cost: **free** to try — the settled region is already computed exactly, and
  after A15-A17 it is cheap enough to evaluate continuously.
- Likely the deepest change on this list for the smallest implementation. It
  turns the settled set from a diagnostic into the object of play, and makes the
  endgame about closing space rather than claiming it.

## 4. Board shape

Non-square boards: a disk, a hexagon, a torus.

- Cost: **bounded**, and mostly concentrated in one place. The inset region and
  A17's edge feature family are the only things that care. A disk board replaces
  the inset-edge feature with a circular one; a torus removes edges entirely and
  with them every corner advantage.
- A torus also breaks board convexity, which A16 uses to truncate unbounded
  directions.

## 5. Stone economy

A fixed budget of stones per player, so passing becomes a resource decision and
the two-pass ending competes with running out.

- Cost: **free**. Position state and the ending rule only.

## 6. Simultaneous placement

Both players commit a move; collisions resolve by some rule.

- Cost: **bounded**, and entirely in the transaction layer rather than the
  geometry. Needs a collision rule for centres closer than `2r`.

---

## Related

- [006: Repetition and ko policy](../006-repetition-and-ko.md) and
  [Human 001](001-find-ko-scenario.md) — an open rules question, not a variant.
