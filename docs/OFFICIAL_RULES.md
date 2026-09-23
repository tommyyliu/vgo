# Compatibility with the current website rules

This note documents `voronoigo.com`'s current behavior, exposed as
`Ruleset::Official` in `crates/vgo-core`. Here, *Official* is a compatibility
label: it means website-compatible, not canonical or preferred. This
repository's rules remain the game defined in [`reference/RULES.md`](../reference/RULES.md).

The purpose of this note is to record what differs, what our compatibility
implementation does about it, and what is still weaker here than in the
website reference.

The reference is [`csun/voronoi-go-rs`](https://github.com/csun/voronoi-go-rs) —
a Rust port of the original TypeScript client, verified against fixtures the
original generated. We do not use it at runtime; see *The oracle* below.

## What actually differs

Almost nothing, which is the point. The board, the Voronoi partition, area
scoring, and two passes ending the game are identical. Two things differ, both
about capture.

### 1. The capture condition

Both rules ask "can this group still be interfered with?" and mean different
things by it.

    ours       alive  <=>  a future stone could take area from the group
                           exists x in region, p in L : |x - p| < d_S(x)

    official   alive  <=>  a future stone could be placed *touching* the group
                           exists p in L : dist(p, region) <= r

where `L` is the legal set of stone centres, `d_S(x)` the distance from `x` to
the nearest stone, and `r` the stone radius.

**The website rule is more aggressive**, and the containment is worth deriving
because it is not obvious in either direction.
If a legal centre `p` lies strictly within `r` of a point `x` of the region,
then

    d_S(x) >= d_S(p) - |x - p| >= 2r - |x - p| > r > |x - p|

using `d_S(p) >= 2r`, which holds because `p` is a legal centre and so at least
one diameter from every stone. So `p` challenges `x`, and our rule calls the
group alive too. At exact tangency, `|x - p| = r`, the argument gives only
`d_S(x) >= r`. If equality holds, choose an owning stone `s`. Legality and
the triangle inequality force `|p - s| = 2r` and `x = (p + s) / 2`.
Any other stone tied at `x` would also have to be opposite `p` on the same
radius-`r` circle, hence coincide with `s`. Thus `s` uniquely owns `x`.
The midpoint is inside the board, so a small step from `x` toward `p` remains
inside `s`'s cell and is strictly closer to `p`. This witnesses the group's
survival even though `x` itself is tied.

The converse fails: `p` can sit `3r` from a large cell and still be strictly
closer to a far corner of it than the owning stone is.
That is the "a group lives while it can still connect out" case, and it exists
only here.

So: **every group the website rules keep alive, ours keeps alive, while ours
keeps some groups that theirs captures.** This is a group-level statement:
the pointwise settled and dead-zone sets still differ at equality boundaries.

The comments in `runs/raster-ab.sh` (on the `archive/pre-prune` branch) record sampled
coverage of 47.9% for the website dead zone and 44.1% for our settled region.
The exact corpus and measurement command were not recorded there, so these
are historical observations, not a reproducible benchmark or a measure of
how often the two rules disagree about captures.

The predicates differ in the rules themselves, independently of numerical
tolerances. Their practical effect needs capture and gameplay comparisons;
pointwise area coverage alone does not establish how small that effect is.

### 2. Self-capture

Ours is legal and global: a move may remove the mover's own groups, including
the one it just joined. A placement that leaves the board *exactly* as it was is
a no-op, and counts as a pass — without which it is a better stall than passing,
since two passes end the game and two no-op suicides end nothing.

The official rules reject a move that would take **only** the mover's own
stones. A move that captures enemy stones is legal even when it also kills
friendlies; the enemy is resolved first and dies first, exactly as here. With
self-capture-only moves illegal there is no no-op placement, so the pass rule
never fires and the whole even-trade question does not arise.

**Refused moves spend no network inference.** `Action::try_apply` resolves a
candidate before `Node::new` evaluates it, so a forbidden self-capture is
dropped at expansion. The geometry work needed to discover the refusal is
still paid. Treating one as a pass instead would be wrong: it would invent
a move the real client rejects, and a bot trained on it would propose moves the
site refuses.

## Testing only cell vertices is exact, for our rule

`alive_groups_of` decides a group by walking its cells' polygon **vertices**,
which looks like an approximation of a rule stated over the whole region. It is
not one, and the reason is two lines once the distances are dropped.

A point `x` is taken by a legal point `p` when `|x - p| < |x - s|` -- when it is
closer to `p` than to our stone. That is one side of the perpendicular bisector
of `s` and `p`: **a half-plane**. So `x` is *settled* when it is on our stone's
side of every such bisector, and

    settled region of s  =  intersection over p in L of { x : |x - s| <= |x - p| }

From there, two routes, and the shorter one needs less.

**Via convexity.** An intersection of half-planes is convex; a Voronoi cell
clipped to a rectangular board is convex too, so it is the hull of its corners;
and a convex set containing every corner contains their hull.

**Via linearity, which does not require convexity.** A half-plane is
`{ f > c }` for a linear `f`, and a linear functional on a polygon attains its
maximum at a vertex. So a half-plane that meets a bounded polygon at all contains
one of its vertices. If any point of the cell is taken by `p`, then `p`'s
half-plane meets the cell, so it contains a corner, so that corner is taken.

The second argument also works for nonconvex polygons. It still requires
polygonal boundaries: on a curved boundary, a linear functional can attain
its maximum away from every corner.

Either way, a cell whose corners are all settled is settled entirely, and
checking corners misses nothing.

**Groups, not just cells.** The union of a connected group's cells can be badly
non-convex, which is why `alive_groups_of` walks cells rather than group regions.
It does not need to do better: any point of the union lies in one of the cells,
and that cell is where the argument above finds its corner.

That also strengthens `AXIOMS.md`'s A16, which records each `R_s` as star-shaped
about `s`. True, but weaker than the fact: it is convex. The radial solve in
`settled.rs` only needs star-shapedness, so nothing there is wrong -- but a
future reader deriving bounds from A16 is leaving something on the table.

`examples/settled_vertex_gap.rs` checks that the code implements this, over
136,122 cells against a 256-sample sweep of every edge. It is a test of the
implementation, not an argument for the theorem.

**None of it transfers to `Ruleset::Official`.** Its question is whether the
region comes within `r` of a legal point -- a distance band with curved edges,
not a half-plane. The same convexity argument does not apply, and a
cell edge really can dip into the band with both its corners outside. That is
why its extra edge tests are necessary, and why the case ours still misses --
the closest approach between the interior of an edge and a smooth arc of `L`'s
boundary -- is real there and vacuous here. The reference measures it exactly
with `AliveZone::closest_distance`.

## Where the reference implementation is better

Recorded for a future revisit rather than as a to-do. None of it is urgent.

- **Exact segment-to-set distance.** `AliveZone::closest_distance(edge)` measures
  a line segment against the alive zone's real outline — arcs included — which is
  exactly the case our edge tests miss under `Official`. Our own rule needs
  nothing here; it is proved exact above.
- **Forced eyes as exact points.** The zone tracks isolated legal points
  explicitly, so a single remaining placement in the middle of a group's
  territory keeps it alive. Ours would find such a point only if it happens to be
  a legal-set vertex — usually true, since a bounded piece of `L` is cornered
  where its constraints meet, but not guaranteed.
- **Incremental structure.** The alive zone adds and removes one stone's disc
  without rebuilding, and `undo_move` restores the previous state bit for bit,
  down to the segment list. We recompute the legal set per position. For a search
  that plays and unplays millions of moves that is a genuine architectural
  advantage, and it is the strongest argument for that design.
- **A tighter tolerance, relative to a stone.** The reference's `EPSILON` is
  `1e-7` at stone radius `1.0`; ours is `1e-7` at radius `0.0557`, so ours is
  about 18x looser measured in stone radii. Our coordinates live in `[0,1]` and
  barely use the exponent range, so scaling the base board up would buy that
  back. Nothing has been traced to this, and f64 has room either way.
- **Degeneracy handling.** Three dead-zone circles through one point, and
  sub-ulp arcs between crossings that resolve a couple of parts in 10^15 apart,
  are handled deliberately and pinned by fixtures.

## The oracle

The reference is a **test dependency, not a runtime one**. It is used at exactly
two moments:

1. When `Ruleset::Official` changes, to measure how often our capture verdict
   disagrees with the reference over a corpus of real positions.
2. Before a run that will train on those rules, as a gate.

Both go through `voronoi-go-engine`, its JSON-over-stdin binary, with positions
replayed move by move — it has no set-position command. Coordinates scale by the
board size: theirs is an 18-unit board with radius 1, ours the unit square with
radius `1/18`, and those are the same game.

Nothing in the shipped binaries links it, so its AGPL licensing does not reach
the client bundle or the training pipeline. That is a deliberate boundary and
should stay one unless the licence question is settled.
