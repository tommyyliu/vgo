# Playing the official rules

`voronoigo.com`'s rules, as `Ruleset::Official` in `crates/vgo-core`. This is
what differs, what our implementation does about it, and what is still weaker
here than in the reference.

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

**The official rule is strictly more aggressive**, and the containment is worth
deriving because it is not obvious in either direction. If a legal centre `p`
lies within `r` of a point `x` of the region, then

    d_S(x) >= d_S(p) - |x - p| >= 2r - r = r >= |x - p|

using `d_S(p) >= 2r`, which holds because `p` is a legal centre and so at least
one diameter from every stone. So `p` challenges `x`, and our rule calls the
group alive too. The converse fails: `p` can sit `3r` from a large cell and still
be strictly closer to a far corner of it than the owning stone is. That is the
"a group lives while it can still connect out" case, and it exists only here.

So: **every group the official rules keep alive, ours keeps alive. Ours keeps
some alive that theirs captures.** Measured as a field on real positions, the
official dead zone covers 47.9% of the board where our settled region covers
44.1%.

Practically the gap is small, because deliberate sacrifice is rarer here than in
Go — a group that cannot be reached usually also cannot be extended. But it is a
real difference in the rules and not a tolerance.

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

**This costs the search nothing.** `Action::apply` resolves a candidate into a
position before `Node::new` evaluates it, so a self-capture is already known by
the time it would reach the network — the candidate is dropped at expansion, no
inference wasted. Treating one as a pass instead would be wrong: it would invent
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

An intersection of half-planes is **convex**. A Voronoi cell clipped to a
rectangular board is convex too, so it is the convex hull of its corners -- and a
convex set containing every corner contains their hull. A cell whose corners are
all settled is therefore settled entirely, and checking corners misses nothing.

That also corrects `AXIOMS.md`'s A16, which records each `R_s` as star-shaped
about `s`. True, but weaker than the fact: it is convex. The radial solve in
`settled.rs` only needs star-shapedness, so nothing there is wrong -- but a
future reader deriving bounds from A16 is leaving something on the table.

`examples/settled_vertex_gap.rs` checks that the code implements this, over
136,122 cells against a 256-sample sweep of every edge. It is a test of the
implementation, not an argument for the theorem.

**None of it transfers to `Ruleset::Official`.** Its question is whether the
region comes within `r` of a legal point -- a distance band with curved edges,
not a half-plane. Nothing is convex, nothing is an intersection of slices, and a
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
  advantage, and it is the strongest argument for his design.
- **A tighter tolerance, relative to a stone.** His `EPSILON` is `1e-7` at stone
  radius `1.0`; ours is `1e-7` at radius `0.0557`, so ours is about 18x looser
  measured in stone radii. Our coordinates live in `[0,1]` and barely use the
  exponent range, so scaling the base board up would buy that back. Nothing has
  been traced to this, and f64 has room either way.
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
