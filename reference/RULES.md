# Voronoi Go Rules

This document defines the game. Mathematical consequences and implementation
lemmas belong in `AXIOMS.md`, not here.

## 1. Board and stones

- The board is the closed unit square `D = [0,1] x [0,1]`.
- A game has a fixed stone radius `r`, with `0 < r < 1/2`.
- Each stone has a color and a center in the inset square
  `C = [r,1-r] x [r,1-r]`.
- Distinct stone centers must be at least `2r` apart. Touching at exactly `2r`
  is legal.
- Changing the radius or loading a diagram is a setup operation, not a move.
  A playable diagram must satisfy the center and separation requirements.

## 2. Regions, groups, and score

For a position `S`, the cell of stone `s` is its closed Voronoi cell on the
board:

```text
V_s(S) = { x in D : ||x-s|| <= ||x-t|| for every t in S }.
```

- Distance ties form cell boundaries and have zero area. They do not affect
  score.
- Two stones are adjacent when their cells share a boundary segment of positive
  length. Contact at only one point is not adjacency.
- A group is a connected component of same-color stones under that adjacency.
- A group's region is the union of its stones' cells.
- A color's score is the total area of its stones' cells.

## 3. Legal placement centers

The legal-placement set of position `S` is

```text
L(S) = { p in C : ||p-s|| >= 2r for every s in S }.
```

Legality depends only on the center, radius, board, and existing centers. The
color of a possible future stone does not affect whether its center is legal.

## 4. Interaction and settlement

Let `d_S(x) = min_{s in S} ||x-s||`. A legal center `p` challenges board point
`x` when

```text
||x-p|| < d_S(x).
```

The inequality is strict. A placement that only ties an existing owner changes
no positive area.

- A board point is settled when no center in `L(S)` challenges it.
- A group is settled when every point in its region is settled.
- If `L(S)` is empty, every group is settled.

Settlement is a property of a position. Capture and self-capture remove settled
groups during move resolution.

## 5. Move resolution

On a player's turn, the player may place a stone at a center in `L(S)` or pass.
A placement is resolved as one transaction:

1. Add the new stone provisionally.
2. In that provisional position, find every settled opponent group.
3. Remove all of those opponent groups simultaneously.
4. Recompute cells, groups, legal placements, and settlement.
5. Find and simultaneously remove every settled group belonging to the mover.
   This is self-capture and is legal.
6. Recompute and commit the position, then give the turn to the opponent.

Self-capture is global: it can remove the new stone's group, a disconnected
friendly group, or both. The new stone can remove the last useful placement
center for a disconnected group without taking area from that group. The
reference implementation warns when a move removes one or more friendly stones.

## 6. Passing and repetition

- Passing changes no stones and gives the turn to the opponent.
- A placement resets the consecutive-pass count.
- Two consecutive passes end the game. The current Voronoi area totals are the
  final scores; the larger score wins, and equal scores are a tie.
- After the game ends, no further move or pass is allowed. Undo or Clear may be
  used to return the reference implementation to an active position.
- The reference rules currently impose no ko or repetition restriction.
