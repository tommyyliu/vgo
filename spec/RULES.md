# Voronoi Go in plain words

The precise statement is [`FORMAL.md`](FORMAL.md), which wins if the two differ.

- **Board.** A square. Stones are equal discs and may go anywhere that fits: there
  are no grid points. Smaller stones make a bigger game.
- **Placing.** A stone must lie fully on the board and may touch other stones but
  not overlap them. Colour doesn't matter for where a stone fits.
- **Territory.** Every point of the board belongs to the nearest stone. A stone's
  area is its Voronoi cell.
- **Groups.** Same-coloured stones whose cells share an edge are connected.
- **Life.** A point is settled when no future stone, wherever it legally fits, could
  be strictly closer to it than the stone that owns it. A group dies when all of
  its area is settled.
- **A move.** Place a stone or pass. After placing, settled enemy groups are removed
  together. Then settled groups of your own are removed, which is legal and can
  happen anywhere on the board.
- **No-op moves.** A placement that changes nothing, because the new stone alone is
  removed, counts as a pass.
- **Ending.** Two passes in a row end the game. The score is each colour's total
  area, with komi deducted from Black's lead. There is no ko rule.

`Ruleset::Official` follows voronoigo.com instead, where a group lives while a stone
could still touch its territory and self-capture is restricted. See
[`spec/OFFICIAL.md`](OFFICIAL.md).
