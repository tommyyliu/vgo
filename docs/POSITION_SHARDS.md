# Store positions, not pictures

## Why

A replay shard currently stores rendered pixels: 6144 samples x 10 channels x
128 x 128 floats, 6.04 GB, of which 4.03 GB is the raster. That makes the
rasterization a property of the *data* rather than of the *experiment*, and
everything downstream inherits the consequences.

Comparing two rasterizations tonight cost two full generation runs, a seed-
matching exercise, and two bugs -- `vgo.channels mismatch: expected 3, got
Some(10)` and `sample dimensions do not match the replay header` -- both of
which were the shard's raster and the model's raster being conflated. Neither
failure is possible if a shard stores the position.

The comparison it produced also argues for making this cheap. On 3058 paired
positions with byte-identical targets, the RGB model reached policy_kl 1.207
against semantic's 4.484 and top-1 0.160 against 0.070, then **lost** the
head-to-head 0.271 (13-35, 95% CI [0.166, 0.410]). Validation loss is not a
reliable proxy for playing strength here, so the honest way to evaluate a
representation is to train it and play it -- which means the cost of trying one
has to be low.

## What changes

A shard record stores the `Position` instead of the raster:

    radius: f64
    to_move: u8
    consecutive_passes: u32
    phase: u8
    stone_count: u32
    stones: [(f64 x, f64 y, u8 color); stone_count]

~522 bytes for a 30-stone position against 655,360 bytes of semantic raster:
**1255x smaller**. A 6.04 GB shard becomes roughly 5 MB of state plus its
policy targets, which then dominate -- see the sparse-policy note below.

Rasterization moves to load time, selected by a training flag rather than fixed
at generation.

## Rasterizing at training time is free

Measured on this box, `rasterize_any_into` at 128x128 with 30 stones:

| layout | per position | 6144 samples, 1 thread |
|---|---|---|
| semantic | 0.405 ms | 2.49 s |
| rgb | 0.346 ms | 2.13 s |

A five-shard window is 30720 positions: **0.8 s per epoch across 16 threads,
about 2% of a 40 s epoch.** No cache and no incremental rendering are needed;
re-rendering every epoch is cheaper than the machinery to avoid it.

This corrects an assumption worth recording: the earlier note that
rasterization was "66% of self-play CPU time" measured it per *MCTS node*,
where it runs hundreds of times per move. Per training sample it runs once.

## What this enables

- **Rasterization becomes an experiment variable.** Semantic, RGB, or anything
  later, from the same shard, with pairing exact by construction rather than
  verified after the fact.
- **Resolution becomes a training-time knob.** A shard generated once can train
  models at 96, 128, or 256 without regenerating self-play.
- **Augmentation can act on positions.** Dihedral symmetry currently reindexes
  pixel grids; applied to stone coordinates it is exact and cheaper.
- **Most of the storage housekeeping becomes unnecessary.** The compression
  script, shard retirement, and the 2.0 GB engine cache pruning all exist
  because shards are large.

## Design notes

**Rasterization needs to be callable from Python.** The learner is Python and
the rasterizer is Rust. Options, in order of preference:

1. A small PyO3 extension exposing `rasterize_any_into` over a batch of
   positions into a caller-provided array. Keeps one implementation, which
   matters because generation and training must agree exactly.
2. A Rust helper binary that converts a position shard to a raster shard,
   invoked once per training run. Simpler to build, but reintroduces a
   materialized raster file.

Option 1 is worth the build cost: two implementations of the same rasterizer
that must agree bit-for-bit is exactly the kind of thing that drifts.

**The policy arrays become the whole cost.** Once states are positions, a shard
is dominated by five policy-shaped arrays of `policy_resolution^2 + 1` float32
slots holding ~24 nonzero values each -- 0.26% density. Sparse (index, value)
pairs would cut what remains by roughly another 500x for those arrays. Worth
doing in the same format revision rather than a second migration.

**Schema version.** This is REPLAY_VERSION 4. Version 3 shards store rasters
and cannot be re-rendered, so the loader must keep reading them: `ddrnet-pipe`
has 53 generations of history that would otherwise become unreadable.

**What stays identical.** Policy targets, visits, beta, proposal counts, value,
selected action, game id, ply, and seed are all functions of board state and
search, not of rendering. Only the state field changes representation.

## Order of work

1. ~~Serialize `Position` into the record; bump to REPLAY_VERSION 4, keeping the
   v3 read path.~~ Written (`c1fecab`). The v3 *read* path is still owed.
2. ~~Sparse policy encoding, in the same revision.~~ Written (`c1fecab`).
3. Python loader for v4, keeping v3 readable -- `ddrnet-pipe` has 53
   generations in the old format.
4. PyO3 rasterizer, with a test asserting it matches the Rust output exactly.
5. `--raster-kind` moves from `vgo-generate-demo` to the learner.

Nothing can train on a v4 shard until 3 and 4 exist. Worth writing first: a
round-trip test asserting that a v4 shard rasterized at load time is
byte-identical to what v3 would have written for the same positions. A silent
mismatch there corrupts training rather than failing loudly.

The RGB-versus-semantic question then reruns as two training runs against one
shard, with no regeneration -- which is the point.

## Verified equivalent to v3

Generation is deterministic given a seed, so the same seed and model under the
v3 and v4 writers produce the same games. Comparing the two through
`load_dataset` -- v3 reading stored rasters, v4 rendering at load time -- is
therefore a direct test of whether the format changed what training sees.

Run on 192 positions from seed 31337 with ddrnet-pipe's update-52 as the
behaviour model, aligned on (game, ply):

    states             IDENTICAL
    policies           IDENTICAL
    policy_masks       IDENTICAL
    visits             IDENTICAL
    betas              IDENTICAL
    proposal_counts    IDENTICAL
    values             IDENTICAL
    selected_actions   IDENTICAL

The shards were 188.8 MB and 0.7 MB respectively.

This is the check worth repeating after any change to the rasterizer or the
record layout. Byte-exactness of the rasterizer alone is not sufficient: it
would not catch a sparse-policy scatter that dropped a cell, an off-by-one in
the record offsets, or a field written in the wrong order. Comparing the loaded
tensors covers the whole read path at once.

## Speculative: store moves, and paint incrementally

Two further ideas, neither urgent. The first is a clean simplification; the
second is an optimization that current measurements do not justify.

### Moves instead of positions

A v4 record spends 2,194 bytes on the position, nearly all of it padding to
`STONE_CAPACITY`. A move is ~17 bytes, and a position is reconstructible by
replaying its game's moves. Storing the move list once per game and a move
index per sample would take per-sample state to roughly zero: a 6144-sample
shard holds ~100 games of ~60 moves, so the whole history is ~100 KB against
13 MB of padded positions.

It also removes the capacity ceiling, which is the least comfortable part of
v4. `STONE_CAPACITY = 128` is measured-safe today -- 88 stones was the longest
game observed -- but it is a hard failure the first time a run uses longer
games, and the bound exists only to keep records fixed-size for the
memory-mapped loader.

The logical end of this direction is one record per *game* rather than per
position, with training samples produced by replaying. That is a larger change
to how the loader presents data, and worth treating separately.

### Incremental painting

Placing a stone changes a small disc, so re-rendering a whole 128x128 raster
looks wasteful. It is not, for two reasons.

**The dirty region is not local.** Most semantic channels depend on more than
the neighbourhood of the move:

- `current_voronoi` / `opponent_voronoi`: a new stone can flip ownership of
  regions far from it.
- `current_distance` / `opponent_distance`: nearest-stone distance changes
  anywhere the new stone becomes nearest.
- `voronoi_ridge`: depends on the nearest *two* stones, so it shifts wherever
  the second-nearest changes.
- Captures remove stones, which can change ownership across the board.

The region to repaint is the union of Voronoi cells that changed, which can be
most of the board. Computing it correctly is harder than the re-render it
replaces.

**Re-rendering is already cheap where it matters.** Measured at 0.405 ms per
position, a five-shard window costs 0.8 s per epoch across 16 threads -- about
2% of a 40 s epoch. There is nothing here to win.

The place it could pay is self-play, where rasterization runs per MCTS node
rather than per training sample, and was measured at 66% of self-play CPU. But
the same non-locality applies there, and the positions differing by one move
are the leaves, which is where the cost already concentrates.

Revisit only if profiling shows self-play rasterization is the bottleneck, and
then measure the dirty-region computation against the full re-render before
committing to it.
