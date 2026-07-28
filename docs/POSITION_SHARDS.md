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

1. Serialize `Position` into the record; bump to REPLAY_VERSION 4, keeping the
   v3 read path.
2. PyO3 rasterizer, with a test asserting it matches the Rust output exactly.
3. `--raster-kind` moves from `vgo-generate-demo` to the learner.
4. Sparse policy encoding, in the same revision.

The RGB-versus-semantic question then reruns as two training runs against one
shard, with no regeneration -- which is the point.
