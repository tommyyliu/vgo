# Rasterization — two open optimisations

Written 2026-08-26, after a round of work that took the `compact-radius` raster
at 38 units and 240 stones from 6.95 ms to 1.60 ms. Two things are left that I
looked at and did not build. This is what I know about each, including what I
tried and what failed, so nobody spends the afternoon I already spent.

## How to work on this

```
cargo run --release -p vgo-raster --bin vgo-raster-bench            # timings
cargo run --release -p vgo-raster --bin vgo-raster-bench -- --verify  # one gate
cargo test --release -p vgo-raster                                    # the other
```

The benchmark is standalone — no GPU, no model, no ONNX Runtime.

**Both gates matter and neither covers everything.** `--verify` compares the
packed writer against the dense one bit for bit, so it catches a change to the
sweep, the ridge or the packing. It does *not* cover `settled`: both writers
call the same `settled_for_raster`, so a change there passes trivially. What
guards settled is `edt::tests::bounded_distance_agrees_with_the_definition`,
which pins it against the mathematical definition. Run both.

Nothing here may change the raster's output. It is a network input: a
difference that looks like rounding is a model quietly wrong about positions,
and no test downstream of this would flag it.

**Measure across stone counts, not at one point.** Half of generated games are
38-unit boards playing to 312 plies, so the positions that dominate carry well
over a hundred stones. Benchmarking at 28 on an 18-unit board is a mistake that
misled several estimates during this work, mine included. Every hot path in the
raster now dispatches on stone count, so a change can improve one side of a
threshold and regress the other.

Current state, 256x256, median of 40, idle machine:

```
  units  stones |   legal transform  settled | sweep+out   packed    dense
     38      28 |   0.011     0.299    0.576 |     0.445    1.021    0.915
     38     120 |   0.042     0.311    0.848 |     0.580    1.428    1.767
     38     240 |   0.096     0.357    1.108 |     0.489    1.597    2.861
```

---

## 1. The second-nearest distance, from Delaunay adjacency

### What it is for

Exactly one channel: `voronoi_ridge`.

```rust
ridge = clamp(1 - (d2 - d1) / r, 0, 1)
```

The two stone-disc planes need only a threshold against `r`, and `settled`
comes from the distance transform. So the whole second-nearest computation, and
the grid search in `packed::sweep_row_chunked` that produces it, exists to feed
one plane of seven.

### The current cost

`sweep+out` above: 0.45–0.58 ms, roughly a third of the raster. It is already
much better than it was — a grid search replaced a flat O(pixels × stones) scan
— but it is still a *search*, proportional to how many candidate stones survive
the bound.

### The idea

The second-nearest site to any point is always a **Delaunay neighbour of its
nearest site**. It cannot be some distant stone. Planar Voronoi diagrams average
six neighbours per site, so:

1. Find each pixel's nearest site *label* with a labelled distance transform —
   O(pixels), independent of stone count. This is the same machinery
   `edt::squared_distance_transform` already runs, carrying a site index
   alongside the distance.
2. `d2 = min over the neighbours of that site`. About six candidates per pixel
   instead of however many survive the current bound.

`vgo_core::voronoi::compute` already returns `Geometry { adjacency: Vec<Vec<usize>>, .. }`,
so the adjacency exists and does not need building.

### The correction that comes free

A grid transform gives the nearest *cell*, and stones sit at continuous
positions, so the label can be wrong right at a boundary. That fixes itself:
take the minimum over `{candidate} ∪ neighbours(candidate)` and you get both
`d1` and `d2` exactly, because the true nearest is either the candidate or one
of its neighbours. The correction and the `d2` lookup are the same operation.

### Why I did not build it

It needs a labelled variant of the transform (new code in `edt.rs`), the
adjacency threaded into the raster path, and the boundary argument to be
exactly right. The version that shipped instead is ~200 lines and bit-identical
to what it replaced. This is a bigger change with a subtler correctness
argument, and correctness here is not negotiable.

### What I would watch for

- `voronoi::compute` is not free. If it costs more than the search it replaces,
  the whole thing is pointless — measure it before building the rest. It is
  currently ~0.7% of generator CPU, but it is called once per position, and
  this would call it once per raster, which is not the same thing.
- Degenerate adjacency. `GeometryDiagnostics` counts `unclassified_edges` and
  `degenerate_edges`; if those are ever non-zero the neighbour list may be
  incomplete, and an incomplete neighbour list silently gives the wrong `d2`.
- The empty board and the one-stone board, where `d2` is infinite. That case
  already produced one bug: `inf - inf` is NaN, and `clamp` propagates NaN
  where `max` does not. See `packed::ridge_at`.

### There is a stronger version

`ridge` is zero wherever `d2 - d1 >= r`, which is most of the board — it is a
band around Voronoi edges. Rasterising only those bands makes the work
proportional to band area rather than board area. Asymptotically the best of
the three, and the fiddliest to get right at the boundary, which is exactly
where a mistake would be invisible.

---

## 2. The fp16 store in the output loop, which costs more than it should

### The observation

The packed writer's per-pixel output loop is slower than the dense writer's,
while writing **twelve times less memory**. At the time I measured it, with
`settled` subtracted from both:

```
  dense writer     0.30 ms     four f32 planes, 1792 KB
  packed writer    0.46 ms     one f16 plane and three bit planes, 152 KB
```

### What it is not

Each of these was measured by replacing one piece with a `black_box` of the
same inputs, so the rest of the function is unchanged:

- **Not the bit packing.** Removing it leaves 0.33 ms — the dense writer's
  time. Packing three planes is free.
- **Not the fp16 conversion.** Replacing `f16::from_f32(x)` with
  `f16::from_bits(x.to_bits() as u16)` — same store, no conversion work — is
  also 0.50 ms. The conversion instruction is not the cost.
- **Not the eight-pixel grouping.** A flat ridge loop with no bit packing
  interleaved is 0.49 ms.
- **Not `f16c`.** The build has `target-cpu=native` and hardware conversion is
  compiled in; 65,536 conversions measure 0.009 ms in isolation.

### What is left

The narrow store itself. The dense writer streams four 4-byte planes; the
packed writer streams one 2-byte plane. The wider stores schedule better
despite moving seven times the bytes.

### Things tried that made it worse

- **Splitting the ridge into its own flat pass**, so the square roots run
  uninterrupted by the bit packing. Worse both before and after `ridge_at` lost
  its branch — 0.48 ms against 0.43 at 240 stones. The second walk over the row
  costs more than the longer vector runs return. There is a comment saying so
  at the call site; please do not re-litigate it without a measurement.
- **Hoisting the destination into a local slice** did help, 0.50 → 0.46, and is
  already in. Indexing `out.dense` while `out.bits` was separately borrowed
  stopped the compiler proving the pointer loop-invariant.

### What I would try next

- Look at the disassembly. This is the point where guessing stopped working for
  me; three plausible hypotheses were all wrong, and each cost a rebuild and a
  benchmark run.
- Whether the f64 → f32 → f16 chain is the problem rather than the store.
  `ridge_at` returns `f32` and the caller converts to `f16`; a direct f64 → f16
  might schedule differently.
- Whether writing the ridge plane as f32 and narrowing it in one pass at the
  end is faster than narrowing per pixel — that trades a second pass for wider
  stores, which is the opposite of the split that failed, and might land the
  other way.

### How much it is worth

The output loop is roughly a fifth of the raster, and the gap is about a third
of that, so perhaps 5–7% of rasterization. Smaller than the first item. Worth
doing mostly because *not understanding it* is uncomfortable — it is the one
measurement in this work that resisted explanation.
