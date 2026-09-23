# MCTS memory: measured first steps

Follow-up: [actual process RAM scaling](RAM_SCALING.md) measures the packed-mask
and trimmed-capacity implementation through 32 actors × 1,400 simulations.

## What changed

**Production:** `DensePolicy` and its `FineGrid` now share an immutable
`Arc<Vec<f32>>` instead of retaining two copies of the logits. The grid keeps
its own legal mask and snapped-placement overrides. Illegal logits need not be
overwritten because every sampling lookup is masked. The existing callback
constructor still creates an owned, masked grid for other policy providers.
No quantization, policy approximation, candidate-budget change, or certificate
cache was introduced into production search.

**Diagnostics:** `SteppedSearch::tree_memory()` inventories retained tree
payload by category and deduplicates shared allocations. Policies can optionally
report their heap allocations; unknown providers are explicitly counted as
unreported. The inventory includes node records, reserved child storage, live
stone payload, policy allocations, and sampling allocations. It excludes
allocator/reference-count headers, hidden excess stone-vector capacity, legacy
candidate-cache allocations, pending batches, driver state, and inference
buffers. It is not RSS or peak process memory. The diagnostic is opt-in and
does not run in the ordinary search loop.

**Laboratory only:** `iteration_lab::ReversiblePosition` retains one current
position and compact undo records containing previous metadata, removed stones
with their old indices, and whether the new stone survived. It restores the
original stone order and coordinate values without recomputing them. Failed
moves leave the stack unchanged. Passes, no-op self-capture, opponent capture,
global self-capture, and website-rules refusal are covered by tests.

This undo experiment still calls the normal rebuild transition in the forward
direction and allocates a reconstructed stone vector during undo. It does not
incrementally maintain Voronoi geometry, support points, or raster inputs, and
is not wired into MCTS. It demonstrates compact, exact *position history*, not
the cost of the eventual full-geometry undo log.

## Reproduce

```sh
cargo test -p vgo-core -p vgo-search -p vgo-raster --features vgo-core/iteration-lab --lib
cargo run --release -p vgo-raster --example tree_memory -- 7
cargo run --release -p vgo-core --features iteration-lab --example undo_lab -- 7
```

Run these sequentially. Results below were collected on 2026-09-11, AMD Ryzen 9
9950X, Rust 1.97.1, release profile. Both experiments alternate variant order
and report median times over seven runs. They use deterministic synthetic
policies/trajectories, not a neural model or recorded production searches.

## Tree results

[`tree-memory-2026-09-11.csv`](../diagnostics/tree-memory-2026-09-11.csv)
compares the shared implementation against a policy wrapper that reproduces the
old copying grid constructor. Both wrappers retain the same original logits.
Each paired run must match the entire `SearchResult`, including root action,
child ordering, priors, values, visits, proposal probabilities/multiplicities,
and search statistics. The experiment uses a 20-stone board, a 128×128 policy,
root exploration noise, and both single-leaf and eight-leaf batching.

| Simulations / batch | Retained nodes | Copied payload | Shared payload | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 64 / 1 | 65 | 6.57 MiB | 4.88 MiB | 25.7% |
| 64 / 8 | 62 | 6.29 MiB | 4.67 MiB | 25.8% |
| 256 / 1 | 257 | 25.64 MiB | 19.21 MiB | 25.1% |
| 256 / 8 | 254 | 25.46 MiB | 19.03 MiB | 25.3% |

The tree reduction is not 50%: many leaves hold a policy but have not built a
sampling grid yet. Shared payload is attributed to policy storage first;
`shared_bytes_avoided` counts duplicate ownership references to that payload,
including the trailing pass entry. Actual copied-versus-shared totals are the
appropriate measure of savings, not that ownership counter alone.

Search construction times fall from 46.435 to 44.930 ms at 256 simulations /
batch 1, and from 46.003 to 44.636 ms at batch 8. These roughly 3% differences
are secondary to the deterministic memory savings. Timings include synthetic
evaluation and tree construction, but exclude memory inventory, final result
assembly/destruction, and neural inference. Do not extrapolate them to GPU
search throughput.

### The priority revealed by measurement

At 256 simulations / batch 1, shared policy storage still accounts for 16.86 MB
of the 20.14 MB tracked total (about 84%). Node positions account for only
0.138 MB (under 1%) in this fixture. Removing node positions first would save
little here while adding replay work. Board-heavy workloads can differ, but
the evidence currently favors bounding policy/sampling storage before
replacing the tree's position representation.

## Active-path results

[`undo-memory-2026-09-11.csv`](../diagnostics/undo-memory-2026-09-11.csv)
compares retaining every ancestor position against compact undo records on
32-move trajectories starting from three iteration-lab played positions.
The current board and verification fixtures are excluded from both history
counts. Snapshot counts use reserved record storage plus live stone payload;
undo counts include reserved record and removed-stone capacity.

| Starting stones | Snapshot history | Undo history | Reduction |
| ---: | ---: | ---: | ---: |
| 10 | 9,960 B | 3,712 B | 62.7% |
| 37 | 37,080 B | 3,456 B | 90.7% |
| 100 | 90,048 B | 2,688 B | 97.0% |

Every restored ancestor is checked, not just the root. These figures describe
one active path, not total MCTS-tree savings. Captures can make individual undo
records large, and adding geometry/certificate deltas will increase history
storage. Forward computation still dominates this experiment; it establishes
no meaningful traversal speedup.

## Next engineering stages

1. **Bound policy/sampling caches.** Keep search statistics, existing candidates,
   priors, multiplicities, proposal counters, and RNG state in the tree. Treat
   dense policy and derived sampling data as cacheable. Eviction must not reset
   widening or sampling. Re-evaluation adds cost and may not be bitwise stable
   across inference backends; prototype and validate that tradeoff before
   enabling eviction. Model identity belongs in any evaluation-cache key.
2. **Extend worker-local reversible state.** Build stable stone/point IDs,
   incremental spatial indexes, and blocker/support deltas on the tested undo
   semantics. Keep certificates out of permanent node records. Retire obsolete
   candidate features instead of retaining every historical dormant point.
   Dropping a positive certificate triggers fallback, never capture.
3. **Add reversible geometry separately.** Restore saved topology/coordinates
   and areas exactly. Group splits, large captures, and negative-certificate
   invalidation remain explicit costs. A rebuild fallback is required.
4. **Only then test compact tree nodes and bounded checkpoints.** Existing
   expanded-edge traversal currently follows pointers without replaying board
   transitions. A compact-node implementation must benchmark the extra replay
   work at realistic depths and batch sizes, not just memory at one node.
5. **Measure the full budget.** Tree records + bounded caches + worker states +
   active undo paths + pending inference inputs. A batched leaf should retain
   its evaluation input and backup path, not a complete geometry engine.

The production tree still owns positions; no checkpoint cache, policy eviction,
transposition table, or full incremental MCTS traversal is implemented here.
The [support-certificate experiment](SUPPORT_CERTIFICATES.md) remains separate.

Update, 2026-09-12: [support certificates can now be enabled in MCTS/self-play](SUPPORT_SEARCH_INTEGRATION.md)
with a bounded worker-local cache. Full incremental traversal and the undo
prototype are still separate.

The subsequent [naive self-play throughput comparison](NAIVE_SELFPLAY_THROUGHPUT.md)
finds no substantial speedup from this memory pass: its evaluator does not use
dense logits, and the reversible/certificate prototypes are not integrated.
