# Naive self-play throughput after the memory changes

This is the historical memory-only comparison. The subsequent
[support-backend integration](SUPPORT_SEARCH_INTEGRATION.md) adds a separate
opt-in self-play acceleration path and its own paired measurements.

This comparison isolates the latest MCTS memory pass. The earlier fix avoiding
a duplicate transition inside `Action::try_apply` is present in **both** builds.
It does not compare against the repository before all prior optimizations.

The naive evaluator returns a heuristic policy, not a dense logit map. It does
not use `FineGrid` or the new shared-logit path. Support certificates and the
compact undo prototype remain laboratory-only and are not used by this search.
There is therefore no direct algorithmic speedup expected from those changes
in this workload; code layout and the smaller inline node layout can still
affect measured timing.

## Results

Median throughput, 2026-09-11:

| Radius | Workers | Before, plies/s | After, plies/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 1/10 | 1 | 385.42 | 391.45 | +1.56% |
| 1/18 | 1 | 126.28 | 125.08 | −0.96% |
| 1/18 | 2 | 225.53 | 229.20 | +1.63% |
| 1/38 | 1 | 158.04 | 158.14 | +0.07% |

There is **no substantial, consistent self-play speedup** in this test. The
sample time ranges overlap between variants in every fixture. These small
differences are not evidence that a cache algorithm is accelerating naive
play. This is consistent with shared-logit storage not being used at all by
the naive evaluator.

Each timed radius-1/10 sample ran 176 plies, with two completed games and two
capped games. The other samples ran 256 plies, with all four games capped.
Across 40 timed samples there were 160 playouts and 9,440 plies. Warmups add
another 32 playouts. Every non-timing aggregate matched within each fixture;
the one- and two-worker radius-1/18 cases also share the same result fingerprint.

To test the memory saving's effect on actual throughput, use a dense-policy
evaluator and a sufficiently large concurrent search workload. To test the
certificate/undo designs with naive self-play, first integrate those opt-in
algorithms into search; their earlier microbenchmark gains cannot be claimed
as self-play speedups today.

## Method

- Release builds on the same AMD Ryzen 9 9950X, Rust 1.97.1.
- Existing `vgo-canary` playout driver and `NaiveEvaluator` on both sides.
- Equal 64-simulation players; two seed pairs (41 and 42), four playouts per
  sample; maximum 64 plies per game. Default canary rules/komi/search settings.
- Radius 1/10, 1/18, and 1/38; one worker, plus a two-worker case at 1/18.
- One warmup per binary per fixture, then five samples in alternating order.
- Measure the driver's wall time, which includes searches, committed moves,
  game accounting, and worker coordination. It excludes compilation/process
  startup, neural inference, replay serialization, and shard writing.
- Check equality of every reported non-timing aggregate, including captures,
  passes, completions, evaluations, generated candidates, and search depth.
  This is an aggregate equivalence gate, not a full per-move replay comparison.
- Report plies/second: most playouts reach the cap, so calling these rates
  completed-games/second would be misleading.

The baseline was prepared in an isolated temporary source copy, preserving the
current workspace except for restoring these six files from commit
`4dd76e605737a8d26151690d78b31b3e2c17bfe2` (their versions before the memory pass):

```text
crates/vgo-search/src/coarse_fine.rs
crates/vgo-search/src/evaluator.rs
crates/vgo-search/src/lib.rs
crates/vgo-search/src/mcts.rs
crates/vgo-search/src/stepped.rs
crates/vgo-raster/src/policy.rs
```

All other current sources, including the earlier `candidates.rs` fix, were
identical between builds. Feature-gated core lab additions were not enabled.
The working repository was not reverted.

Binary SHA-256 fingerprints for this run:

```text
before a19c3cd7e07ca01047e03751392f0bf65e1186433358722f008935855d477a43
after  aeb0ac6065c226696a6bb98c2baef0015c6e560427c07292af5e554378c40af5
```

The paired runner accepts two prebuilt `vgo-canary` binaries:

```sh
python3 runs/naive-throughput.py /path/to/before /path/to/after --samples 5
```

Raw samples and aggregate-result fingerprints are in
[`naive-throughput-2026-09-11.csv`](../diagnostics/naive-throughput-2026-09-11.csv).
