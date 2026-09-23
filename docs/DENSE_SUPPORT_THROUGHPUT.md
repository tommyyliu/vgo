# Dense-policy certificate throughput experiment

This extends the [naive self-play measurement](SUPPORT_SEARCH_INTEGRATION.md)
to CPU searches with production-like dense policies and actor counts. It is
**not neural self-play throughput**: inference, input rasterization, replay
writing, and game-to-game scheduling are absent.

## Method

The RAM probe now accepts `--supports`, `--verify-supports`, and `--stones`.
Each worker enters its own support scope with a 1 MiB retained-entry budget;
the baseline uses the same feature-enabled binary without that scope. Both
variants have shared logits, packed legality masks, and trimmed override arrays.
All actors start together, retain completed trees together, then finish and
release them. Full `SearchResult` fingerprints must match across both variants
and all repetitions of each fixture. Verification runs additionally compare
every accelerated transition with baseline, outside the timing experiment.

Settings: 1,400 simulations, leaf batch 4, 128×128 dense logits, coarse pool 16,
widening coefficient 6, maximum 321 candidates, radius 1/38, and seeds 73+actor.
Logits depend deterministically on cell index and stone count; values are zero.
The 20-stone fixture is the original sparse memory-test board. The 80/160-stone
fixtures use the first 80/160 sites of a 16-column lattice, spacing 1/17, with
alternating colors. These are controlled stress fixtures, not representative
samples of the distribution of played games.

Five paired samples follow one warmup per variant; execution order alternates.
The primary timer covers fresh subprocess startup, evaluation, search, memory
inventory, result fingerprinting, and teardown. The separate `build_seconds`
column stops with all trees held; neither timer is pure transition time.
Runs are sequential across variants, with an 8 GiB virtual-memory limit and
180-second timeout per process. See [RAM scaling](RAM_SCALING.md) for admission
checks and memory-model limitations.

## Controlled-grid results, 2026-09-12

Median external wall seconds, baseline → supports (throughput multiplier):

| Stones | 1 actor | 8 actors | 32 actors |
| ---: | ---: | ---: | ---: |
| 20 | 0.213 → 0.218 (0.98×) | 0.227 → 0.231 (0.98×) | 0.391 → 0.421 (0.93×) |
| 80 | 2.304 → 0.883 (2.61×) | 2.435 → 0.975 (2.50×) | 4.235 → 1.821 (2.33×) |
| 160 | 12.745 → 2.885 (4.42×) | 13.488 → 3.217 (4.19×) | 24.465 → 5.530 (4.42×) |

All 90 timed process runs and 18 warmups passed the fingerprint gate. A separate
64-simulation, two-actor verification sweep covered all three board densities.
The sparse-board differences are small and some timing ranges overlap: no
sparse-board benefit is established. The dense-grid gains are large and persist
with actor concurrency. However, board shape, connectivity, and policy shape
all affect the cost; stone count alone is not a sufficient dispatch rule.

At 160 stones and 32 actors the support process peaks around 3.02 GiB, versus
2.98 GiB baseline. The bounded proof cache remains a small addition to the tree.
At 1 actor it rebuilt 877 parent caches in 2,800 transition calls, with 1,923
hits and 871 evictions. Thus the gain includes substantial preparation and
eviction, not just a prewarmed sibling-transition microbenchmark.

## Reproduce

```sh
cargo build --release -p vgo-raster --features iteration-lab --example ram_scaling
python3 runs/dense-support-throughput.py target/release/examples/ram_scaling --verify --samples 1
python3 runs/dense-support-throughput.py target/release/examples/ram_scaling --samples 5
```

Raw outputs are under `diagnostics/dense-support-{verification,throughput}-2026-09-12`
as CSV samples and logs containing cache metrics and timing summaries.

The optional `--shard PATH` runner mode uses records 80, 160, and 240 of a v8
replay, one actor each. The probe reconstructs full-precision centers, colors,
radius, komi, side to move, and previous-pass count, requiring a playing phase.
It also restores the recorded ply for search seeding and temperature scheduling.
The format does not encode rules, so this benchmark explicitly uses Vgo rules.
These runs still use synthetic policies and zero values, not the original model.

Replay source for the 2026-09-12 check:
`artifacts/vgo-continuous/games/gen-000003-09010018-3f09463f/game-051514242/dataset.vgo`,
SHA-256 `70e051e52b5bb455d2869d37262f649e1d3689395c22e5b4c2574bf065bccdd2`.
This is one saved game, not a random sample of the production corpus.

```sh
python3 runs/dense-support-throughput.py target/release/examples/ram_scaling --verify --samples 1 --shard artifacts/vgo-continuous/games/gen-000003-09010018-3f09463f/game-051514242/dataset.vgo
python3 runs/dense-support-throughput.py target/release/examples/ram_scaling --samples 5 --shard artifacts/vgo-continuous/games/gen-000003-09010018-3f09463f/game-051514242/dataset.vgo
```

Replay samples and verification runs are saved separately under
`diagnostics/replay-support-{verification,throughput}-2026-09-12`, as CSV and logs.

## Saved-position results

Five paired timing samples per position, one actor, 1,400 simulations:

| Replay record / ply | Stones | Baseline seconds | Supports seconds | Throughput multiplier |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 80 | 0.977 | 1.173 | 0.83× |
| 160 | 153 | 2.568 | 3.480 | 0.74× |
| 240 | 217 | 8.634 | 7.668 | 1.13× |

All paired full-result fingerprints matched, and separate 64-simulation runs
with transition verification passed. The first two positions take 20% and 35%
longer; the last delivers 12.6% greater throughput. The sum of these three
median times is about 1.2% worse with supports, but this artificial weighting
is **not an estimate of average production throughput**.

This is the key limitation on the controlled-grid result: a large win is
possible, but the current integration does not reliably improve played-position
searches. Keep it opt-in. The 217-stone search builds 1,233 caches in 2,778 calls,
with 1,545 hits and 1,229 evictions despite staying below 1 MiB retained payload.
That makes preparation and eviction useful profiling targets; these counters
alone do not attribute elapsed time. Geometry and policy shape matter too.

Next: measure preparation, accelerated settlement, and fallback time separately
on a broader saved-position set; then test cheaper or incrementally maintained
proof preparation and cache admission. A stone-count threshold alone is not
justified. Only after that should full neural-generator A/B runs establish
whether CPU gains survive inference and rasterization costs.

Validation after the memory changes and benchmark additions: 119 core/search/
raster library tests passed, 5 existing timing/GPU tests ignored; the diagnostic
also builds without the experimental feature. No production backend default or
actor/simulation setting was changed.
