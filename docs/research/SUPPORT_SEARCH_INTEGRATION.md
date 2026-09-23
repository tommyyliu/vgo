# Support certificates in MCTS and naive self-play

The support-certificate experiment is now connected to `vgo-search::Action`
transitions and the `vgo-canary` self-play driver. It remains **opt-in**. Normal
builds and runs retain the baseline algorithm; geometry is still rebuilt.

## Run

```sh
cargo build --release -p vgo-selfplay --features iteration-lab --bin vgo-canary

# Baseline, same binary:
target/release/vgo-canary --pairs 2 --first 64 --second 64 \
  --max-plies 64 --radius 0.05555555555555555 --threads 1 --seed 41

# Support backend; default retained-entry budget is 1 MiB per worker/game:
target/release/vgo-canary --supports --support-cache-mib 1 \
  --pairs 2 --first 64 --second 64 --max-plies 64 \
  --radius 0.05555555555555555 --threads 1 --seed 41

# Slow correctness mode: compare every accelerated full result with baseline.
target/release/vgo-canary --supports --verify-supports \
  --pairs 2 --first 64 --second 64 --max-plies 64 \
  --radius 0.05555555555555555 --threads 2 --seed 41

# Alternating paired playouts, verification disabled for timing:
python3 runs/naive-throughput.py target/release/vgo-canary \
  target/release/vgo-canary --samples 5 --after-arg=--supports
```

The feature exposes `transition_lab::with_support_backend(config, operation)`
for other callers. Wrap a synchronous search or entire playout on its worker
thread. The canary wraps each game, so caches survive consecutive searches but
are released between games. The wrapper covers naive policy action scoring,
search expansion, result selection, and committed `Action` placements. Passes
remain baseline. The website ruleset bypasses the support backend.

Sequential, batched, and synchronously driven stepped searches can use the same
scope. It does not propagate to evaluator threads or asynchronous tasks, and
direct `vgo_core::place` calls remain unchanged. Other generators/browser clients
do not gain a new default backend merely by compiling the feature.

## Memory and correctness contract

- Store owned parent support caches in a thread-local LRU, **not in tree nodes**.
- Retain all support edges; do not eagerly cache negative cell polygons.
- Bound retained entry payload by the configured byte budget and separately cap
  the LRU at 64 entries. Entries own their parent position and prepared arrays.
  Fingerprints accelerate lookup, but reuse also requires exact ordered stone
  coordinates/colors, radius, ruleset, komi, turn, phase, and pass count.
- Eviction drops only optional proofs. Every uncertified group uses the existing
  analytic settlement search; cache misses never imply capture.
- A zero budget bypasses preparation. An oversized entry is discarded and the
  transition uses baseline. The limit is **not peak process memory**: allocation
  headers, LRU slack/metadata, construction scratch, and geometry/results are
  outside it. A candidate must currently be prepared before its exact retained
  payload can be checked, so an oversized parent can repeatedly waste preparation.
- Scope exit frees its cache and restores any enclosing scope, including during
  unwinding. Tests cover nesting, thread isolation, eviction, zero/tiny budgets,
  and metadata-sensitive identity.

No fully incremental geometry, reversible MCTS traversal, persistent per-node
proof graph, or policy-cache eviction is introduced here. The
[compact undo prototype](MCTS_MEMORY_LAB.md) remains separate.

## Validation

Unit tests compare complete naive `SearchResult`s for both rulesets, three
radii, and sequential/batched/stepped searches, with transition verification
enabled. The core support tests continue to cover tangencies, dormant-point
reactivation, captures, and independently seeded trajectories.

Three full-playout verification fixtures at radii 1/10, 1/18, and 1/38 checked
**117,082 accelerated transitions** (30,680 + 43,234 + 43,168) against baseline
complete results without disagreement. Each fixture ran four games, 64
simulations per move, at most 64 plies, on two workers. Radius 1/10 included
ordinary and self-capture events. These are engine differential checks, not an
independent proof of ideal real-arithmetic correctness.

Reported verification outcomes are in
[`support-selfplay-verification-2026-09-12.log`](../diagnostics/support-selfplay-verification-2026-09-12.log).
Throughput runs additionally require equality of all non-timing game/search
aggregates. The backend reports calls, hits, builds, evictions, oversized entries,
and peak retained payload to stderr; those are diagnostic counts, not gameplay
outputs.

## Self-play throughput

Final paired measurements on 2026-09-12, AMD Ryzen 9 9950X, Rust 1.97.1,
release profile. Both variants use the **same feature-enabled binary**, toggling
only `--supports`; verification is disabled while timing. Each case has one
warmup per variant followed by five alternating samples. A sample runs four
naive-evaluator games with 64 simulations per move and a 64-ply cap, seeds 41/42.
Preparation, lookup, eviction, geometry rebuilding, search, and committed moves
are all included in the driver's wall time.

| Radius | Workers | Baseline, plies/s | Supports, plies/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| 1/10 | 1 | 391.94 | 415.30 | **6.0%** |
| 1/18 | 1 | 129.33 | 130.56 | **1.0%** |
| 1/18 | 2 | 233.14 | 241.34 | **3.5%** |
| 1/38 | 1 | 160.41 | 174.40 | **8.7%** |

The gains are modest and workload-dependent; the 1% middle-radius single-worker
result is too small to treat as a compelling speedup. This is not the earlier
1.45× sibling-transition microbenchmark: full search pays cache preparation for
new parents, candidate generation, scoring, traversal, and geometry work.
The current implementation is still a rebuild engine with reusable proofs.

Raw paired samples, including result fingerprints, are in
[`support-selfplay-throughput-2026-09-12.csv`](../diagnostics/support-selfplay-throughput-2026-09-12.csv).
The [companion log](../diagnostics/support-selfplay-throughput-2026-09-12.log)
contains cache metrics and timing ranges. Each timed variant matched all
non-timing aggregate results. Two of four radius-1/10 games finished; the others
hit the cap, so rates are **plies**, not completed games, per second. No neural
inference or training-shard output is included.

Keep the backend opt-in. This establishes an end-to-end benefit on some naive
workloads, not a universal gain or a reason to promote the experiment to all
production generators without wider validation.

Follow-up: [dense-policy CPU search measurements](DENSE_SUPPORT_THROUGHPUT.md)
test 1,400 simulations with 1/8/32 actors, including denser boards and a
saved-replay input. [Process RAM scaling](RAM_SCALING.md) covers the latest
exact memory reductions separately.

## Cache budget experiment

At radius 1/18, three alternating samples comparing 1 MiB with 4 MiB produced
exactly the same **39,024 hits and 4,210 builds** in each four-game sample.
Peak retained payload rose from 1,048,520 B to 3,037,888 B. Median throughput
was 130.98 versus 130.46 plies/s: no demonstrated benefit from the larger budget.
These are a separate pilot, not timings paired with the baseline table.

Raw observations are in
[`support-cache-budget-2026-09-12.json`](../diagnostics/support-cache-budget-2026-09-12.json).
Keep the 1 MiB default for now. The evidence points toward preparation on new
parents and repeated geometry work, rather than insufficient retained cache
capacity, as the next optimization targets in this fixture.
