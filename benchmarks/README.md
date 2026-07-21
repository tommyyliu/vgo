# Simulator Benchmarks

Benchmarks compare semantically equivalent work, not isolated language syntax.
Every backend must consume the same serialized positions and produce outputs
that pass conformance checks before timing is considered.

## Workloads

### Position analysis

For sparse, midgame, dense, boundary-heavy, and empty-legal-set positions:

- validate the position;
- construct all Voronoi cells and positive-length adjacency;
- enumerate legal-set vertices;
- classify every settled group;
- compute both scores.

Report latency distributions and analyses per second by stone count.

### Move batches

Apply deterministic legal move sequences with opponent capture and self-capture
resolution. Report moves per second, allocations, and the number of full
analyses performed.

### State encoding

Once the initial tensor schema is fixed, encode batches rather than individual
positions. At minimum report:

- positions per second by batch size and raster resolution;
- output bytes and temporary allocations;
- time spent in simulation, encoding, and language-boundary transfer;
- exact channel checksums for conformance.

CPU Rust, JavaScript, and later vectorized/GPU training encoders should be
reported separately. End-to-end actor throughput is the deciding metric.

### Inference boundary

Exercise the Rust inference broker against both a deterministic fake service
and the Python model service. Sweep actor count, batch size, maximum queue wait,
and number of in-flight batches. Report queue delay, transport time, model time,
actor utilization, and complete games per second.

## Method

- Use deterministic fixture files and fixed seeds.
- Warm up each runtime before measurement.
- Separate construction from steady-state iteration when both matter.
- Record toolchain, CPU, build mode, and benchmark parameters.
- Use release/optimized builds.
- Retain raw JSON results so comparisons can be regenerated.

The Rust self-play canary is implemented by `vgo-canary`. Geometry conformance
fixtures and cross-language throughput harnesses remain future work.

## Self-play canary

Before introducing a learned model, use the same deterministic spread-out
policy and neutral nonterminal value for every player. The primary system test
is whether additional MCTS computation produces stronger play.

### Primary match

```text
high-compute player: 1000 MCTS simulations per move
low-compute player:    10 MCTS simulations per move
policy/value model:    identical naive evaluator
```

Run games in paired trials. For every match seed, play once with the
high-compute player moving first and once with the colors reversed. A draw
contributes one half point.

Candidate sequences must be deterministic prefixes keyed by position and match
seed. The 1000-simulation search therefore begins with the same suggestions as
the 10-simulation search and obtains additional candidates only through
progressive widening. Evaluation move selection is deterministic from root
visits; training temperature and exploration noise are disabled.

The canary agent is history-aware without changing game legality. It walks the
root moves in visit/value order and skips any move that recreates an earlier
complete position in that game. If no unseen root move exists, it passes. This
turns a repeated suggestion into deterministic resampling and lets games finish
under the actual pass/pass rule. `repetition_avoids` reports every skipped root
move; the engine itself still permits unrestricted repetition.

The initial acceptance criterion is:

- at least 200 completed games, organized as 100 color-swapped pairs;
- the high-compute player's 95% confidence lower bound exceeds 50% match score;
- truncated games are at most 2% and are reported separately;
- `10 vs 10` and `1000 vs 1000` controls remain statistically compatible with
  50% after color pairing.

Also run a `10, 30, 100, 300, 1000` simulation ladder. Estimated strength should
trend upward with compute even if adjacent confidence intervals overlap. Record
candidate counts, search depth, terminal leaves, captures, passes, repetitions,
wall time, and nodes per second for every budget.

Here one MCTS simulation means one complete tree-selection, leaf-evaluation,
and backup operation. It does not mean a complete game rollout.

### Running it

From the repository root:

```powershell
cargo run --release -p vgo-selfplay --bin vgo-canary -- `
  --pairs 100 --first 1000 --second 10 --max-plies 48 --threads 4 --seed 10001
```

The neutral nonterminal value is represented as `0` on the internal `[-1, 1]`
utility scale, equivalent to `0.5` win probability. Terminal values come from
exact scoring. The naive policy prefers clearance, strongly demotes placements
that self-capture, and never removes them from the legal candidate stream.

The first acceptance run is recorded in
[`results/2026-07-21-selfplay-canary.json`](results/2026-07-21-selfplay-canary.json).
The primary match completed all 200 games and scored 94.5% for 1000 simulations
against 10, with a 90.4% Wilson lower bound. Both equal-budget controls scored
exactly 50% after pairing.
