# Replay audit, tree reuse, and inference deadlines

## Scope and reproduction

This follows the [restart pilots](RESTART_2026-10-14.md). It uses their existing
completed games for the tail audit, then compares one fixed model against
itself with and without tree reuse. Generation/training defaults stay unchanged.

Artifacts are in `artifacts/reuse-2026-10-14/`: input hashes, per-game audit
results, exact commands, binary/model hashes, exit codes, sampled process RSS,
arena JSON and SGFs, and test logs. The experiment runner refuses overwrites
and bounds each child process independently of the native deadlines.

```bash
training/.venv/bin/python scripts/experiments/audit-replay-tails.py \
  artifacts/experiments-2026-10-14/budget-games/fixed \
  artifacts/experiments-2026-10-14/budget-games/randomized \
  --output artifacts/reuse-2026-10-14/tail-audit.json
source scripts/env/ort.sh
training/.venv/bin/python scripts/experiments/reuse-2026-10-14.py smoke
training/.venv/bin/python scripts/experiments/reuse-2026-10-14.py arena
```

For a repeat, pass `--output artifacts/reuse-repeat-N` with a fresh directory.
`final-smoke` repeats the healthy two-slot check plus the guarded missing-library
case after the last build.

## Existing replay audit: closed with a calibration fix

The audit reads all binary rows of 32 fixed-budget and 90 randomized-budget
pilot games. It independently clips Voronoi cells from the stored final stones
and checks final areas, margin, winner, sample eligibility and mover-relative
value labels. All 122 final scores/winners agree within `3e-6`, allowing for
rounded JSON coordinates/komi; all replay joins and value labels agree.
No continuation games were generated for this audit.

| Completed-game measurement | Fixed | Randomized |
| --- | ---: | ---: |
| Training rows | 3,930 | 3,615 |
| Capped games | 7 / 32 | 22 / 90 |
| Rows strictly after the soft trigger | 2,311 (58.8%) | 1,206 (33.4%) |
| Tail / total simulations | 924,400 / 2,543,400 (36.3%) | 1,212,800 / 5,017,600 (24.2%) |
| Tail rows repeating an earlier recorded board | 213 (9.2%) | 38 (3.2%) |
| Unchanged boards across adjacent recorded tail plies | 201 / 2,288 (8.8%) | 26 / 287 (9.1%) |
| Actual soft concessions by the eventual winner | 2 / 23 | 14 / 49 |

Compute totals here cover completed games only. Earlier generation telemetry
includes unfinished games too. The trigger ply itself uses its original search
budget; cheaper search starts on the following ply. Board comparison ignores
the mover but uses exact binary coordinates. Sparse randomized replay cannot
establish repetition or no-op rates across unrecorded transitions.

The reversal counts also follow directly from the actual soft trigger's mover
and the final winner, independently of calibration summaries. They describe
the completed games under cheaper continuation, not an estimate of outcomes
under hypothetical full-budget continuation. Different board mixtures, one
incumbent, and the short timed cutoff prevent generalizing the rates.

Closing boards can change winner: 18 fixed and 31 randomized games contain an
opposite-signed score among the recorded final eight pre-move positions. This
does not establish a scoring bug or predict hypothetical continuation. It
supports scoring the committed final board, as the engine already does.

**Decision:** retain soft continuation, exact final-board labels, and current
tail weights. Most tail boards change, and some reverse an early concession.
These observations do not justify broadly excluding or downweighting the tail.
The bounded audit is closed; reopen for a concrete new scoring/target defect.

### Calibration defect and repair

The counterfactual sweep ignored `resign_minimum_ply`, although the live rule
honors it. The sweep now applies the same floor while accumulating each seat's
streak before the floor, matching live behavior. It also counts a concession
on a final draw as a false decision.

New calibration rows include `minimum_ply`. The threshold picker filters that
field, and the bulk loop passes its live floor. Legacy rows have an effective
counterfactual floor of zero; they remain readable but cannot be pooled with
new floor-20 rows. Until enough matching firings accumulate, the existing
fallback threshold is used. Existing pilot records were not rewritten.

## Recovering prior work

Searches of current sources, all local/remote branch histories, commit messages,
and reroot/reuse symbol history found no existing MCTS reroot implementation.
The removed game-tree UI is a variation/record tree, separate from MCTS.

Useful work does exist: `SteppedSearch` already preserves a live tree between
evaluation rounds. Commit `3709771` removed tree-memory accounting; its parent
contains the original `memory.rs`, policy allocation hooks, and node/grid
accounting. Those pieces were recovered and extended to account for legacy
candidate payloads and the naive policy. Shared policy/grid logits are counted
once. This recovers the diagnostic without restoring the removed binaries.

## Reuse prototype

`ReusableSearch` keeps each seat's own evaluated tree. After every actual move,
it extracts the matching evaluated child and drops siblings. Exact `Position`
equality checks ordered stones, captures, mover, pass state, phase, radius,
komi, and rules. Missing/unevaluated moves discard the tree. Model revision,
configuration, seed, or requested-position mismatches also discard it.

Inherited visits participate in selection and widening; the simulation budget
and search statistics count only new work. Root exploration noise is rejected
because inherited proposals have different sampling probabilities from a newly
noised root. Reuse is exposed in the arena only, with independent seat flags.
Self-play target generation remains fresh-tree search.

Retention defaults to 4,096 nodes and 256 MiB of tracked payload per seat;
exceeding either limit discards the cache. Unknown policy allocations also
disable retention. Accounting excludes allocator/reference-count/hash-control
metadata and stone capacity beyond the public live slice, so this is not an
RSS guarantee. Node count additionally limits retained structure; process RSS
is sampled independently. Searches may transiently grow beyond retention limits
before the completed tree is accepted or discarded.

Reroot/drop/accounting time between moves is deducted from the seat's next
advisory time budget. Final accounting is included in reported search elapsed
time. One completed simulation round remains guaranteed, so deadlines can
overrun during inference or bookkeeping, just as fresh searches can overrun.

### Arena protocol and results

The incumbent `r128-w150k.onnx` plays itself, candidate reuse enabled and
opponent reuse disabled. Each board uses 24 color-swapped pairs (48 games),
seed 940001, 200 ms per move, 10,000 new-simulation ceiling, 16 game threads,
two independent TensorRT FP16 sessions, maximum batch 32, leaf batch 4,
coarse pool 16, candidate ceiling 321, widening coefficient 6, zero root noise,
komi `28 / units²`, and caps 70 / 312 for 18 / 38 units.

| Measurement | 18 units | 38 units |
| --- | ---: | ---: |
| Reuse wins–losses | 24–24 | 23–25 |
| Approximate per-game 95% score interval | 36.4–63.6% | 34.5–61.7% |
| Searches with a retained root | 975 / 1,409 (69.2%) | 2,229 / 4,006 (55.6%) |
| Inherited visits per hit | 12.2 | 8.6 |
| New simulations/move, reuse / fresh | 179 / 171 | 137 / 146 |
| Mean search ms, reuse / fresh | 202.7 / 202.9 | 205.0 / 205.2 |
| Maximum tracked retained payload per seat | 20.5 MiB | 21.3 MiB |
| Maximum retained nodes per seat | 341 | 332 |
| Reroot/drop/accounting time, entire reuse seat | 179 ms | 338 ms |
| Sampled peak process RSS | 1.27 GiB | 1.26 GiB |

Both 48-game matches exited zero, with no inference failures or cache-limit
drops. The score intervals are per-game Wilson intervals, not paired-bootstrap
or equivalence tests. A single match per board cannot establish strength parity.
The two seats also search different positions, so their new-simulation averages
are not an isolated implementation-throughput benchmark.

**Decision:** keep reuse opt-in. Useful branches frequently survive, but each
hit carries relatively few visits. These matches establish a working bounded
prototype, not a strength or efficiency gain. No campaign or browser default
changes follow from them. Consider longer-budget or controlled-position
measurements only if reuse becomes a priority again.

## Inference deadlines and failure behavior

ONNX loading now has a separate readiness deadline, default 1,800 seconds,
configured by `VGO_INFERENCE_STARTUP_SECONDS`. It logs model/provider/device,
batch/cache settings, readiness duration, and a waiting message every 30 s.
TensorRT construction can legitimately be slow; a waiting message only says
the native call has not returned. It cannot prove whether a builder is making
progress inside ONNX Runtime.

Both single-session and pooled brokers use executor deadlines, default 120
seconds per dispatched batch, configured by `VGO_INFERENCE_TIMEOUT_SECONDS`.
Errors identify model/provider, slot, batch sequence, request ID range, elapsed
time and limit. A pool checks its oldest outstanding batch so faster slots
cannot indefinitely hide a stalled slot. Waiting/queued callers fail when the
broker stops. Invalid/nonpositive deadline settings are rejected.

On a native timeout, channels disconnect and the stuck worker is detached:
Rust cannot safely interrupt an arbitrary native GPU call. Healthy shutdown
joins workers normally. The owner must exit and restart in a fresh process;
this is bounded failure handling, not in-process native recovery. Generation
and arena errors already exit; the move server now sends its error response
and exits too. A real missing-runtime check showed that returning from `main`
can itself hang in native exit cleanup after the timeout. Native failures now
also arm a three-second final watchdog. On Linux it calls safe
`rustix::runtime::exit_group(70)`, bypassing native exit handlers; other targets
use process abort. Healthy runs do not arm this watchdog. Published game files
remain atomic; in-flight games/staging may be abandoned on a fatal failure.

The first missing-library attempt used a 15 s outer cutoff against a 120 s
startup deadline and was excluded as a runner mistake. The second reported its
2 s startup error but required the outer supervisor to kill native cleanup.
Both failure traces remain in `runs.jsonl`; the final guarded retry is the
acceptance check for bounded process exit.
It reported the startup timeout at 2.000 s and exited with status 70 after
5.03 s, without an outer-supervisor kill. A final healthy two-slot GPU run
exited zero in 3.79 s including startup and a 3 s generation window.

### Verification

Deterministic gated-service tests cover stalled single/pool slots, queued caller
failure, a faster slot not masking an older stall, bounded teardown, startup
success/error/panic/timeout, and reuse cold parity, inherited visits, passes,
captures, invalidation, noise rejection, and retention limits. Calibration tests
cover the minimum-ply floor, draw outcomes, and picker identity.

Real GPU smoke runs with two and four slots both exited zero. The brief
four-slot check did not reproduce the historical stall and does not establish
four slots as suitable for campaigns. Keep the measured two-slot setup.

The final suite passed 101 Rust tests (one pre-existing test ignored) and the
Python calibration-picker regression. Both arena matches completed all 48
games with zero inference failures. Arenas preceded the final native-exit
watchdog addition; the search implementation was unchanged by that repair.
Each run records its actual binary digest rather than assuming all runs used
the final build.
