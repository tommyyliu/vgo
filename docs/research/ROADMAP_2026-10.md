# Project restart roadmap

Updated: 2026-10-14. Status: items 1–5 have completed bounded measurements;
item 6 has a measured arena prototype; item 7 has bounded failure
handling and verification. Experiment defaults retained pending stronger strength evidence.

## Direction

Improve playing strength per hour on the local machine before expanding the
big-board run or renting more cloud resources. CPU performance and the quality
of generated training data are the main questions. Use fresh arena games to
judge strength, and recorded timings to judge speed; validation loss alone does
not establish either.

The order below balances expected benefit, implementation effort, and how
quickly an experiment can answer a useful question. P0 establishes the baseline,
P1 is the next work, P2 follows once that work provides evidence, and P3 is
optional exploration.

## Prioritized work

| Order | Priority | Work | Result needed |
| --- | --- | --- | --- |
| 1 | P0 | Restore continuous-run performance visibility | Know where CPU time goes and how much useful data is produced per hour. |
| 2 | P1 | Optimize the measured CPU hot path | Demonstrate an improvement in real neural self-play, with rule parity. |
| 3 | P1 | Close the inexpensive model experiments | Decide whether 128 inputs and one norm per block improve strength per hour. |
| 4 | P1 | Experiment with randomized search budgets | Produce more independent games while retaining useful policy targets. |
| 5 | P2 | Audit capped-game scoring and tail samples | Establish whether the existing stalling mitigation distorts ratings or learning. |
| 6 | P2 | Prototype search-tree reuse | Improve strength at a fixed time budget by retaining useful search work. |
| 7 | P2 | Harden inference failures | Make startup and stalled requests fail with actionable diagnostics. |
| 8 | P2 | Make shared rule regressions easy to run | Check Rust and JavaScript behavior on common fixtures from Linux. |
| 9 | P3 | Ablate the expensive settled input plane | Measure whether its strength benefit justifies its CPU cost. |
| 10 | P3 | Improve the browser analysis experience | Expose candidate moves, evaluation, and model identity. |

### 1. Continuous-run performance visibility

Status: Done. Periodic/final generator telemetry and loop lifecycle events are
implemented and verified. Tests cover interval accounting, board/stage
separation, batch fill, and unchanged replay targets. CPU smoke runs reconcile
with published manifests. The fixed-position CPU baseline and a 120-second
neural self-play profile on the RTX 5070 Ti cover opening, middle, and late
stages and identify exact legal-space queries in raster classification as the
first optimization target. See [restart measurements](../docs/research/RESTART_2026-10-13.md).

- Report periodic plies/s, completed games/h, samples/h, inference encoding,
  queue and execution timings, observed batch sizes, and process memory.
- Separate results by board size and game stage. Mark training and overlapping
  generator periods so contention does not look like a code regression.
- Record the model digest, commit, hardware, and complete run configuration.
- Use the existing generator profiling feature and `scripts/bench.py`; collect
  representative middle/late-game work rather than only empty-board startup.
- Acceptance: one reproducible local baseline with a profile and stage metrics
  sufficient to choose the first CPU optimization. Concurrent timing totals
  must not be presented as additive shares of wall time.

`docs/SELFPLAY_ARCHITECTURE.md` and `docs/RUNNING.md` now describe the actual
continuous reporting and distinguish summed/position-weighted timings from CPU
time. Startup metadata records the backend's actual input layout; export
sidecars were also corrected to stop advertising packed graphs as dense inputs.

### 2. CPU optimization selected from the profile

Status: Done for the first measured optimization. The change reuses the legal-set stone buckets to
bound radial candidate enumeration in indexed threshold queries. Correctness
checks cover exhaustive comparisons, near-boundary thresholds, and the
geometry/search lab. Two alternating CPU benchmark pairs show about 13–14%
less big-board raster time. One short neural comparison shows about 3.9% more
searched plies/s after the first 30 seconds, with identical published-game and
sample counts. Two additional alternating neural benchmark pairs at 32 actors,
1,000 simulations and 256 inputs show a 4.4% mean big-board search-throughput
gain and 12.9% less encoding time; the individual gains vary from 1.5% to 7.3%.
These saved-position searches cover mature boards, but exclude publication and
do not establish a sustained learning-rate improvement. Retain the equivalent
pruning and prioritize model cost next. See [production results](../docs/research/RESTART_2026-10-14.md).

- Investigate whichever dominates: board transitions and settlement, packed
  raster construction, or legal policy-grid construction. Prefer eliminating
  repeated work and reusing indexes/scratch before a large geometry rewrite.
- Reuse the fixed played-position corpus and expand it if the profile exposes
  a missing workload. Keep synthetic lattice stress cases separate.
- Keep support certificates opt-in until preparation, lookup, fallback, and
  eviction costs are measured on a broader played-position corpus. Improve
  preparation or cache admission if those costs justify it.
- Acceptance: preserve complete move/search results where the change claims
  equivalent behavior, pass relevant rule regressions, and demonstrate an
  improvement beyond timing noise in both the affected benchmark and a matched
  neural self-play run.

Evidence: [benchmark history](../benchmarks/history.jsonl),
[geometry lab](../docs/research/BOARD_ITERATION_LAB.md), and
[support throughput](../docs/research/DENSE_SUPPORT_THROUGHPUT.md). The support
backend's large synthetic-grid wins did not reliably transfer to played
positions. This work continues [ticket 007](007-geometry-performance.md).

### 3. Inexpensive model experiments

Status: Done for the bounded pilots. Existing 128/256 checkpoints have identical
ordered dataset digests and recipes; 128 is much cheaper, but fresh equal-time
strength matches remain inconclusive (24–24 small, 30–18 big). The one-norm
architecture, checkpoint compatibility and packed export pass their checks;
same-corpus training is stable and serving is modestly faster. Four fresh arenas
show mixed, inconclusive strength results. Retain two norms and the existing
campaign settings; 128 is the leading candidate for a new efficiency-focused
campaign, subject to a larger strength replication. See [results and recipes](../docs/research/RESTART_2026-10-14.md).

- First compare 128 and 256 input rasters at the same 128 policy resolution,
  using the same corpus and training recipe. Measure generation throughput,
  training time, memory, and strength on both board sizes.
- Separately compare one versus two normalizations per residual block. Verify
  training stability and exported fp16 behavior, then run an arena.
- Evaluate promising variants at equal thinking time as well as equal
  simulations. Report uncertainty; a small inconclusive match is not evidence
  of equal strength.
- Acceptance: a recorded adopt/reject/inconclusive decision for each variant,
  with model artifacts and reproducible commands.

Evidence: [resolution match](../artifacts/res-ab/arena-r38.json) and
[normalization measurements](../docs/MODEL_ARCHITECTURE.md). One norm per block
has measured speed gains but no measured strength result.

### 4. Randomized search budgets and sample quality

Status: Bounded pilots complete; strength inconclusive, retain fixed budgets.
Full/fast selection, per-ply
metadata, eligible-turn telemetry and full-selected replay are implemented and
tested; every turn still enters resignation calibration. In matched 600-second
generation slices, randomized budgets publish 90 games/3,615 rows versus
32/3,930 for fixed. However, the tiny from-scratch randomized student fails to
learn its deployed value head and loses all four 48-game arenas. Stored plies,
movers and value labels are verified correct. Keep fixed budgets as the default.
The follow-up adds each arm to the same 150k common replay corpus. Both students
learn their deployed value heads; optimizer work differs by less than 0.05% and
generation-plus-training wall time by 0.6%. Randomized data scores 28–20 small
and 26–22 big at equal thinking time, with both intervals including 50%.
Adoption needs a larger-turnover, multiple-seed or multiple-update replication;
the current controls stay opt-in. See
[results, caveats and failed-attempt provenance](../docs/research/RESTART_2026-10-14.md).

- Mix occasional full searches with cheaper searches on other turns. Keep this
  separate from the existing reduction after soft resignation.
- Record the search budget or training eligibility for each sample. Use full
  searches for policy targets in the first experiment; explicitly specify
  which positions contribute value targets and their weights.
- Compare against the current fixed-budget recipe after equal total elapsed
  compute. Measure independent games, eligible samples, and fresh arena
  strength, not just raw replay volume.
- Acceptance: a bounded local A/B with a recorded decision and enough metadata
  to explain the resulting training distribution.

Reference: [KataGo playout cap randomization, section 3.1](https://arxiv.org/html/1902.10565v5#S3.SS1).
Its generation benefit is measured in a bounded slice; its strength-per-hour
benefit remains unresolved.

### 5. Bounded audit of capped games and cheap tails

Status: Done. Existing 122 pilot games have consistent final scores/winners and
replay labels. Most tail boards change; actual soft-concession reversals were
2/23 fixed and 14/49 randomized, so broad tail exclusion is unsupported. Keep
soft continuation and current weights. Fixed the calibration sweep's missing
minimum-ply floor and separated incompatible calibration histories. See
[audit and follow-up](../docs/research/TREE_REUSE_2026-10-14.md).

**Established project context:** earlier investigations allowed bots to continue
playing and found that they stalled even on completely settled boards. Soft
resignation was implemented to reduce the compute spent on those tails. A high
move-cap rate is therefore a known behavior, not a new finding or sufficient
reason to repeat the longer-game experiment whenever the repository is reviewed.

- Inspect existing final boards, scores, game records, and resignation data
  before generating additional games.
- Quantify the share of replay samples and compute spent after soft resignation,
  and how much of that tail is repetition or no-op play.
- Check whether capped-board scoring creates suspect outcomes, and whether
  cheap tail positions deserve lower policy weight or exclusion. Coordinate
  any weighting experiment with item 4.
- Continue selected games only to answer a specific unresolved question about
  score or winner stability. Increasing the cap is not the default fix.
- Acceptance: document whether a concrete scoring or sample-quality problem
  exists. If no material problem is found, retain the current mitigation and
  close this audit. Reopen only with new evidence or a changed termination rule.

This audit should precede a long new big-board campaign. It does not require
changing the game's rules or automatically declaring settled positions terminal.

### 6. Search-tree reuse

Status: Done for the bounded arena prototype. Equal-time matches scored 24–24
small / 23–25 big, with reuse hits on 69% / 56% of moves. Inherited visits per
hit were only 12 / 9; strength and efficiency benefit remain unproven. Keep opt-in.
Recovered memory accounting from before `3709771`; history searches found no
existing reroot implementation. Reuse is opt-in and independent per seat;
exact-position/model/config/seed validation, node/payload limits, inherited
visit accounting and reroot time charging are implemented. Self-play remains
fresh-tree search; root noise is rejected by the prototype.

- Start with arena or browser searches under a fixed model, retaining the
  subtree matching the move actually played.
- Invalidate on model/configuration changes or a position mismatch; handle
  captures, passes, undo, and moves absent from the retained tree.
- Bound memory and define how root noise and inherited visits behave before
  extending reuse to self-play policy targets.
- Acceptance: measure reuse hit rate, retained memory, and playing strength at
  equal elapsed time. Adopt only if the benefit survives the bookkeeping cost.

### 7. Inference failure handling

Status: Bounded startup/batch deadlines implemented for single and pooled
brokers, with gated-service regressions and real GPU smoke checks. Startup
logs distinguish waiting for native construction from readiness. A timeout
disconnects callers and detaches the stuck native worker so the owner can
exit; a final three-second watchdog bounds stuck native exit handlers too.
Recovery requires a fresh process. The short four-slot smoke passed,
so the historical stall was not reproduced and two slots remain recommended.

- Reproduce the documented ONNX Runtime startup hang and four-slot stall with
  bounded diagnostics. Use the known working two-slot configuration meanwhile.
- Add startup readiness and progress deadlines with useful model, backend, and
  request diagnostics. Ensure blocked callers receive failure and sessions can
  shut down cleanly.
- Acceptance: missing runtime libraries, backend startup errors, and a stalled
  service fail observably instead of leaving a run indefinitely stuck. Normal
  TensorRT engine construction must remain distinguishable from a stall.

### 8. Shared rule regressions and backlog accuracy

- Create common transaction fixtures for Rust and the JavaScript reference,
  including capture stages, revival after opponent removal, tangencies,
  pass/pass endings, and both supported rulesets where applicable.
- Make the reference browser suite runnable on Linux without its fixed Windows
  Chrome path. Add a documented command and an automated gate.
- Reconcile existing tickets with current code. Game-grouped validation already
  exists in `_stable_game_hashes` and has a regression test; review any remaining
  cross-shard criteria before closing ticket 008. Ticket 009's promotion-path
  description predates the current ungated bulk loop.
- Acceptance: both engines consume the common fixtures, failures return
  nonzero, and backlog statuses describe remaining work accurately.

### 9. Settled-plane ablation

- Retrain matched models with and without the settled input plane, preserving
  exact game rules and legal move sampling.
- Measure the actual raster CPU saving and the strength cost on both board
  sizes. Account for feature information the model must learn to reconstruct.
- Acceptance: decide from strength per hour and end-to-end measurements. This
  is a representation experiment, not a claim that the feature is unnecessary.

### 10. Browser analysis

- Expose top candidate moves, a clearly labeled win estimate, and model identity.
- Add a small gallery of interesting tactical positions from recorded games.
- Measure browser memory and prevent overlapping searches on one bot.
- Acceptance: a playable analysis view that supports undo/cancellation and
  helps a human inspect model weaknesses. Reuse item 6 if it proves worthwhile.

## Restart sequence

1. Establish the local baseline and choose one CPU optimization.
2. Close the resolution/normalization experiments and run the search-budget A/B.
3. Complete the bounded capped-game audit before investing in a long campaign.
4. Resume big-board training using the measured winners, with periodic ratings
   and performance records.

Cloud expansion and a broad architecture sweep remain deferred until local
measurements establish what extra resources would improve. Reassess cloud use
for training separately from CPU-heavy self-play.
