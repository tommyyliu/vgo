# Notes for agents

Start with `README.md`, then `docs/OVERVIEW.md`. The work list is `dashboard/ideas.json`.

## Rules for changes

- Run `correctness/check.sh` before calling work done. It takes about a minute.
- Engine and game output must not change unless that is the point of the change. A golden
  mismatch needs an explanation in the commit before `check.sh --update-golden`.
  `vgo-verify` covers the engine; `correctness/game-digest.sh` covers self-play and arena.
- Every engine shortcut needs a written lemma in `spec/LEMMAS.md` and a `[ID]` citation in the
  code. Tests and game comparisons can refute a lemma. They never prove one, so don't call work
  solid on their evidence. A shortcut that can't be proved is listed there as open.
- When claiming a test protects something, inject the bug and watch the test fail. Harnesses must
  pass on clean code first, and must distinguish a failed test from a tool that didn't run.
- Commit as you go, on a branch rather than `main`. Push only when asked: the repo is public,
  so scan first.
- Verify with tests and tiny runs, one stage at a time. Full RL runs are expensive and prove
  little here, so start one only when asked.
- `artifacts/` holds run data and is not in git. `artifacts/old/` is an archive that needs a
  write-up of its results in `docs/FINDINGS.md` before it is deleted.

## Working on performance

```bash
cargo build --release -p vgo-bench --bin vgo-bench --features profiling --target-dir <scratch>
taskset -c 2 <scratch>/release/vgo-bench --seconds 8 --only search_raster_128 \
    --profile-output out.folded    # add --positions FILE for a single board size
target/release/vgo-verify          # fixture digest; --fuzz-games N for whole games
performance/run.py                 # compare with history and record; needs a quiet machine
```

- Keep a baseline binary from before the change. Build a pristine one with `git archive`.
- Compare `search_*` lines across versions. Fixture paths such as `raster_*` hit caches.
- `perf` is not installed. Profiles attribute time loosely, so confirm a gain with timings.
  Sampled self time misled more than once: put timers around stages in a scratch copy first.
- `vgo-bench --scaling 1,16,32` shows what sharing a machine costs. Check a real generator with
  `--profile-output` too, because a short run only reaches early-game positions.
- Fixture searches run 200 simulations on a fixed policy. Real generation runs 1000, so its
  shares differ: snapping was 9% of a fixture search and 20% of generation. Profile the generator
  before choosing a target. `--profile-output` there writes a flamegraph SVG, and a frame named
  after a closure may be the generic function that closure was passed to.
- To compare generators, use the process's own CPU per position over a late window, from
  `/proc/<pid>/task/*/stat`. Machine-wide busy cores include about a core of everything else, and
  profile shares from runs of different lengths cover different stages of the game.
- Never run timings alongside anything else, and bound every background job and wait.
- Kill processes by exact PID. `pkill -f` and grep loops match their own shell.

## Current cost, 128 raster, 2026-11-01

Share of a fixture search with a fixed policy, so inference is absent.

| Path | Big board | Small board |
|---|---|---|
| Move rules, mainly Voronoi | 44% | 18% |
| Distance transform | 16% | 29% |
| Nearest-stone sweep and fold | 13% | 7% |
| Cheap settled cases | 7% | 11% |
| Candidate grid and sampling | 6% | 11% |
| Exact settled test and vertex check | 2% | 7% |

In real generation the GPU is the limit, not this table, from about 8 vCPU on this card: see
"Sizing a machine" in `docs/RUNNING.md`. Measure over whole games. Early-game runs read about half the
CPU of later boards and overstated what three lanes and 8 cores could do. A run with 64 actors
needs about 7 minutes before games finish.

`nvidia-smi` utilization and busy cores can both sit below 100% with work queued, which means a
hand-off is stalling, not that a resource is short. Broker telemetry shows which: look for
`completion_wait_seconds` near 1 with positions in the channel and the lanes not busy.

Bigger nets are closed in `docs/FINDINGS.md`. Tree reuse is parked in `dashboard/ideas.json`.
