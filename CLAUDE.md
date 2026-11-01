# Notes for agents

Start with `README.md`, then `docs/OVERVIEW.md`. The work list is `dashboard/ideas.json`.

## Rules for changes

- Run `correctness/check.sh` before calling work done. It takes about a minute.
- Engine output must not change unless that is the point of the change. A golden mismatch
  needs an explanation in the commit before `check.sh --update-golden`.
- Every engine shortcut needs a written lemma in `spec/LEMMAS.md` and a `[ID]` citation in the
  code. Tests and game comparisons can refute a lemma. They never prove one, so don't call work
  solid on their evidence. A shortcut that can't be proved is listed there as open.
- When claiming a test protects something, inject the bug and watch the test fail. Harnesses must
  pass on clean code first, and must distinguish a failed test from a tool that didn't run.
- The repo is public. Scan before pushing. Commit and push only when asked.
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
- Never run timings alongside anything else, and bound every background job and wait.
- Kill processes by exact PID. `pkill -f` and grep loops match their own shell.

## Current cost, 128 raster, 2026-10-31

| Path | Big board | Small board |
|---|---|---|
| Exact settled test | 25% | 16% |
| Distance transform and legal sampling | 10% | 20% |
| Nearest-stone sweep | 12% | 13% |
| Fine grid | 10% | 17% |
| Liveness witnesses | 10% | low |
| Legal vertices | 9% | low |

Bigger nets are closed in `docs/FINDINGS.md`. Tree reuse is parked in `dashboard/ideas.json`.
