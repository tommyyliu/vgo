# Backlog

**Goal:** the strongest Voronoi Go bot per hour of compute, reachable through the browser.
`Vgo` rules are primary, and `Official` is a compatibility target. The headline metric is Elo
per hour on fixed hardware against fixed anchors.

## Now

At most three.

| Bet | Question | Done when |
|---|---|---|
| Measure real self-play on rented hardware | Are we still CPU-bound after the engine work? | Generation throughput and GPU use are recorded on the rented box |
| Ownership loss arena | Does sigmago's largest gain carry over? | A multi-seed, matched-compute arena has a verdict |
| Settled plane (O1) | Is its approximation harmless, or should it be exact or removed? | O1 is proved or bounded, or the plane's strength value is measured |

## Later

Finished tickets were removed on 2026-10-31 and remain in git history.

| Item | State | Done when |
|---|---|---|
| Rule regression fixtures | In progress | Capture cases have reproducible positions: nonlocal capture through pocket closure, disconnected self-capture, opponent removal reviving a group, and simultaneous captures. Fixtures run from one command and are shared with Rust |
| Browser tests in `check.sh` | Open | `js-app/tests` run headless on Linux. Today they need PowerShell and Windows Chrome |
| Game-grouped validation | Open | All positions from one game land in one split, deterministically, tested with overlapping game IDs across shards. Arena games still decide promotion, so this only affects diagnostics |
| Deterministic parallel arena | Open | Decide between bitwise repeatability and bounded stability, then show identical results across repeated fixed-seed runs |
| Incremental legal vertices | Deferred | Reuse the parent's vertex list with identical output. Vertex order makes this delicate |
| Repetition and ko policy | Needs human input | Decide between unrestricted repetition, simple ko and superko. Automated play reaches cycles, but it's unknown whether a strategically meaningful human cycle exists |
| Sensible ko scenario | Needs human play | A concrete position that players would rationally repeat, or a list of failed attempts |
| Official vertex reach test | Open | Find a position where the cell-vertex check in `official_cell_is_alive` alone decides a group, or prove the other two checks cover it. A bug there passes every test, and a search of up to 3,000 random games found no such position |
| Lint and format | Open | `cargo fmt` applied and clippy clean, so `check.sh` enforces both |
| Vgo learnings | Open | Results that exist only in `artifacts/old/` are written up. Required before that folder is deleted |

Rule variants worth trying are parked in
[`GAMEPLAY_VARIANTS.md`](history/GAMEPLAY_VARIANTS.md).
