# Backlog

Current priorities are in [`DIRECTION.md`](../DIRECTION.md). This is everything else. Finished
tickets were removed on 2026-10-31 and remain in git history.

| Item | State | Done when |
|---|---|---|
| Rule regression fixtures | In progress | Capture cases have reproducible positions: nonlocal capture through pocket closure, disconnected self-capture, opponent removal reviving a group, and simultaneous captures. Fixtures run from one command and are shared with Rust |
| Browser tests in `check.sh` | Open | `js-app/tests` run headless on Linux. Today they need PowerShell and Windows Chrome |
| Game-grouped validation | Open | All positions from one game land in one split, deterministically, tested with overlapping game IDs across shards. Arena games still decide promotion, so this only affects diagnostics |
| Deterministic parallel arena | Open | Decide between bitwise repeatability and bounded stability, then show identical results across repeated fixed-seed runs |
| Incremental legal vertices | Deferred | Reuse the parent's vertex list with identical output. Vertex order makes this delicate |
| Repetition and ko policy | Needs human input | Decide between unrestricted repetition, simple ko and superko. Automated play reaches cycles, but it's unknown whether a strategically meaningful human cycle exists |
| Sensible ko scenario | Needs human play | A concrete position that players would rationally repeat, or a list of failed attempts |
| Lint and format | Open | `cargo fmt` applied and clippy clean, so `check.sh` enforces both |
| Vgo learnings | Open | Results that exist only in `artifacts/old/` are written up. Required before that folder is deleted |

Rule variants worth trying are parked in
[`GAMEPLAY_VARIANTS.md`](history/GAMEPLAY_VARIANTS.md).
