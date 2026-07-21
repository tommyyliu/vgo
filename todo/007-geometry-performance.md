# 007: Geometry Performance

- Status: Deferred
- Priority: Low
- Owner: Developer

## Problem

The current implementation favors clear analytic geometry over asymptotic
performance. Candidate construction, validation, and repeated free-distance
queries are roughly cubic in the number of stones. Capture resolution also
rechecks after removals even though removal monotonicity proves that no new
surviving group can become settled.

## Acceptance criteria

- Add repeatable timing fixtures for representative radii and stone counts.
- Profile before selecting an optimization.
- Remove the redundant capture-resolution loop or document why it remains.
- Cache reusable free-set and group data within one move transaction.
- Consider spatial indexing only after measurements identify the dominant work.
- Preserve every rule-regression fixture from ticket 004.

