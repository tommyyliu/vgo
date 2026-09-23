# 007: Geometry Performance

- Status: In progress (benchmark lab and initial experiments available)
- Priority: Low
- Owner: Developer

## Problem

The engine has optimized clipping and legal-set construction, but still
rebuilds geometry and repeatedly searches for survival witnesses. The detailed
review, reproducible benchmark commands, initial results, and proposed
incremental design are in [`BOARD_ITERATION_LAB.md`](../docs/research/BOARD_ITERATION_LAB.md).
The follow-up [`SUPPORT_CERTIFICATES.md`](../docs/research/SUPPORT_CERTIFICATES.md) records
the stable-support theorem, dormant-point prototype, negative-cell cache, and
the next engineering steps. These experiments remain opt-in.
The [`MCTS_MEMORY_LAB.md`](../docs/research/MCTS_MEMORY_LAB.md) follow-up adds tree-memory
accounting and a lab-only compact undo prototype. Shared immutable policy logits
are enabled in production; incremental geometry and compact MCTS nodes are not.
Support certificates are now wired into an opt-in bounded MCTS/self-play backend;
see [`SUPPORT_SEARCH_INTEGRATION.md`](../docs/research/SUPPORT_SEARCH_INTEGRATION.md).

Capture resolution already uses exactly two simultaneous removal stages.
Reanalysis after opponent removal is needed to discover revived friendly
groups; there is no redundant fixed-point removal loop to eliminate.

## Acceptance criteria

- Add repeatable timing fixtures for representative radii and stone counts.
- Profile before selecting an optimization.
- Remove the redundant capture-resolution loop or document why it remains.
- Cache reusable free-set and group data within one move transaction.
- Consider spatial indexing only after measurements identify the dominant work.
- Preserve every rule-regression fixture from ticket 004.
