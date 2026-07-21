# 004: Permanent Rule Regression Suite

- Status: In progress
- Priority: High
- Owner: Developer

## Problem

Current verification is performed through focused browser checks, but those
checks are not stored as repeatable project tests. Rule changes can therefore
regress previously discovered gameplay cases.

## Acceptance criteria

- Tests run from one documented command and fail nonzero.
- Tests exercise the actual game functions rather than a separate rewrite.
- Fixtures cover:
  - Nonlocal opponent capture through escape-pocket closure.
  - Disconnected legal self-capture.
  - Opponent removal reviving an otherwise self-captured group.
  - Simultaneous captures.
  - Empty legal-placement space.
  - Exact board, disk, and circle tangencies.
  - Pass/pass scoring, post-game blocking, and Undo.
- The human-found examples are recorded with coordinates or complete move
  sequences so failures are reproducible.

## Progress

`reference/tests/run-tests.ps1` now fails nonzero and exercises the production
modules and the real UI. Current fixtures cover immutable move transactions, empty legal
space, board and disk tangency, point-only cell contact, pass/pass conclusion,
post-game blocking, setup validation, Load, and Undo. The capture-specific
gameplay fixtures above still need reproducible positions.
