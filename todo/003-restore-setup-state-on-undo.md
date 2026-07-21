# 003: Restore Setup State on Undo

- Status: Done
- Priority: Medium
- Owner: Developer

## Problem

History snapshots store stones, turn, consecutive passes, and game-over state,
but not the radius or contour detail. Loading a diagram can change the radius;
Undo then restores the old stones under the loaded radius.

The load path also applies the incoming radius before pushing its history entry.

## Acceptance criteria

- A snapshot contains every setup value that Undo promises to restore.
- Loading records the complete pre-load state before applying imported values.
- Undo after Load restores stones, turn, radius, pass count, game-over state, and
  relevant control labels.
- Radius-slider history behavior is explicitly chosen and documented.
- Tests cover Load, Undo, and Undo after a concluded game.

## Resolution

History stores immutable whole positions, including radius, pass count, and
phase. Load pushes the old position before applying the new one, and Undo
resynchronizes the radius control. Contour depth is presentation state and is
not changed by loading or undone. Dragging the radius slider is intentionally
one continuous setup edit and does not create history entries; the next move,
pass, clear, or load records the resulting radius. Browser tests cover Load,
Undo, and reopening a concluded game.
