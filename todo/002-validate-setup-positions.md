# 002: Validate Setup Positions

- Status: Done
- Priority: High
- Owner: Developer

## Problem

SGF loading accepts finite coordinates without checking the inset board or
pairwise separation constraints. Changing the radius can also make an existing
diagram unplayable.

Flexible diagram editing is useful, so invalid diagrams do not need to be
silently corrected or discarded. They do need an explicit state.

## Acceptance criteria

- A shared validator checks board inset, pairwise `2r` separation, finite
  coordinates, known colors, and a valid radius.
- An invalid diagram remains viewable but is visibly marked unplayable.
- Placement and passing are disabled while the diagram is unplayable.
- The warning identifies enough information to locate the invalid stones.
- Loading and radius changes invoke the same validator.
- Tests cover out-of-board, overlapping, duplicate, tangent, and valid stones.

## Resolution

`VGO.model.validate()` is shared by analysis, placements, passes, SGF loads,
and radius changes. Invalid diagrams remain rendered with an indexed warning;
game actions are disabled. The engine suite covers every listed position type.
