# TODO Tickets

This folder is the project backlog. Each ticket should describe one bounded
problem, its current status, and observable acceptance criteria.

Status values:

- `Open`: ready to work.
- `In progress`: actively being changed.
- `Needs human input`: requires gameplay judgment or an example from a person.
- `Deferred`: valid work that is intentionally postponed.
- `Done`: acceptance criteria have been verified.

## Index

| ID | Ticket | Priority | Owner | Status |
| --- | --- | --- | --- | --- |
| 001 | [Robust capture comparison](001-robust-capture-comparison.md) | High | Developer | Open |
| 002 | [Validate setup positions](002-validate-setup-positions.md) | High | Developer | Done |
| 003 | [Restore setup state on Undo](003-restore-setup-state-on-undo.md) | Medium | Developer | Done |
| 004 | [Permanent rule regression suite](004-rule-regression-suite.md) | High | Developer | In progress |
| 005 | [Contour topology diagnostics](005-contour-topology-diagnostics.md) | Medium | Developer | In progress |
| 006 | [Repetition and ko policy](006-repetition-and-ko.md) | Low | Human + Developer | Needs human input |
| 007 | [Geometry performance](007-geometry-performance.md) | Low | Developer | Deferred |

Human gameplay investigations live in [`human/`](human/README.md). Findings
from those tickets should be linked back to the relevant main ticket.
