# 006: Repetition and Ko Policy

- Status: Needs human input
- Priority: Low
- Owner: Human + Developer

## Current state

- [x] Two consecutive passes end the game.
- [x] Final scoring uses the current Voronoi cell areas.
- [x] The final result is displayed and reversible through Undo.
- [x] Establish that reachable repetition cycles exist under automated play.
- [ ] Establish whether a strategically meaningful human-play cycle exists.
- [ ] Decide whether unrestricted repetition, simple ko, or positional superko
      belongs in the rules.

The initial Rust self-play canary repeatedly found cycles through legal
self-captures. These were exact full-position repetitions, not contour or hash
artifacts: the arena canonicalizes stone order and retains absolute colors,
player to move, radius, phase, and pass count when recognizing them. For
benchmarking only, the agent skips a ranked move that repeats an earlier state
and tries the next suggestion; this is an action-selection policy, not a game
rule.

Mutual cuts still appear substantially harder to construct than in ordinary
Go. A ko restriction should not be added only by analogy or because a synthetic
search agent can stall through disposable self-captures. The decision should
follow a gameplay example or a convincing argument that the synthetic loop is
itself strategically relevant.

## Human investigation

See [human/001-find-ko-scenario.md](human/001-find-ko-scenario.md).

## Acceptance criteria

- The human investigation records either a strategically credible reproducible
  cycle or the explored families that failed to produce one.
- Any proposed rule states which notion of position repeats: stones and colors,
  player to move, radius, and any other relevant state.
- The policy explains its gameplay benefit rather than relying only on standard
  Go precedent.
- [`reference/RULES.md`](../reference/RULES.md) and regression tests are updated
  if a restriction is adopted.
