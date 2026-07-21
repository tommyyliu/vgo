# Human 001: Find a Sensible Ko or Repetition Scenario

- Status: Ready
- Priority: Exploratory
- Owner: Human
- Parent: [006: Repetition and ko policy](../006-repetition-and-ko.md)

## Question

Can legal play under [`reference/RULES.md`](../../reference/RULES.md) return to an earlier complete game position in
a way that players would rationally repeat?

A repeated position should initially mean the same radius, stone centers,
colors, and player to move. Record strategically repeating but not exactly equal
positions separately. Do not use passes inside the proposed cycle because two
consecutive passes end the game.

## Deliverable

For each credible example, record:

- The radius and starting VGO-SGF position.
- Every move center in order and the player making it.
- Stones removed after each move, including self-captures.
- The earlier and later states claimed to be equal.
- Why each move is legal and why each capture occurs.
- Whether immediate recapture, a longer cycle, or only a strategic loop is
  demonstrated.

Screenshots are helpful, but coordinates or SGF snapshots are required so the
example can become an automated regression fixture.

If no scenario is found, record the classes of cuts, escape pockets, and
recaptures explored. That evidence will help decide whether unrestricted
repetition is an acceptable deliberate rule.
