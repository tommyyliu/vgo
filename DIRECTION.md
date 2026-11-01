# Direction

Reviewed 2026-10-31. This file is short on purpose. If it grows past a page, cut it.

## Goal

The strongest Voronoi Go bot per hour of compute, reachable by people through the browser
client. `Vgo` rules are primary, and `Official` is a compatibility target.

**Headline metric:** Elo per hour on fixed hardware, measured against fixed anchors.

## Active bets

At most three. Anything else waits in [`docs/BACKLOG.md`](docs/BACKLOG.md).

| Bet | Question | Done when |
|---|---|---|
| Measure real self-play on rented hardware | Are we still CPU-bound after the engine work? | Generation throughput and GPU use are recorded on the rented box |
| Ownership loss arena | Does sigmago's largest gain carry over? | A multi-seed, matched-compute arena has a verdict |
| Settled plane (O1) | Is its approximation harmless, or should it be exact or removed? | O1 is proved or bounded, or the plane's strength value is measured |

## Rules for the work

1. **Adoption.** A change is adopted on a multi-seed, matched-compute arena. Loss curves and
   small pilots don't decide it.
2. **Shortcuts.** An engine shortcut needs a lemma in [`spec/LEMMAS.md`](spec/LEMMAS.md), or a
   stated approximation listed there as open. Tests can refute a lemma and can't prove one.
3. **One command.** `scripts/check.sh` passes before anything merges.
4. **Results.** Every experiment ends with a written verdict. A result that exists only under
   `artifacts/` isn't recorded.

## Closed, with evidence

Don't reopen these without new evidence.

| Idea | Verdict |
|---|---|
| Warm-started incremental training | Lost 41-7 to a from-scratch model on the same corpus |
| Muon optimizer | Led early A/Bs, overtaken by update 24 |
| Bigger nets at this compute | Dead end. One doubling of simulations is worth about 61 Elo |
| Tanh value head with MSE | Gradient vanished on confidently wrong positions |
| Random-candidate sampler | Target carried no board-dependent signal |
