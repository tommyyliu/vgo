# Findings

What the project has learned, and how results get decided. Dated detail is in
[`history/`](history/README.md).

## How results are decided

1. **Adoption.** A change is adopted on a multi-seed, matched-compute arena. Loss curves and
   small pilots don't decide it.
2. **Shortcuts.** An engine shortcut needs a lemma in [`LEMMAS.md`](../spec/LEMMAS.md), or a
   stated approximation listed there as open. Tests can refute a lemma and can't prove one.
3. **Results.** Every experiment ends with a written verdict here. A result that exists only
   under `artifacts/` isn't recorded.

## Closed ideas

Don't reopen these without new evidence.

| Idea | Verdict |
|---|---|
| Warm-started incremental training | Lost 41-7 to a from-scratch model on the same corpus |
| Muon optimizer | Led early A/Bs, overtaken by update 24 |
| Bigger nets at this compute | Dead end. One doubling of simulations is worth about 61 Elo |
| Tanh value head with MSE | Gradient vanished on confidently wrong positions |
| Random-candidate sampler | Target carried no board-dependent signal |
