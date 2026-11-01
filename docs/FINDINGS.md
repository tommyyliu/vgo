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
| Three inference lanes | No gain over whole games: 18.5k positions/s against 18.6k at 128, 7.3k against 7.4k at 192. Hung at 256, 2 runs of 2 |
| Maximum batch 64 | Slower: 16.8k positions/s against 18.8k at 32, two lanes at 128. The card runs near its compute limit at 32 |
| Lower priority for actor threads | No gain pinned to 8 cores: lane wake delay went from 119 to 102 us, and the GPU did not rise. The stall was the broker round trip, now removed by the deck |
