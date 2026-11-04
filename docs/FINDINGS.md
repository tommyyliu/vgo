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
| Three inference lanes on the current net | No gain over whole games: 18.5k positions/s against 18.6k at 128, 7.3k against 7.4k at 192. Hung at 256, 2 runs of 2. A net 1.6x faster or more does gain, +15% in 140 s early-game runs at 128. The lighter net then ran 15 minutes on three lanes at 128 with no failures, at 33.8k positions/s. At 192 three lanes lose: 13.5k against 16.4k, because each session slows from 2.8 ms to 4.8 ms a batch |
| Maximum batch 64 | Slower: 16.8k positions/s against 18.8k at 32, two lanes at 128. The card runs near its compute limit at 32 |
| Vectorized bounded transform in place of the envelope | Big boards flat, small boards 25% slower. A row's reach is set by its farthest pixel, which on a small board is 40 offsets or more |
| Fixed-batch TensorRT engines | +0.9% and -1.2% against a batch 1 to 32 profile, so the graph's `Shape` nodes cost nothing |
| GroupNorm lowered through pooling | 17% slower than the exported `InstanceNormalization`, for identical results |
| Depthwise, inverted and grouped blocks | Worst speed for their arithmetic: inverted -24%, grouped -20%, depthwise +10% on 61% of the arithmetic. Fewer plain blocks buy more |
| One resignation threshold for every board and stage | Held its fallback on 23 of 24 picks across three runs. At 128 input it was wrong 2.4% of the time on the 18 board and 17.8% on the 38 board, and on the 38 board no threshold or window beat 12%. Stage mattered far more: 23.2% in the first 15% of the cap, 1.8% past halfway. Replaced by thresholds by board and stage |
| Hard resignation | Unused: all 10,580 games in r128-s2000 read `resigned: false`. Removed, so every game reaches a real result |
| PixelShuffle policy upsampling | No speed gain, and +0.073 validation policy KL after 8 epochs against a seed spread of 0.003. Taking it out of two stacked shapes moved them from +0.075 to +0.013 and +0.018. Option removed |
| Lower priority for actor threads | No gain pinned to 8 cores: lane wake delay went from 119 to 102 us, and the GPU did not rise. The stall was the broker round trip, now removed by the deck |
| Ownership loss off | Lost 88-112 to update 23 and its twin on the 38 board, where the same window with ownership on scored even; colour-swapped pairs went 10-25, p 0.017. Kept at weight 1.5 |
| Weight decay 0.05 (AdamW) | No change in held-out policy KL or value error on update 23's window, alone or with value thinning; the thinning carried that arm's gain |
