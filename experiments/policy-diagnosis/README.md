# Why the policy head does not train (2026-07-24)

The RL loop improves value steadily and leaves policy flat. These experiments
narrow down why. All were run on `decoupled-run2` replay: 96x96 raster, 32x32
placement grid, radius 1/18 (9-across board), 256 simulations.

## The observation

Iteration 2's training log, 80 epochs on identical batches and gradients:

| metric | start | end |
| --- | ---: | ---: |
| `policy_kl` | 1.784 | 1.790 |
| `value_mae` | 0.545 | 0.357 |

Value improves 35%. Policy moves in the third decimal.

## What was ruled out

**Not epoch-limited.** `600-epoch-probe.log`: KL bottoms at 1.798 by epoch 100
and never improves through epoch 600. `best_epoch` had been pinning at the
budget every iteration, which looked like truncation; it was not.

**Not an inconsistent target.** `overfit.py` trains 32 positions with policy
loss only and no augmentation: KL reaches ~1e-4 and top-1 97%, with zero
duplicate boards. The targets are individually learnable and mutually
consistent, so the plateau is not label noise in the "same input, contradictory
label" sense. See `overfit_loss.png`.

**Not capacity or optimization.** `bigfit.py` on 2048 positions shows a textbook
overfitting curve -- train KL 2.54 -> 0.57 while held-out KL rises 2.14 -> 2.87,
diverging from epoch 10. The network has ample capacity; it memorizes.

## What the evidence supports

A generalization failure driven by target sparsity.

| positions | train KL | held-out KL |
| ---: | ---: | ---: |
| 32 | ~0.0001 | (memorized) |
| 2,048 | 0.566 | 2.87 (diverging) |
| ~12,000 | 1.79 | 1.79 (plateau) |

Reference points on the same data: a predictor emitting uniform over all legal
cells scores KL 3.68; one emitting uniform over just the explored support scores
0.56. The trained net sits at 1.79 -- it has learned roughly *where* candidates
land (a board-dependent, learnable envelope) and none of the relative weights
among them.

`input_vs_target.png` shows why. The input is a rich, structured 96x96 field:
clean Voronoi ownership, sharp ridges, well-formed legality. The target is one
bright cell plus a few specks over 1024, with a typical maximum mass of 0.36 on
a single cell. Nothing in the input distinguishes the chosen cell from its
neighbours; the difference came from which candidates the sampler drew and how
256 simulations split among them. `policy_maps.png` shows the same thing across
four positions, with the overfit prediction beside each target.

Note this is *not* the failure the coarse-to-fine redesign was built to fix.
That one was candidate support relocating between games, measured as ply-0
candidate Jaccard, and it did improve: 0.002 -> 0.085 across the decoupling and
three RL iterations. Support stability improved 40x and the policy still did not
learn, because per-sample sparsity is a separate problem.

## Untested directions

- **Smooth the target spatially** before the softmax. Adjacent 32x32 cells are
  1/32 apart while a stone is 1/9 across, so neighbours are nearly the same
  move; treating them as independent classes discards the structure a
  convolution is good at. Cheapest test, no replay regeneration.
- **More data.** The curve from 2k to 12k is still moving, and generation is now
  ~2x faster. Likely worth tenths of a nat rather than the ~1.8 needed.
- **A smaller board.** At 9-across, ~33 proposal draws cover ~2% of the ~570
  legal cells. A 5-across board would make the target dense relative to the
  action space and validate the pipeline where the problem is easy.
- **Ridge-aware policy parameterization.** Good moves plausibly relate to the
  Voronoi structure visible in channel 6; a policy over regions rather than raw
  cells might be predictable where a cell index is not.

## Reproducing

```bash
cd training
uv run --with matplotlib python ../experiments/policy-diagnosis/overfit.py 32 3000
uv run --with matplotlib python ../experiments/policy-diagnosis/bigfit.py
uv run --with matplotlib python ../experiments/policy-diagnosis/render_inputs.py
```

Each script hardcodes its shard paths; edit them for other replay.

Images (too large for git) are in `artifacts/policy-diagnosis/`:
`input_channels.png`, `input_vs_target.png`, `policy_maps.png`, `overfit_loss.png`.
