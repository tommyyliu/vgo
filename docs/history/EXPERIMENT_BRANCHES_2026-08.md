# Experiment worktrees

One worktree per candidate change, all branched from the same commit so their
results are comparable. `.worktree/` is gitignored; the branches are real.

The shared motivation is the plateau measured 2026-08-04 on `ddrnet-packed`:
updates 12, 17 and 22 are pairwise indistinguishable over 64 games each, yet
the same model at 1024 simulations beats itself at 512 by 5-1. Search finds
improvements the training loop does not absorb. Ranked by how directly each
change addresses that.

| branch | idea | why |
|---|---|---|
| `experiment/policy-surprise-weighting` | oversample positions where MCTS most contradicted the policy prior | targets the failure mode exactly; KataGo calls it one of their larger training gains |
| `experiment/short-term-value-targets` | auxiliary heads for MCTS value at ~6/16/50 turns | lower-variance feedback for the value head, which is where our metrics move most |
| `experiment/optimistic-policy` | second policy head weighted toward better-than-predicted outcomes | KataGo measures 40-90 Elo, and it is test-time only so it cannot destabilise a run |
| `experiment/global-pooling` | mean, size-scaled mean and max into the heads | cheap global context; we have no equivalent |
| `experiment/context-attention` | transformer blocks in the context branch | already wired behind `context_attention_blocks`; needs pipeline plumbing and an arena |

Details and sources: `vgo-katago-methods-worth-porting` in the session memory.

**Judge these in the arena, not on training metrics.** This session watched
`ddrnet-komi3` improve policy KL and value MAE monotonically through update 34
while its playing strength went flat, and `ddrnet-packed` do the same. A
validation win is a reason to run an arena, not a result.
