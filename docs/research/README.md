# Research

Write-ups of geometry and search-performance research: exact locality bounds
for contestability, support certificates, the board-iteration lab, the settled
region problem and Voronoi construction.

The lab code these notes describe was removed from the working tree on 2026-10-31,
since production never used it. It is on the `archive/iteration-labs` branch.

Some reproduction commands here name drivers that the 2026-09-23 prune removed:
`vgo-canary`, the `runs/*` measurement scripts, and the `ram_scaling` and
`vgo-raster-bench` tools. They are on the `archive/pre-prune` branch. The
measurements are still valid for the code they were taken on.

Raw `diagnostics/` files these notes link to were never tracked and no longer exist.
Their measurements are recorded in the notes themselves.

| doc | question |
|---|---|
| [ROADMAP_2026-10.md](ROADMAP_2026-10.md), [PROJECT_OUTLINE_2026-10.md](PROJECT_OUTLINE_2026-10.md) | the plan and idea outline as of the restart, frozen |
| [EXPERIMENT_BRANCHES_2026-08.md](EXPERIMENT_BRANCHES_2026-08.md) | why each experiment branch exists |
| [TREE_REUSE_2026-10-14.md](TREE_REUSE_2026-10-14.md) | do existing tails need filtering, can MCTS retain useful work, and do native failures return within a deadline? |
| [RESTART_2026-10-13.md](RESTART_2026-10-13.md) | what does the current GPU/CPU profile show, and does indexed radial enumeration help? |
| [RESTART_2026-10-14.md](RESTART_2026-10-14.md) | production-scale pruning, matched model variants, and a randomized search-budget pilot |
| [LOCAL_CONTESTABILITY.md](LOCAL_CONTESTABILITY.md) | how far away can a stone change another stone's status? |
| [LOCAL_CONTESTABILITY_SHARP_BOUND.md](LOCAL_CONTESTABILITY_SHARP_BOUND.md) | the (2 + 4 sqrt 2) r upper bound |
| [LOCAL_CONTESTABILITY_FLOOR.md](LOCAL_CONTESTABILITY_FLOOR.md) | the 7.4918 r lower bound |
| [BOARD_ITERATION_LAB.md](BOARD_ITERATION_LAB.md) | measured board-transition algorithms |
| [SUPPORT_CERTIFICATES.md](SUPPORT_CERTIFICATES.md), [SUPPORT_SEARCH_INTEGRATION.md](SUPPORT_SEARCH_INTEGRATION.md), [DENSE_SUPPORT_THROUGHPUT.md](DENSE_SUPPORT_THROUGHPUT.md) | stable support certificates and their cost in search |
| [NAIVE_SELFPLAY_THROUGHPUT.md](NAIVE_SELFPLAY_THROUGHPUT.md), [MCTS_MEMORY_LAB.md](MCTS_MEMORY_LAB.md), [RAM_SCALING.md](RAM_SCALING.md) | search memory and throughput |
| [SETTLED_REGION_PROBLEM.md](SETTLED_REGION_PROBLEM.md) | the raster's most expensive plane |
| [VORONOI_CUTOFF.md](VORONOI_CUTOFF.md) | Voronoi cells by sorted insertion with a cutoff |
