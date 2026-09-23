# Research

Write-ups of geometry and search-performance research: exact locality bounds
for contestability, support certificates, the board-iteration lab, the settled
region problem and Voronoi construction. The code lives behind the
`iteration-lab` feature in `vgo-core` and `vgo-search`:

```bash
cargo test --release -p vgo-core -p vgo-search --features vgo-search/iteration-lab
cargo run --release -p vgo-core --features iteration-lab --example iteration_lab -- 10
```

Some reproduction commands here name drivers that the 2026-09-23 prune removed:
`vgo-canary`, the `runs/*` measurement scripts, and the `ram_scaling` and
`vgo-raster-bench` tools. They are on the `archive/pre-prune` branch. The
measurements are still valid for the code they were taken on.

| doc | question |
|---|---|
| [LOCAL_CONTESTABILITY.md](LOCAL_CONTESTABILITY.md) | how far away can a stone change another stone's status? |
| [LOCAL_CONTESTABILITY_SHARP_BOUND.md](LOCAL_CONTESTABILITY_SHARP_BOUND.md) | the (2 + 4 sqrt 2) r upper bound |
| [LOCAL_CONTESTABILITY_FLOOR.md](LOCAL_CONTESTABILITY_FLOOR.md) | the 7.4918 r lower bound |
| [BOARD_ITERATION_LAB.md](BOARD_ITERATION_LAB.md) | measured board-transition algorithms |
| [SUPPORT_CERTIFICATES.md](SUPPORT_CERTIFICATES.md), [SUPPORT_SEARCH_INTEGRATION.md](SUPPORT_SEARCH_INTEGRATION.md), [DENSE_SUPPORT_THROUGHPUT.md](DENSE_SUPPORT_THROUGHPUT.md) | stable support certificates and their cost in search |
| [NAIVE_SELFPLAY_THROUGHPUT.md](NAIVE_SELFPLAY_THROUGHPUT.md), [MCTS_MEMORY_LAB.md](MCTS_MEMORY_LAB.md), [RAM_SCALING.md](RAM_SCALING.md) | search memory and throughput |
| [SETTLED_REGION_PROBLEM.md](SETTLED_REGION_PROBLEM.md) | the raster's most expensive plane |
| [VORONOI_CUTOFF.md](VORONOI_CUTOFF.md) | Voronoi cells by sorted insertion with a cutoff |
