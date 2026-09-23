# Search RAM scaling (2026-09-12)

The second memory pass packs `FineGrid` legality into `u64` words and trims
unused snapped-placement vector capacity. A 128×128 mask falls from 16 KiB to
2 KiB. Coordinates, logits, and sampling probabilities retain their precision.
This is additional to the earlier shared-logit change in [MCTS memory](MCTS_MEMORY_LAB.md).

## Process measurements

`crates/vgo-raster/examples/ram_scaling.rs` holds every actor's tree alive at a
barrier while reading Linux VmRSS/VmHWM. A fresh process runs each observation.
The baseline has the earlier shared logits, but neither new memory change.
Three alternating before/after samples in each of 21 configurations produced
126 successful process runs. All full-search-result fingerprints matched.

Median peak process RAM after the changes, MiB:

| Actors | 350 simulations | 700 simulations | 1,400 simulations |
| ---: | ---: | ---: | ---: |
| 1 | 26.8 | 49.7 | 95.8 |
| 8 | 192.1 | 376.3 | 744.3 |
| 16 | 381.4 | 749.8 | 1,486.0 |
| 32 | 759.5 | 1,496.9 | 2,969.1 |

At 2,800 simulations, 1/2/4 actors used 188.7/374.4/745.4 MiB.
At 32×1,400, peak RAM fell from 3,077.6 to 2,969.1 MiB: another 108.5 MiB
saved (3.5%). Across the sweep the additional reduction is roughly 3–5%.
The measured shape is approximately actors × simulations, around 70 KiB per
retained node on this fixture. This is a workload model, not an upper bound.

The fixture uses a 20-stone radius-1/38 board, dense synthetic 128×128 policies,
leaf batch 4, coarse pool 16, widening coefficient 6, and 321 maximum candidates.
It includes real MCTS trees, process/allocator overhead and simultaneous actor
threads, but **not** neural inference, raster tensors, replay writing, training,
or support caches. It does not establish a safe RAM limit for the full generator.

## Reproduce and safety

Build the example in separate baseline/current workspaces and target directories;
the baseline reverses only bit-packing and `placement_overrides.shrink_to_fit()`.

```sh
cargo build --release -p vgo-raster --example ram_scaling
python3 runs/ram-scaling.py BASELINE_BINARY target/release/examples/ram_scaling --samples 3
python3 runs/ram-scaling.py BASELINE_BINARY target/release/examples/ram_scaling --samples 3 --large-only
```

The probe rejects estimates above 8 GiB or one quarter of available system RAM.
The runner caps virtual address space at 8 GiB (the initial small sweep used
6 GiB), with a 180-second process timeout. These are diagnostic safeguards,
**not production memory admission control**. No run exhausted its limit.

Raw samples: [initial sweep](../diagnostics/ram-scaling-2026-09-12.csv),
[16/32 actors](../diagnostics/ram-scaling-large-2026-09-12.csv), with matching
`.log` files recording each successful parity check. Timing columns are
secondary; these short fresh-process measurements primarily establish RAM use.
