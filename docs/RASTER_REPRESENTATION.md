# Semantic Raster Representation

## Contract

`vgo-raster` converts an exact `Position` into a player-relative contiguous
`f32` tensor with shape `[10, height, width]`. It samples the center of every
pixel:

```text
x = (column + 0.5) / width
y = (row + 0.5) / height
```

Row zero is therefore near board `y = 0`, matching the existing browser board
orientation. Absolute Black and White are never encoded; current and opponent
channels swap with the player to move. Reordering stones does not alter the
tensor.

## Channels

| Index | Name | Range | Definition |
|---:|---|---:|---|
| 0 | `current_stones` | `[0, 1]` | Pixel center lies inside a current-player disk. |
| 1 | `opponent_stones` | `[0, 1]` | Pixel center lies inside an opponent disk. |
| 2 | `current_voronoi` | `[0, 1]` | Current player owns the nearest site; exact opposing ties are `0.5`. |
| 3 | `opponent_voronoi` | `[0, 1]` | Opponent owns the nearest site; exact opposing ties are `0.5`. |
| 4 | `current_distance` | `[0, 1]` | Distance to nearest current stone, divided by `4r` and clipped. |
| 5 | `opponent_distance` | `[0, 1]` | Distance to nearest opponent stone, divided by `4r` and clipped. |
| 6 | `voronoi_ridge` | `[0, 1]` | `1 - (d2 - d1) / r`, clipped; bright where the two nearest sites tie. |
| 7 | `legal_clearance` | `[-1, 1]` | Signed clearance from board inset and all `2r` exclusion disks, divided by `r`. |
| 8 | `radius` | `(0, 1)` | Constant plane containing normalized diameter `2r`. |
| 9 | `previous_pass` | `[0, 1]` | Constant plane indicating one immediately preceding pass. |

The exact simulator remains authoritative. The raster is the single
model-facing state used for training, replay, inference, policy coordinates,
and diagnostics.

Raster resolution is independent of stone radius. The active throughput canary
uses radius `1/6`, which has a small 3x3-like effective game size, sampled at
128x128. This deliberately exercises the memory and convolution pipeline at a
resolution representative of larger future games without increasing the early
self-play search space.

## Numeric precision

The raster does not need `f32` storage precision. Channels 0-3 and 9 are
binary/ternary in practice, channels 4-6 and 8 are bounded in `[0, 1]`, and
channel 7 is bounded in `[-1, 1]`. A channel-aware one-byte encoding can use
unsigned fixed-point for unit channels and signed fixed-point for legal
clearance, with worst-case reconstruction errors of about `0.00196` and
`0.00394`, respectively.

On the 96-position 128x128 canary, FP16 round-trip preserved the model's sampled
top action on all positions. FP8 E4M3 and channel-aware fixed 8-bit each
preserved 94 of 96, but fixed 8-bit produced 9.6 times less mean policy-logit
change and 19 times less mean value change than FP8. E5M2 and fixed 4-bit were
materially worse. These are sensitivity results for one overfit canary, not a
final gameplay validation; adoption still requires training on the encoded
form and comparison on held-out replay and self-play outcomes. Reproduce the
measurement with `vgo_training.benchmark_precision`.

## Policy coordinates

The network emits `policy_resolution^2` placement logits and one final pass
logit. A continuous candidate maps to the containing policy cell. MCTS visits
from candidates that share a cell are added together.

**The placement grid is independent of the render resolution.** They were once
the same number; they answer different questions. The raster resolution controls
how much geometric detail the convolution tower sees -- channels 2, 3, 6, and 7
are continuous fields whose *edges* carry the information, and coarsening them
aliases the Voronoi ridge away. The placement resolution controls how finely a
move can be aimed, and a board roughly nine stones across does not need 16384
distinct placements.

Keeping them equal was actively harmful. Progressive widening draws
`K = min(96, ceil(2*sqrt(N+1)))` proposals -- 33 at 256 simulations. Spread over
a 128x128 grid, 33 draws essentially never land on the same cell twice, so the
sampled candidate set relocates every game and the policy target has no stable
support. Over 32x32 the same 33 draws revisit cells 21% of the time. Measured
ply-0 candidate overlap rose from `0.002` to `0.034` on that change alone.

Set `--policy-resolution` at generation, arena, and duel time; it is stored in
the checkpoint and frozen into the exported ONNX metadata as `vgo.policy_size`.
The replay header records it, and training derives the model's policy head from
the replay rather than from a flag. The full-legal training mask is max-pooled
from the raster's clearance channel down to the placement grid: a policy cell is
legal if *any* raster pixel inside it is legal, because a cell containing one
playable point is a playable move.

Sampled search does not imply that unvisited pixels are bad moves. Every current
replay record therefore contains the sampled-candidate mask, raw visit counts,
the coarse-to-fine proposal probability beta, and the raw proposal multiplicity
for each policy cell. Training derives the full legal raster mask from
`legal_clearance >= 0`, adds pass, and unions sampled boundary aliases. The loss
takes its softmax over that full legal mask while its corrected target has
nonzero mass only on visited candidates. Unexplored legal cells consequently
receive the denominator's negative signal; illegal cells receive no gradient.

For a sampled placement, the sparse target's unnormalized mass is
`visits * proposal_count / (K * beta)`, where `K` is the sum of placement
proposal counts in that row. Deterministic pass retains its raw visit mass.
Rows from older schemas or the non-spatial fallback have zero proposal counts
and use normalized raw visits. Corrected targets and legal masks are prepared
once on CPU and cached for both training and metrics.

## Dataset format

All binary datasets begin with this little-endian header. Legacy tensors use
magic `VGODATA1` and version 2; replay shards use `VGORPLY1` and replay version
1, 2, or 3.

```text
8 bytes  magic
u32      version
u32      samples
u32      channels
u32      height
u32      width
u32      policy size
```

Current replay shards use magic `VGORPLY1`, replay version 3, and then contain:

```text
state[channels * height * width]
policy_target[height * width + 1]
sampled_candidate_mask[height * width + 1]
raw_visits[height * width + 1]
sampling_beta[height * width + 1]
proposal_counts[height * width + 1]   # little-endian u32
current_player_terminal_value[1]
selected_action[u32]
game[u64]
ply[u32]
seed[u64]
```

Replay version 1 omits `raw_visits`, `sampling_beta`, and `proposal_counts`.
Version 2 adds visits and beta but has no proposal counts. The loader synthesizes
visits from normalized policy when necessary and uses zeros for unavailable beta
or counts, so replay windows may span all three versions. The Python loader
verifies the exact file size, finite/nonnegative visits, beta bounds, count/beta
support agreement, deterministic-pass zeros, binary masks,
normalized-visits/policy agreement, and selected-action consistency before
exposing tensors.

## Diagnostics

RGB diagnostics are calculated from `SemanticRaster` channels, never by a
second position renderer. The generator writes enlarged 24-bit BMP files:

- `sample-NNN-overview.bmp` composites territory, legal clearance, ridges, and
  stones;
- `sample-NNN-CC-channel_name.bmp` renders every individual channel using a
  scale-aware color map.

Generated demo data lives under `artifacts/raster-demo/` and is intentionally
ignored by Git. Reproduction commands are in [`../training/README.md`](../training/README.md).
