# Semantic Raster Representation

## Contract

`vgo-raster` converts an exact `Position` into a player-relative contiguous
`f32` tensor with shape `[channels, height, width]`, where `channels` is set by
the layout. It samples the center of every pixel:

```text
x = (column + 0.5) / width
y = (row + 0.5) / height
```

Row zero is therefore near board `y = 0`, matching the existing browser board
orientation. Absolute Black and White are never encoded; current and opponent
channels swap with the player to move. Reordering stones does not alter the
tensor.

## Channels

`CHANNELS` in `crates/vgo-raster/src/lib.rs` is a **catalogue**, not a layout: it
holds every channel any layout can name, and a layout is a list of indices into
it. That is why `CHANNEL_SPEC_COUNT` (15) is larger than `CHANNEL_COUNT` (12,
the semantic tensor's width) -- growing the catalogue is safe, growing the
semantic width is not, because that number is baked into inference frames and
ONNX profile shapes.

| Index | Name | Range | Definition |
|---:|---|---:|---|
| 0 | `current_stones` | `[0, 1]` | Pixel center lies inside a current-player disk. |
| 1 | `opponent_stones` | `[0, 1]` | Pixel center lies inside an opponent disk. |
| 2 | `current_voronoi` | `[0, 1]` | Current player owns the nearest site; exact opposing ties are `0.5`. |
| 3 | `opponent_voronoi` | `[0, 1]` | Opponent owns the nearest site; exact opposing ties are `0.5`. |
| 4 | `current_distance` | `[0, 1]` | Distance to nearest current stone, divided by `4r` and clipped. |
| 5 | `opponent_distance` | `[0, 1]` | Distance to nearest opponent stone, divided by `4r` and clipped. |
| 6 | `voronoi_ridge` | `[0, 1]` | `1 - (d2 - d1) / r`, clipped; bright where the two nearest sites tie. |
| 7 | `legal_clearance` | `[-1, 1]` | Signed clearance from board inset and all `2r` exclusion disks, divided by `r`. Not the distance to the legal set: it is the slack on the *nearest violated constraint*, so it bounds that distance from below rather than giving it. |
| 8 | `radius` | `(0, 1)` | Constant plane containing normalized diameter `2r`. |
| 9 | `previous_pass` | `[0, 1]` | Constant plane, `1` when the last move was a pass. Two passes end the game, so a live position is only ever at zero or one -- this is the pass state, not a summary of it. |
| 10 | `settled` | `[0, 1]` | This repository's capture predicate: no legal centre can get strictly closer to the point than the stone that owns it. Reads as *can anyone still take this area*. |
| 11 | `komi` | `[-1, 1]` | Constant plane, signed for the side to move. |
| 12 | `dead_zone` | `[0, 1]` | voronoigo.com's capture predicate: `dist(x, L) > r`, where no stone can be placed covering the point. Reads as *can anyone still reach this area*. Strictly contains `settled`; see [`OFFICIAL_RULES.md`](OFFICIAL_RULES.md). |
| 13 | `current_connections` | `[0, 1]` | A one-cell line between each pair of current-player stones that no enemy pair can wedge apart. |
| 14 | `opponent_connections` | `[0, 1]` | The same for the opponent. |

Two of these are constant across the board -- `komi` and `previous_pass` -- and
`radius` is a third. They are planes only because the network is a convolution
stack with no other way to receive a scalar.

The stone and connection planes are **relative to the side to move**, not black
and white, which is why no "whose turn" channel exists.

## Layouts

A layout names its channels by index. `RasterKind::indices()` is the list;
`raster_kind` is configured at training time and travels with the model (see
[`OVERVIEW.md`](OVERVIEW.md)), because a shard stores positions rather than
pictures and the raster is rendered at load.

| Kind | Ch | Channels |
|---|---:|---|
| `semantic` | 12 | the first twelve of the catalogue |
| `rgb` | 3 | the board as a player sees it; no derived fields |
| `compact` | 5 | `current_stones`, `opponent_stones`, `voronoi_ridge`, `settled`, `komi` |
| `compact-pass` | 6 | `compact` + `previous_pass` |
| `compact-dead-zone` | 6 | `compact-pass` with `dead_zone` in place of `settled` |
| `compact-connected` | 9 | both capture fields, both connection planes, and the two scalars |

`compact-pass` and `compact-dead-zone` differ in exactly one slot, which is
deliberate: slot 3 is *the capture predicate*, so a model crosses between
rulesets by reinitialising one input slice, and comparing the two rulesets is a
one-plane A/B rather than a change of representation. A test asserts it.

`compact-connected` breaks that symmetry on purpose. `settled` is the wrong
capture predicate under the official rules, but it is also the only plane that
says which board can still change hands, which is ownership rather than legality
and is worth having under either ruleset.

### Cost

At 128 square, milliseconds per position, one thread:

| layout | mean over 0-52 stones | 28 stones | 52 stones |
|---|---:|---:|---:|
| `semantic` 12ch | 0.70 | 0.81 | 1.27 |
| `compact` 5ch | 0.37 | 0.51 | 0.77 |
| `compact-dead-zone` 6ch | 0.35 | 0.36 | 0.60 |

`settled` and `dead_zone` are two thresholds on one distance field, so a layout
carrying both pays for one transform. `crates/vgo-raster/examples/raster_cost.rs`
is the measurement.

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
