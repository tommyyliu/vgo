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

## Policy coordinates

The network emits `height * width` placement logits and one final pass logit.
A continuous candidate maps to the containing pixel. MCTS visits from candidates
that share a pixel are added together.

Sampled search does not imply that unvisited pixels are bad moves. Every dataset
record therefore contains both a policy target and a policy mask. The loss takes
a softmax only over sampled candidate pixels and pass. This preserves the
meaning of progressive widening while keeping the policy head raster-native.

## Dataset format

Version 2 files begin with this little-endian header:

```text
8 bytes  magic: VGODATA1
u32      version: 2
u32      samples
u32      channels
u32      height
u32      width
u32      policy size
```

Each record then contains contiguous little-endian `f32` arrays:

```text
state[channels * height * width]
policy_target[height * width + 1]
policy_mask[height * width + 1]
current_player_terminal_value[1]
```

The Python loader verifies the exact file size, finite values, binary masks,
policy normalization, and target/mask agreement before exposing tensors.

## Diagnostics

RGB diagnostics are calculated from `SemanticRaster` channels, never by a
second position renderer. The generator writes enlarged 24-bit BMP files:

- `sample-NNN-overview.bmp` composites territory, legal clearance, ridges, and
  stones;
- `sample-NNN-CC-channel_name.bmp` renders every individual channel using a
  scale-aware color map.

Generated demo data lives under `artifacts/raster-demo/` and is intentionally
ignored by Git. Reproduction commands are in [`../training/README.md`](../training/README.md).
