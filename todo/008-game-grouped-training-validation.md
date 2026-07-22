# Game-grouped training validation

## Problem

The training/validation split currently operates on individual replay samples.
Adjacent positions from one self-play game can therefore appear on both sides,
making validation loss optimistic.

## Completion criteria

- Track a shard-local game identity through replay concatenation.
- Assign every position from one game to exactly one split.
- Preserve a deterministic split for a fixed training seed.
- Add tests for multiple shards whose numeric game IDs overlap.

Fresh arena games remain the promotion criterion, so this affects diagnostics
rather than model acceptance correctness.
