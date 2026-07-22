# Deterministic parallel arena

## Problem

FP16 inference can change a nearly tied policy ordering when concurrent requests
form different batch shapes. This makes a multi-actor arena faster but not fully
repeatable.

## Completion criteria

- Define whether promotion requires bitwise repeatability or bounded numerical
  stability.
- Make batch composition deterministic, or define a tie policy robust to the
  measured inference error.
- Demonstrate identical promotion results across repeated fixed-seed runs.
- Re-enable multiple promotion actors only after those checks pass.

The RL driver currently uses one arena actor by default, which is repeatable and
keeps this issue outside the promotion path.
