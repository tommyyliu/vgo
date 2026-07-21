# Python Training

This directory will contain model definition, inference serving, replay input,
and optimization code. It deliberately has no Rust package dependency.

The Python inference service communicates with the Rust self-play executable
through the versioned batch protocol described in
[`docs/SELFPLAY_ARCHITECTURE.md`](../docs/SELFPLAY_ARCHITECTURE.md). Training
reads immutable replay shards written by Rust.

No simulator or game-rule implementation belongs here.
