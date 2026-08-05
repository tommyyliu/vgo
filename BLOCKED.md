# Blocked: the targets are not in the shards

This branch has the training half only. It cannot run until generation records
per-ply root search values.

`crates/vgo-selfplay/src/generate_demo.rs` writes one value per sample, the
game's terminal utility, and says so itself:

    The root values are discarded after a playout, so this cannot be
    recovered later; it has to be measured here.

## What finishing it needs

1. Record the root search value at each ply during the playout.
2. Carry it through `replay_stream.rs` into the shard record.
3. Bump the replay version; `dataset.py` synthesizes missing fields for older
   versions, so a mixed window keeps loading.
4. Compute horizon targets at preparation and mask samples that lack them, the
   way `ownership_loss` already masks shards written before `final_stones`.

Steps 1-3 are Rust and change the shard format, which is why this is the most
expensive of the five branches despite the training code being small.

## What is here

`short_term_value.py` plus tests, written against synthetic values. It defines
the target exactly -- exponentially weighted future root values at horizons 6,
16 and 50 -- so the Rust side has an unambiguous specification to write against.
