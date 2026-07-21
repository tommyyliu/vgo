# ADR 0001: Native Simulator Backend

- Status: Proposed
- Date: 2026-07-21

## Context

Self-play will execute substantially more state transitions and analyses than
the interactive reference. Training also needs batched state encoding or
rasterization into contiguous numeric buffers. Keeping those loops in Python
would make interpreter overhead and object allocation part of the hot path.

The JavaScript implementation is a clear behavioral oracle, but browser-based
execution is not a convenient training interface. Maintaining unrelated Python
and JavaScript rule engines would also increase conformance risk.

## Proposed direction

Evaluate Rust as the canonical simulator core.

The same Rust workspace should eventually expose:

- a native Rust API for the complete search and self-play loop;
- an out-of-process, versioned inference protocol implemented by Python;
- WebAssembly bindings for browser use;
- batched encoders that write directly into caller-owned or contiguous buffers.

Python bindings are not part of the initial architecture. Python training code
should not import a Rust extension or reproduce simulator behavior. Rust actors
send batched encoded states to a Python inference service and write
language-neutral replay shards that Python reads directly.

The first Rust implementation should port only authoritative behavior: position
validation, Voronoi geometry, legal-set geometry, global analysis, move
transactions, scoring, and serialization. Contour rendering remains reference
UI code.

## Why Rust is plausible

- Search can own compact positions and reuse allocations without a garbage
  collector in the simulation loop.
- Rasterization and feature encoding can use predictable contiguous layouts,
  threads, and later explicit SIMD.
- Python can train and serve models without owning per-point geometry loops or
  depending on the Rust build toolchain.
- WebAssembly provides a path toward one authoritative engine for both browser
  and training clients.

Rust is not assumed to win automatically. Binding copies, polygon allocation,
and exact-analysis complexity may dominate rasterization. GPU-side batched
encoding may also outperform CPU rasterization once training is established.

## Evaluation requirements

The decision is accepted only if a Rust prototype:

1. Passes shared conformance fixtures against the JavaScript reference.
2. Matches documented numerical tolerances and boundary behavior.
3. Beats or clearly scales better than JavaScript on representative analysis
   and move workloads.
4. Produces batched feature buffers with measured allocation and transfer cost.
5. Demonstrates bounded-overhead batched inference across the process boundary.

## Consequences

During evaluation, JavaScript remains the behavioral oracle. Rust benchmark
code must not simplify capture or legal-set analysis merely to produce a better
number. If Rust is selected, duplicated JavaScript rule logic should eventually
be replaced by the WASM build or retained only as an independent oracle.
