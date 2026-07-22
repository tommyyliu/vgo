# 001: Robust Capture Comparison

- Status: Resolved
- Priority: High
- Owner: Developer

## Problem

The exact escape condition is `freeDist(v) < rho(v)`, but
`VGO.analysis.analyze()` currently requires
`freeDist(v) < rho(v) - captureMargin`. The centralized fixed deadband can
remove a group with a very narrow valid escape.

`rho` is a geometric threshold. The tolerance is an implementation detail and
must not become part of the rules.

## Goal

Make near-boundary capture decisions deterministic and as accurate as practical
without slowing ordinary positions unnecessarily.

## Acceptance criteria

- The fast path computes a signed margin and an explicit numerical error bound.
- Clearly separated margins are decided with normal floating-point arithmetic.
- Indeterminate margins use a higher-precision or otherwise certified fallback.
- Exact ties remain settled because they transfer no positive area.
- Tests cover a definite escape, a definite capture, an exact tie, and cases on
  both sides of the previous `1e-7` deadband.
- [`reference/AXIOMS.md`](../reference/AXIOMS.md) describes the implemented
  numerical policy accurately.

## Resolution

Both the Rust engine and JavaScript reference use outward-rounded binary64
intervals as the fast path. Only comparisons whose interval contains zero fall
back to an exact dyadic squared-distance comparison backed by arbitrary-size
integers. The capture predicate is strict, so exact ties remain settled. Unit
and browser regression tests cover ordinary cases, exact equality, and both
sides of the removed `1e-7` deadband.
