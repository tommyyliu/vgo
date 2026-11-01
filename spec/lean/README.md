# Lean proofs

A placeholder. There is no toolchain or project here yet, on purpose. The written
proofs in [`LEMMAS.md`](../LEMMAS.md) are the current standard.

Start with statements about real numbers, where Mathlib already has what is needed:

| target | why it is a good start |
|---|---|
| E1 tile candidates | triangle inequality over a finite set, and small |
| A2 cells are convex polygons | intersection of half-planes |
| A3 vertex witness | a linear function peaks at a vertex; capture depends on it |
| E6 clamping | projection onto a box does not increase distance |

Leave the floating-point lemmas (N1, N2) until these exist, since formalizing IEEE
rounding is far more work. E5 is a statement about a program, so it would need a model of
`compute_cell`.
