"""Compact storage for policy targets and their legal masks.

The window's dominant cost, once states are packed and ownership released, is
the policy target: dense float32 over `policy_size` cells, of which about 54 are
nonzero. Measured on a real shard, `policies` was 65,540 bytes per sample and
`policy_masks` 16,385, together 80% and 20% of everything the window held --
for a tensor that is 99.67% zeros and a mask that is one byte per bit.

Both compress on their own terms. The target is sparse, so it keeps indices and
values and nothing else. The mask is genuinely dense -- it is the *full legal*
set derived from the board's clearance, not from the search -- but it is
boolean, so it packs eight cells to the byte.

The stager expands only the rows a batch needs, the same arrangement
`packed_states` uses. Expansion is a zero-fill of the destination plus a scatter
of the live cells, which is cheaper per batch than the state expansion already
running at 0.04% of a training step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedPolicy:
    """One shard's policy targets and legal masks, compactly.

    `indices` and `values` hold the sparse target padded to the widest row;
    `counts` says how much of each row is live. `mask_bits` holds the legal mask
    at one bit per cell, little-endian within each byte.
    """

    indices: torch.Tensor  # int32 [samples, width]
    values: torch.Tensor  # float32 [samples, width]
    counts: torch.Tensor  # int32 [samples]
    mask_bits: torch.Tensor  # uint8 [samples, ceil(policy_size / 8)]
    policy_size: int

    @property
    def samples(self) -> int:
        return int(self.counts.numel())

    @property
    def nbytes(self) -> int:
        return (
            self.indices.nbytes
            + self.values.nbytes
            + self.counts.nbytes
            + self.mask_bits.nbytes
        )

    def expand_policies(self, rows: torch.Tensor, out: torch.Tensor) -> None:
        """Write the dense targets for `rows` into `out`.

        `out` is zeroed first: a sparse row says nothing about the cells it
        omits, and a destination buffer is recycled across batches, so anything
        left there would read as target mass on cells this search never touched.
        """
        width = self.indices.shape[1]
        if width == 0:
            out.zero_()
            return
        index = self.indices.index_select(0, rows).long()
        value = self.values.index_select(0, rows).to(out.dtype)
        live = (
            torch.arange(width, device=index.device)[None, :]
            < self.counts.index_select(0, rows).long()[:, None]
        )
        # Padding slots carry index 0, which is a real cell, so they are sent to
        # one extra scratch column instead. Scattering into a padded buffer and
        # slicing it off is what keeps a padded slot from writing a zero over
        # live mass that another slot in the same row already placed at cell 0.
        index = torch.where(live, index, torch.full_like(index, self.policy_size))
        padded = torch.zeros(
            (out.shape[0], self.policy_size + 1), dtype=out.dtype, device=out.device
        )
        padded.scatter_(1, index, value)
        out.copy_(padded[:, : self.policy_size])

    def expand_masks(self, rows: torch.Tensor, out: torch.Tensor) -> None:
        """Write the dense boolean legal masks for `rows` into `out`."""
        bits = self.mask_bits.index_select(0, rows)
        unpacked = torch.zeros(
            (bits.shape[0], bits.shape[1] * 8), dtype=torch.uint8, device=bits.device
        )
        for bit in range(8):
            unpacked[:, bit::8] = (bits >> bit) & 1
        out.copy_(unpacked[:, : self.policy_size].to(out.dtype))


def is_packable(policies: torch.Tensor, masks: torch.Tensor) -> bool:
    """Whether these targets have the shape the packer assumes."""
    return (
        policies.ndim == 2
        and masks.ndim == 2
        and policies.shape == masks.shape
        and policies.shape[0] > 0
    )


def pack(policies: torch.Tensor, masks: torch.Tensor) -> PackedPolicy:
    """Pack dense targets and masks.

    The padded width is the widest live row in the shard rather than a constant:
    a shard whose search stayed narrow should not carry the padding of one whose
    search went wide.
    """
    samples, policy_size = policies.shape
    live = policies != 0
    counts = live.sum(dim=1)
    width = int(counts.max().item()) if samples else 0

    indices = torch.zeros((samples, width), dtype=torch.int32)
    values = torch.zeros((samples, width), dtype=torch.float32)
    for row in range(samples):
        columns = live[row].nonzero(as_tuple=True)[0]
        n = int(columns.numel())
        if n:
            indices[row, :n] = columns.to(torch.int32)
            values[row, :n] = policies[row, columns].to(torch.float32)

    packed_bits = torch.zeros((samples, (policy_size + 7) // 8), dtype=torch.uint8)
    flat = masks.to(torch.uint8)
    for bit in range(8):
        columns = flat[:, bit::8]
        packed_bits[:, : columns.shape[1]] |= columns << bit

    return PackedPolicy(
        indices=indices,
        values=values,
        counts=counts.to(torch.int32),
        mask_bits=packed_bits,
        policy_size=policy_size,
    )
