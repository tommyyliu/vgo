"""Expanding a compressed inference input into the raster the model reads.

Staging is what limits inference, not arithmetic. At 256x256 a seven-plane
float32 raster is 1792 KB per position, and the two inference lanes measured
87% CPU each while thirty-four actor threads sat at 39% -- the host was busy
copying, and the GPU was waiting for it.

Most of those bytes carry nothing. Three of the seven planes hold a single
value repeated 65,536 times (komi, previous_pass, radius), three more are
strictly binary, and the data is already fp16 on disk: it is widened to
float32 only to be handed across, and the model casts it straight back.

    seven planes float32                    1792 KB
    drop the three constant planes          1024 KB
    ... and fp16                             512 KB
    ... and the binary planes as bits        152 KB

This module is the other half of that: the compressed triple arrives, and the
graph expands it on the device. The arithmetic below the wrapper is unchanged,
so this is not an approximation -- the assembled tensor is bit-identical to
what the dense path produced, which `tests/test_packed_input.py` asserts
against the real rasterizer output.

Bits are expanded by `Gather` against a 256x8 table rather than by `BitShift`
and `BitwiseAnd`. The bitwise operators are the obvious spelling, but their
TensorRT coverage is uneven and a single unsupported node forces a subgraph
back to another provider, which would cost more than the copy this saves.
Gather is supported everywhere and the table is 4 KB.
"""

from __future__ import annotations

import torch
from torch import nn


def bit_expansion_table(dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """A 256x8 table mapping a byte to its eight bits, lowest bit first.

    Lowest-bit-first matches the packing order in `vgo-raster`, which fills a
    byte from bit 0 as it walks pixels in row-major order. The two orders have
    to agree and neither is more natural than the other, so the constraint is
    recorded in both places rather than inferred.
    """
    values = torch.arange(256, dtype=torch.int64).unsqueeze(1)
    shifts = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    return ((values >> shifts) & 1).to(dtype)


class PackedInputModel(nn.Module):
    """Wraps a policy/value net so it accepts the compressed input triple.

    `bits` holds the binary planes packed eight pixels to a byte, `dense` the
    planes that are neither binary nor constant, and `scalars` one value per
    constant plane. Channel positions come from the layout, so a raster that
    grows a plane changes one table rather than this code.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        binary: tuple[int, ...],
        continuous: tuple[int, ...],
        scalar: tuple[int, ...],
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        total = len(binary) + len(continuous) + len(scalar)
        covered = sorted((*binary, *continuous, *scalar))
        if covered != list(range(total)):
            raise ValueError(
                f"layout must cover channels 0..{total - 1} exactly once, got {covered}"
            )
        self.model = model
        self.binary = tuple(binary)
        self.continuous = tuple(continuous)
        self.scalar = tuple(scalar)
        self.height = int(height)
        self.width = int(width)
        self.channels = total
        # The saving is in what crosses PCIe, not in what the model computes:
        # inputs arrive as fp16 and are widened here, on the device, to whatever
        # the weights are. Assembling in fp16 and feeding that straight in fails
        # -- a half input against float32 bias is a type error, and forcing the
        # weights to half would change the arithmetic this is supposed to leave
        # alone.
        weights = next(model.parameters(), None)
        self.compute_dtype = torch.float32 if weights is None else weights.dtype
        self.register_buffer(
            "table", bit_expansion_table(self.compute_dtype), persistent=False
        )

    def forward(
        self, bits: torch.Tensor, dense: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = dense.shape[0]
        pixels = self.height * self.width

        # (B, nbinary, bytes) -> (B, nbinary, bytes, 8) -> (B, nbinary, pixels).
        # The trailing bits of the last byte are padding and are sliced off.
        expanded = self.table[bits.to(torch.int64)]
        expanded = expanded.reshape(batch, len(self.binary), -1)[:, :, :pixels]
        expanded = expanded.reshape(batch, len(self.binary), self.height, self.width)

        planes: list[torch.Tensor | None] = [None] * self.channels
        for slot, channel in enumerate(self.binary):
            planes[channel] = expanded[:, slot : slot + 1]
        for slot, channel in enumerate(self.continuous):
            planes[channel] = dense[:, slot : slot + 1]
        for slot, channel in enumerate(self.scalar):
            planes[channel] = scalars[:, slot].reshape(batch, 1, 1, 1).expand(
                batch, 1, self.height, self.width
            )

        states = torch.cat(
            [p.to(self.compute_dtype) for p in planes if p is not None], dim=1
        )
        return self.model(states)


def compress(
    states: torch.Tensor,
    *,
    binary: tuple[int, ...],
    continuous: tuple[int, ...],
    scalar: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The inverse of `PackedInputModel.forward`, for tests and for tooling.

    This is the reference the Rust rasterizer has to agree with, so it is
    written the way the Rust is: bits filled lowest-first in row-major pixel
    order, one byte per eight pixels, the final byte zero-padded. Production
    never calls this -- the generator packs as it rasterizes rather than
    building a dense raster and compressing it, which would pay the cost this
    exists to avoid.
    """
    batch, _, height, width = states.shape
    pixels = height * width

    flat = states[:, list(binary)].reshape(batch, len(binary), pixels)
    if not torch.isin(flat, torch.tensor([0.0, 1.0], dtype=flat.dtype)).all():
        raise ValueError("binary planes must hold only 0.0 or 1.0")
    padded = -(-pixels // 8) * 8
    if padded != pixels:
        pad = torch.zeros(batch, len(binary), padded - pixels, dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=2)
    weights = (1 << torch.arange(8, dtype=torch.int64)).to(flat.dtype)
    bits = (flat.reshape(batch, len(binary), -1, 8) * weights).sum(-1).to(torch.uint8)

    dense = states[:, list(continuous)].contiguous()

    columns = []
    for channel in scalar:
        plane = states[:, channel].reshape(batch, -1)
        first = plane[:, :1]
        if not torch.equal(plane, first.expand_as(plane)):
            raise ValueError(f"channel {channel} is not constant across the plane")
        columns.append(first)
    scalars = torch.cat(columns, dim=1)
    return bits, dense, scalars
