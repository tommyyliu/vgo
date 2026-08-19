"""Compact storage for the resident state planes.

Three of the five compact channels are strictly binary and `komi` is constant
across its plane -- and the six-plane layouts add `previous_pass`, which is
both, so it costs one fp16 per sample rather than a bit plane -- so a dense fp16 tensor spends 160 KB per sample on data that
needs 38 KB. This packs the binary planes to bits and komi to one scalar, and
expands them again per batch.

The window is what this saves: a shard's rasters are rendered at load and held
until the shard leaves the replay window, so anything resident is multiplied by
the window size. Unpacking costs 0.159 ms per batch of 256 on the GPU -- 0.04%
of a training step -- and the smaller host-to-device copy more than pays it
back (40.0 MB at 0.813 ms dense against 9.5 MB at 0.312 ms packed).

The packed form is a runtime representation only. Shards store stone positions,
not rasters (see docs/POSITION_SHARDS.md), so nothing here touches the on-disk
format and no migration is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

@dataclass(frozen=True)
class Layout:
    """How one raster layout's planes divide into storage classes.

    `binary` planes hold only 0.0 or 1.0 and survive a bit round trip.
    `scalar` planes are constant across the plane, so one value per sample
    reconstructs them. `continuous` keeps full precision.
    """

    channels: int
    binary: tuple[int, ...]
    scalar: tuple[int, ...]
    continuous: tuple[int, ...]

    def __post_init__(self) -> None:
        covered = sorted((*self.binary, *self.scalar, *self.continuous))
        if covered != list(range(self.channels)):
            raise ValueError(f"layout does not classify every plane exactly once: {covered}")


# Keyed by channel count, from the `COMPACT*_CHANNELS` lists in
# crates/vgo-raster/src/lib.rs.
#
# The six-plane layouts -- `compact-pass` and `compact-dead-zone` -- share an
# entry, and that is sound rather than a shortcut: they differ only in *which*
# capture predicate sits in slot 3, and both are binary, so the storage classes
# are identical. `_validate` checks the structure it assumes on the data itself,
# so a layout that ever stopped fitting falls back to dense rather than being
# silently corrupted.
#
# `previous_pass` is a scalar rather than a bit plane. It is binary, but it is
# also constant across the plane, and one fp16 beats 2 KB of bits.
_LAYOUTS: dict[int, Layout] = {
    # current_stones, opponent_stones, voronoi_ridge, settled, komi
    5: Layout(channels=5, binary=(0, 1, 3), scalar=(4,), continuous=(2,)),
    # ..., komi, previous_pass
    6: Layout(channels=6, binary=(0, 1, 3), scalar=(4, 5), continuous=(2,)),
    # current_stones, opponent_stones, voronoi_ridge, settled, dead_zone,
    # connections, komi, previous_pass.
    #
    # `connections` is signed rather than binary -- +1 for the mover's, -1 for
    # the opponent's -- so it keeps full precision. Everything else splits as
    # before.
    # ..., settled, dead_zone, current_connections, opponent_connections, komi,
    # previous_pass. Six binary planes: splitting connections by side rather than
    # signing them is what keeps them packable, 2 bits against fp16's 16.
    9: Layout(channels=9, binary=(0, 1, 3, 4, 5, 6), scalar=(7, 8), continuous=(2,)),
}

# The compact planes by name, for readers and for tests. The six-plane layouts
# keep these positions and add `previous_pass` at 5.
CURRENT_STONES = 0
OPPONENT_STONES = 1
VORONOI_RIDGE = 2
SETTLED = 3
KOMI = 4
PREVIOUS_PASS = 5

COMPACT_CHANNEL_COUNT = 5
COMPACT_LAYOUT = _LAYOUTS[COMPACT_CHANNEL_COUNT]
BINARY_CHANNELS = COMPACT_LAYOUT.binary
SCALAR_CHANNELS = COMPACT_LAYOUT.scalar
CONTINUOUS_CHANNELS = COMPACT_LAYOUT.continuous


class PackingUnsupported(ValueError):
    """Raised when a layout does not have the structure packing assumes."""


@dataclass(frozen=True)
class PackedStates:
    """Compact planes for one shard.

    `bits` holds `BINARY_CHANNELS` as one bit per pixel, `continuous` holds
    `CONTINUOUS_CHANNELS` at fp16, and `scalars` holds one value per
    `SCALAR_CHANNELS` entry per sample. `expand` reverses this exactly.
    """

    bits: torch.Tensor  # (samples, len(BINARY_CHANNELS), ceil(pixels/8)) uint8
    continuous: torch.Tensor  # (samples, len(CONTINUOUS_CHANNELS), H, W) fp16
    scalars: torch.Tensor  # (samples, len(layout.scalar)) fp16
    height: int
    width: int
    layout: Layout = COMPACT_LAYOUT

    @property
    def samples(self) -> int:
        return self.bits.shape[0]

    @property
    def channels(self) -> int:
        return self.layout.channels

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """The shape `expand` produces, so callers can size buffers."""
        return (self.samples, self.layout.channels, self.height, self.width)

    @property
    def dtype(self) -> torch.dtype:
        return self.continuous.dtype

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.bits, self.continuous, self.scalars)
        )

    def expand(
        self, rows: torch.Tensor | None = None, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reconstruct dense `[rows, channels, H, W]` fp16 planes.

        `rows` selects a subset, which is how the batch stager uses this: the
        bit expansion then runs over one batch rather than the whole window.
        """
        bits = self.bits if rows is None else self.bits.index_select(0, rows)
        continuous = (
            self.continuous if rows is None else self.continuous.index_select(0, rows)
        )
        scalars = self.scalars if rows is None else self.scalars.index_select(0, rows)

        count = bits.shape[0]
        pixels = self.height * self.width
        device = bits.device
        if out is None:
            out = torch.empty(
                (count, self.layout.channels, self.height, self.width),
                dtype=self.continuous.dtype,
                device=device,
            )

        # One bit per pixel, least-significant bit first, matching `pack`.
        offsets = torch.arange(8, device=device, dtype=torch.uint8)
        unpacked = torch.bitwise_and(
            bits.unsqueeze(-1) >> offsets, 1
        ).view(count, len(self.layout.binary), -1)[:, :, :pixels]
        for slot, channel in enumerate(self.layout.binary):
            out[:, channel] = unpacked[:, slot].view(
                count, self.height, self.width
            ).to(out.dtype)
        for slot, channel in enumerate(self.layout.continuous):
            out[:, channel] = continuous[:, slot]
        for slot, channel in enumerate(self.layout.scalar):
            out[:, channel] = scalars[:, slot].view(count, 1, 1).expand(
                count, self.height, self.width
            )
        return out

    def to(self, device: torch.device | str) -> "PackedStates":
        return PackedStates(
            bits=self.bits.to(device),
            continuous=self.continuous.to(device),
            scalars=self.scalars.to(device),
            height=self.height,
            width=self.width,
            layout=self.layout,
        )


def is_packable(states: np.ndarray | torch.Tensor) -> bool:
    """Whether `states` has the structure `pack` assumes.

    Checked rather than assumed: a raster layout that grows a channel, or a
    komi plane that stops being constant, must fall back to dense storage
    instead of being silently corrupted.
    """
    try:
        _validate(states)
    except PackingUnsupported:
        return False
    return True


def _validate(states: np.ndarray | torch.Tensor) -> Layout:
    if states.ndim != 4:
        raise PackingUnsupported("states must be [samples, channels, height, width]")
    layout = _LAYOUTS.get(states.shape[1])
    if layout is None:
        raise PackingUnsupported(
            f"no packing layout for {states.shape[1]} channels; known: "
            f"{sorted(_LAYOUTS)}"
        )
    tensor = _as_tensor(states)
    for channel in layout.binary:
        plane = tensor[:, channel]
        if not bool(((plane == 0) | (plane == 1)).all()):
            raise PackingUnsupported(
                f"channel {channel} is not binary and cannot be packed to bits"
            )
    for channel in layout.scalar:
        plane = tensor[:, channel].reshape(tensor.shape[0], -1)
        if not bool((plane.amin(dim=1) == plane.amax(dim=1)).all()):
            raise PackingUnsupported(
                f"channel {channel} is not constant across its plane"
            )
    return layout


def _as_tensor(states: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        return states
    return torch.from_numpy(np.ascontiguousarray(states))


def pack(states: np.ndarray | torch.Tensor) -> PackedStates:
    """Pack dense planes. Raises if the layout does not fit."""
    layout = _validate(states)
    tensor = _as_tensor(states)
    samples, _, height, width = tensor.shape
    pixels = height * width

    binary = tensor[:, list(layout.binary)].reshape(
        samples, len(layout.binary), pixels
    )
    # numpy packs bits big-endian by default; expand reads little-endian, so
    # ask for the matching order rather than reversing it on the hot path.
    bits = torch.from_numpy(
        np.packbits(
            binary.to(torch.uint8).cpu().numpy(), axis=-1, bitorder="little"
        )
    )
    continuous = tensor[:, list(layout.continuous)].to(torch.float16).clone()
    scalars = (
        tensor[:, list(layout.scalar)]
        .reshape(samples, len(layout.scalar), pixels)[:, :, 0]
        .to(torch.float16)
        .clone()
    )
    return PackedStates(
        bits=bits,
        continuous=continuous,
        scalars=scalars,
        height=height,
        width=width,
        layout=layout,
    )


def conformance_error(
    states: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Round-trip `states` and report the per-channel error.

    Binary and scalar channels must come back exactly; the continuous channel
    is only fp16-exact, which is what it was already stored as. The stager
    feeds every gradient from the expanded form, so a mismatch here corrupts
    training silently rather than failing loudly.
    """
    reference = _as_tensor(states).to(torch.float16)
    restored = pack(states).expand()
    report: dict[str, float] = {}
    for channel in range(_as_tensor(states).shape[1]):
        difference = (restored[:, channel].float() - reference[:, channel].float())
        report[f"channel_{channel}"] = float(difference.abs().max().item())
    return report
