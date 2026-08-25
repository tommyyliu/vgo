"""The compressed inference input must rebuild the dense raster exactly.

Not approximately. The compression throws nothing away: the binary planes
really are binary, the constant planes really are constant, and the states are
fp16 on disk already. If any of those stops holding, the rebuilt raster
diverges from what the model was trained to read and nothing downstream would
notice -- the network would just be slightly wrong about positions, in a way
indistinguishable from ordinary training noise.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from vgo_training.packed_input import PackedInputModel, bit_expansion_table, compress
from vgo_training.packed_states import _LAYOUTS

LAYOUT = _LAYOUTS[7]


class Echo(torch.nn.Module):
    """Returns its input, so a comparison isolates the assembly.

    Carries a float32 parameter because the wrapper widens to the weights'
    dtype, and a module with no parameters would not exercise that.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return states, states.mean(dim=(1, 2, 3))


def wrapper(height: int = 16, width: int = 16) -> PackedInputModel:
    return PackedInputModel(
        Echo(),
        binary=LAYOUT.binary,
        continuous=LAYOUT.continuous,
        scalar=LAYOUT.scalar,
        height=height,
        width=width,
    )


def sample_states(samples: int = 5, height: int = 16, width: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    states = torch.zeros(samples, 7, height, width, dtype=torch.float16)
    for channel in LAYOUT.binary:
        states[:, channel] = (
            torch.rand(samples, height, width, generator=generator) > 0.5
        ).half()
    for channel in LAYOUT.continuous:
        states[:, channel] = torch.rand(
            samples, height, width, generator=generator
        ).half()
    for offset, channel in enumerate(LAYOUT.scalar):
        for sample in range(samples):
            states[sample, channel] = float(sample + offset) * 0.0625
    return states


def compressed(states: torch.Tensor):
    return compress(
        states,
        binary=LAYOUT.binary,
        continuous=LAYOUT.continuous,
        scalar=LAYOUT.scalar,
    )


class PackedInputTests(unittest.TestCase):
    def test_round_trip_is_exact(self) -> None:
        states = sample_states()
        rebuilt, _ = wrapper()(*compressed(states))
        self.assertTrue(torch.equal(rebuilt, states.float()))

    def test_bit_order_matches_numpy(self) -> None:
        """`np.packbits(bitorder="little")` is the order `vgo-raster` writes.

        The two have to agree and neither is more natural than the other, so
        both ends spell it out rather than inferring it.
        """
        table = bit_expansion_table(torch.float32)
        generator = np.random.default_rng(3)
        plane = (generator.random(64) > 0.5).astype(np.uint8)
        packed = np.packbits(plane, bitorder="little")
        expanded = table[torch.from_numpy(packed).long()].reshape(-1)[:64]
        self.assertTrue(np.array_equal(expanded.numpy().astype(np.uint8), plane))

    def test_shrinks_the_staged_bytes(self) -> None:
        states = sample_states(4, 256, 256)
        dense_bytes = states[0].numel() * 4
        packed_bytes = sum(t.numel() * t.element_size() for t in compressed(states)) // 4
        self.assertLess(packed_bytes * 10, dense_bytes)

    def test_rejects_a_varying_scalar_plane(self) -> None:
        """Averaging it away silently would be a quiet train/serve skew."""
        states = sample_states()
        states[0, LAYOUT.scalar[0], 0, 0] += 1.0
        with self.assertRaises(ValueError):
            compressed(states)

    def test_rejects_a_non_binary_plane(self) -> None:
        states = sample_states()
        states[0, LAYOUT.binary[0], 0, 0] = 0.5
        with self.assertRaises(ValueError):
            compressed(states)

    def test_ignores_the_padding_bits(self) -> None:
        """A pixel count that is not a multiple of eight leaves spare bits."""
        height, width = 5, 5           # 25 pixels -> 4 bytes, 7 bits of padding
        states = sample_states(3, height, width)
        bits, dense, scalars = compressed(states)
        self.assertEqual(bits.shape[-1], 4)
        rebuilt, _ = wrapper(height, width)(bits, dense, scalars)
        self.assertTrue(torch.equal(rebuilt, states.float()))

    def test_rejects_a_layout_with_a_gap(self) -> None:
        with self.assertRaises(ValueError):
            PackedInputModel(
                Echo(), binary=(0,), continuous=(2,), scalar=(), height=4, width=4
            )


if __name__ == "__main__":
    unittest.main()
