import unittest

import numpy as np
import torch

from vgo_training.packed_states import (
    _LAYOUTS,
    BINARY_CHANNELS,
    COMPACT_CHANNEL_COUNT,
    KOMI,
    PackingUnsupported,
    VORONOI_RIDGE,
    conformance_error,
    is_packable,
    pack,
)


def sample_states(samples: int = 7, height: int = 8, width: int = 8) -> torch.Tensor:
    """Compact planes with the structure real shards have.

    Binary stone and settled masks, a continuous ridge field, and one komi
    value broadcast across its plane.
    """
    generator = torch.Generator().manual_seed(11)
    states = torch.zeros(
        (samples, COMPACT_CHANNEL_COUNT, height, width), dtype=torch.float16
    )
    for channel in BINARY_CHANNELS:
        states[:, channel] = (
            torch.rand((samples, height, width), generator=generator) > 0.5
        ).to(torch.float16)
    states[:, VORONOI_RIDGE] = torch.rand(
        (samples, height, width), generator=generator
    ).to(torch.float16)
    komi = torch.linspace(-0.234, 0.234, samples).to(torch.float16)
    states[:, KOMI] = komi.view(samples, 1, 1).expand(samples, height, width)
    return states


class PackedStateTests(unittest.TestCase):
    def test_round_trip_is_exact(self) -> None:
        # Every channel must return bit-exact: the stager feeds each gradient
        # from the expanded form, so drift here is silent training corruption.
        states = sample_states()
        for channel, error in conformance_error(states).items():
            self.assertEqual(error, 0.0, channel)

    def test_expand_selects_rows(self) -> None:
        states = sample_states(9)
        packed = pack(states)
        rows = torch.tensor([7, 0, 3, 3])
        self.assertTrue(
            torch.equal(packed.expand(rows), states.index_select(0, rows))
        )

    def test_packing_shrinks_the_planes(self) -> None:
        states = sample_states(16, 32, 32)
        dense = states.numel() * states.element_size()
        self.assertLess(pack(states).nbytes() * 3, dense)

    def test_expand_accepts_a_destination(self) -> None:
        states = sample_states(5)
        packed = pack(states)
        destination = torch.empty(packed.shape, dtype=torch.float16)
        returned = packed.expand(out=destination)
        self.assertIs(returned, destination)
        self.assertTrue(torch.equal(destination, states))

    def test_rejects_a_non_binary_channel(self) -> None:
        # A layout change that makes a "binary" plane continuous has to fall
        # back to dense storage rather than silently truncating to one bit.
        states = sample_states()
        states[0, BINARY_CHANNELS[0], 0, 0] = 0.5
        self.assertFalse(is_packable(states))
        with self.assertRaises(PackingUnsupported):
            pack(states)

    def test_rejects_a_varying_scalar_channel(self) -> None:
        states = sample_states()
        states[0, KOMI, 0, 0] = 0.9
        self.assertFalse(is_packable(states))
        with self.assertRaises(PackingUnsupported):
            pack(states)

    def test_rejects_a_foreign_channel_count(self) -> None:
        # A width with no layout. Deliberately not COMPACT_CHANNEL_COUNT + 1:
        # six planes is `compact-pass` / `compact-dead-zone` and is supported,
        # so using it here would assert the opposite of what this test is for.
        unknown = max(_LAYOUTS) + 1
        states = torch.zeros((2, unknown, 4, 4), dtype=torch.float16)
        self.assertFalse(is_packable(states))

    def test_packs_the_six_plane_layouts(self) -> None:
        """Both six-plane rasters pack, and `previous_pass` costs a scalar.

        The two share a layout because they differ only in *which* capture
        predicate sits in slot 3, and both are binary -- so the storage classes
        are identical and only the meaning changes.
        """
        samples, height, width = 6, 8, 8
        states = torch.zeros((samples, 6, height, width), dtype=torch.float16)
        generator = torch.Generator().manual_seed(7)
        for channel in (0, 1, 3):  # stones, stones, capture predicate
            states[:, channel] = torch.randint(
                0, 2, (samples, height, width), generator=generator
            ).to(torch.float16)
        states[:, 2] = torch.rand(
            (samples, height, width), generator=generator
        ).to(torch.float16)
        for sample in range(samples):
            states[sample, 4] = 0.104 * sample          # komi
            states[sample, 5] = float(sample % 2)       # previous_pass

        self.assertTrue(is_packable(states))
        packed = pack(states)
        self.assertEqual(packed.channels, 6)
        self.assertEqual(packed.shape, (samples, 6, height, width))
        self.assertTrue(torch.equal(packed.expand(), states))

        # A scalar, not a bit plane: one value per sample for each of komi and
        # previous_pass, and bits for the three genuinely spatial binaries.
        self.assertEqual(packed.scalars.shape, (samples, 2))
        self.assertEqual(packed.bits.shape[1], 3)

    def test_a_non_constant_pass_plane_falls_back(self) -> None:
        """`previous_pass` is a scalar only because it is constant.

        If a layout change ever made it vary across the board, packing must
        decline rather than keep the first pixel and discard the rest.
        """
        states = torch.zeros((2, 6, 4, 4), dtype=torch.float16)
        states[0, 5, 1, 1] = 1.0
        self.assertFalse(is_packable(states))

    def test_accepts_numpy_input(self) -> None:
        states = sample_states(4)
        packed = pack(states.numpy().astype(np.float32))
        self.assertTrue(torch.equal(packed.expand(), states))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_round_trip_is_exact_on_device(self) -> None:
        states = sample_states(12, 16, 16)
        packed = pack(states).to("cuda")
        rows = torch.tensor([11, 2, 5], device="cuda")
        expected = states.to("cuda").index_select(0, rows)
        self.assertTrue(torch.equal(packed.expand(rows), expected))


if __name__ == "__main__":
    unittest.main()
