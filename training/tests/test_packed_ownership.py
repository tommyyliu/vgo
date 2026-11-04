from __future__ import annotations

import unittest

import torch

from vgo_training.dataset import PreparedRasterDataset
from vgo_training.learner import _drop_ownership, _pack_ownership
from vgo_training.packed_policy import is_ownership_packable, pack_ownership


def targets(samples: int = 9, cells: int = 16 * 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    signs = torch.randint(0, 2, (samples, cells), generator=generator)
    return (signs * 2 - 1).to(torch.int8)


def prepared(ownerships: torch.Tensor | None) -> PreparedRasterDataset:
    samples = 9 if ownerships is None else ownerships.shape[0]
    return PreparedRasterDataset(
        states=torch.zeros(samples, 1, 4, 4),
        policies=torch.zeros(samples, 257),
        policy_masks=torch.ones(samples, 257, dtype=torch.bool),
        values=torch.zeros(samples),
        height=4,
        width=4,
        sources=("fixture",),
        ownerships=ownerships,
    )


class PackedOwnershipTests(unittest.TestCase):
    def test_round_trip_is_exact_for_any_rows(self) -> None:
        # 250 cells, not a multiple of 8, so the last byte is partly padding.
        original = targets(cells=250)
        packed = pack_ownership(original)
        rows = torch.tensor([7, 0, 3, 3])
        out = torch.full((4, 250), 5.0)  # a recycled buffer holds junk
        packed.expand(rows, out)
        self.assertTrue(torch.equal(out, original[rows].to(torch.float32)))
        self.assertEqual(packed.nbytes, 9 * 32)

    def test_only_plus_minus_one_is_packed(self) -> None:
        with_zero = targets()
        with_zero[2, 5] = 0
        self.assertFalse(is_ownership_packable(with_zero))
        kept = _pack_ownership(prepared(with_zero))
        self.assertIsNone(kept.packed_ownership)
        self.assertIs(kept.ownerships, with_zero)

    def test_packed_shard_still_reads_as_having_targets(self) -> None:
        packed = _pack_ownership(prepared(targets()))
        self.assertIsNotNone(packed.packed_ownership)
        self.assertIsNotNone(packed.ownerships)
        self.assertEqual(packed.ownerships.shape[0], 0)
        self.assertIsNone(_pack_ownership(prepared(None)).packed_ownership)
        dropped = _drop_ownership(packed)
        self.assertIsNone(dropped.ownerships)
        self.assertIsNone(dropped.packed_ownership)


if __name__ == "__main__":
    unittest.main()
