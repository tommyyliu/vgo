import unittest

import torch

from vgo_training.recency import effective_window, row_weights, shard_weights


class ShardWeightTests(unittest.TestCase):
    def test_uniform_decay_is_uniform(self) -> None:
        weights = shard_weights(torch.arange(6.0), decay=1.0)
        torch.testing.assert_close(weights, torch.ones(6))

    def test_weights_average_one(self) -> None:
        # Reweighting must redistribute the sampling budget, not resize it --
        # otherwise this is a learning-rate change in disguise.
        for decay in (0.99, 0.9, 0.7):
            weights = shard_weights(torch.arange(20.0), decay=decay)
            self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

    def test_newer_shards_outweigh_older(self) -> None:
        weights = shard_weights(torch.arange(5.0), decay=0.8)
        self.assertTrue(bool((weights[:-1] > weights[1:]).all()))

    def test_floor_keeps_the_tail_alive(self) -> None:
        # A long window with aggressive decay must not silently become a short
        # one; the point is to keep diversity, not discard it more cheaply.
        weights = shard_weights(torch.arange(40.0), decay=0.5, floor=0.05)
        self.assertGreater(weights.min().item(), 0.0)

    def test_rejects_an_invalid_decay(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                shard_weights(torch.arange(3.0), decay=bad)


class RowWeightTests(unittest.TestCase):
    def test_uniform_decay_short_circuits(self) -> None:
        weights = row_weights([10, 10, 10], decay=1.0)
        torch.testing.assert_close(weights, torch.ones(30))

    def test_rows_inherit_their_shard_weight(self) -> None:
        weights = row_weights([2, 2], decay=0.5)
        self.assertAlmostEqual(weights[0].item(), weights[1].item(), places=6)
        self.assertGreater(weights[0].item(), weights[2].item())

    def test_row_weights_average_one(self) -> None:
        weights = row_weights([100] * 12, decay=0.85)
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_unequal_shard_sizes_still_average_one(self) -> None:
        # A large old shard must not outweigh a small new one just by size.
        weights = row_weights([500, 100, 100], decay=0.8)
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_newest_outweighs_oldest(self) -> None:
        weights = row_weights([50] * 10, decay=0.9)
        self.assertGreater(weights[0].item(), weights[-1].item())


class EffectiveWindowTests(unittest.TestCase):
    def test_uniform_uses_the_whole_window(self) -> None:
        self.assertAlmostEqual(effective_window([100] * 8, decay=1.0), 8.0, places=3)

    def test_decay_shrinks_the_effective_window(self) -> None:
        sizes = [100] * 20
        self.assertLess(
            effective_window(sizes, 0.8), effective_window(sizes, 0.95)
        )

    def test_reports_what_a_decay_costs(self) -> None:
        # The number a config should be chosen against: 25 shards at 0.9 behave
        # like roughly 16, which is the diversity actually being trained on.
        self.assertAlmostEqual(effective_window([4000] * 25, 0.9), 16.5, delta=1.0)


if __name__ == "__main__":
    unittest.main()
