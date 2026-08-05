import unittest

import torch

from vgo_training.short_term_value import (
    DEFAULT_HORIZONS,
    ShortTermValueHead,
    exponential_horizon_targets,
    short_term_value_loss,
)


class HorizonTargetTests(unittest.TestCase):
    def test_constant_values_average_to_themselves(self) -> None:
        values = torch.full((6,), 0.4)
        games = torch.zeros(6, dtype=torch.long)
        plies = torch.arange(6)
        targets = exponential_horizon_targets(values, games, plies)
        torch.testing.assert_close(targets, torch.full_like(targets, 0.4))

    def test_only_the_future_contributes(self) -> None:
        # Ply 0 sees the later values; the last ply has no future and falls
        # back to its own.
        values = torch.tensor([0.0, 1.0, 1.0, 1.0])
        games = torch.zeros(4, dtype=torch.long)
        plies = torch.arange(4)
        targets = exponential_horizon_targets(values, games, plies, horizons=(6,))
        self.assertGreater(targets[0, 0].item(), 0.9)
        self.assertAlmostEqual(targets[3, 0].item(), 1.0, places=5)

    def test_games_do_not_leak_into_each_other(self) -> None:
        values = torch.tensor([1.0, 1.0, -1.0, -1.0])
        games = torch.tensor([0, 0, 1, 1])
        plies = torch.tensor([0, 1, 0, 1])
        targets = exponential_horizon_targets(values, games, plies, horizons=(6,))
        self.assertGreater(targets[0, 0].item(), 0.0)
        self.assertLess(targets[2, 0].item(), 0.0)

    def test_a_short_horizon_is_more_local(self) -> None:
        # Value flips late. A short horizon should still see mostly the near
        # future; a long one picks up more of the flip.
        values = torch.cat([torch.zeros(20), torch.ones(20)])
        games = torch.zeros(40, dtype=torch.long)
        plies = torch.arange(40)
        targets = exponential_horizon_targets(values, games, plies, horizons=(2, 50))
        self.assertLess(targets[0, 0].item(), targets[0, 1].item())

    def test_one_column_per_horizon(self) -> None:
        values = torch.rand(12)
        games = torch.zeros(12, dtype=torch.long)
        plies = torch.arange(12)
        targets = exponential_horizon_targets(values, games, plies)
        self.assertEqual(targets.shape, (12, len(DEFAULT_HORIZONS)))

    def test_rejects_misaligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            exponential_horizon_targets(
                torch.rand(4), torch.zeros(3, dtype=torch.long), torch.arange(4)
            )


class HeadTests(unittest.TestCase):
    def test_output_is_bounded_and_shaped(self) -> None:
        head = ShortTermValueHead(channels=16)
        out = head(torch.randn(5, 16))
        self.assertEqual(out.shape, (5, len(DEFAULT_HORIZONS)))
        self.assertTrue(bool((out.abs() <= 1.0).all()))

    def test_loss_is_zero_on_a_perfect_prediction(self) -> None:
        targets = torch.rand(4, 3) * 2 - 1
        self.assertAlmostEqual(
            short_term_value_loss(targets.clone(), targets).item(), 0.0, places=6
        )

    def test_gradients_reach_the_head(self) -> None:
        head = ShortTermValueHead(channels=8)
        short_term_value_loss(head(torch.randn(3, 8)), torch.zeros(3, 3)).backward()
        self.assertIsNotNone(head.layers[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
