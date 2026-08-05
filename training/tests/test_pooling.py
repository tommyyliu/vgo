import unittest

import torch

from vgo_training.model import build_model
from vgo_training.pooling import GlobalPool, global_pool


class GlobalPoolTests(unittest.TestCase):
    def test_output_is_three_statistics_per_channel(self) -> None:
        pooled = global_pool(torch.randn(4, 8, 6, 6))
        self.assertEqual(pooled.shape, (4, 24))

    def test_mean_and_max_are_what_they_claim(self) -> None:
        features = torch.randn(3, 5, 4, 4)
        pooled = global_pool(features)
        torch.testing.assert_close(pooled[:, :5], features.mean(dim=(2, 3)))
        torch.testing.assert_close(pooled[:, 10:], features.amax(dim=(2, 3)))

    def test_max_distinguishes_boards_a_mean_cannot(self) -> None:
        # The reason to pool more than a mean: concentrated and spread features
        # have the same average and mean different things.
        spread = torch.full((1, 1, 4, 4), 1.0)
        concentrated = torch.zeros(1, 1, 4, 4)
        concentrated[0, 0, 0, 0] = 16.0
        a, b = global_pool(spread), global_pool(concentrated)
        torch.testing.assert_close(a[:, 0], b[:, 0])  # same mean
        self.assertNotAlmostEqual(a[0, 2].item(), b[0, 2].item())  # different max

    def test_masked_cells_are_excluded_from_the_mean(self) -> None:
        features = torch.ones(1, 1, 4, 4)
        features[0, 0, 2:] = 5.0
        mask = torch.ones(1, 1, 4, 4)
        mask[0, 0, 2:] = 0.0
        pooled = global_pool(features, mask)
        torch.testing.assert_close(pooled[0, 0], torch.tensor(1.0))

    def test_masked_cells_cannot_win_the_max(self) -> None:
        # Off-board cells hold post-ReLU values >= 0, so the mask shifts them to
        # -1 rather than letting garbage dominate.
        features = torch.zeros(1, 1, 4, 4)
        features[0, 0, 3, 3] = 9.0
        mask = torch.ones(1, 1, 4, 4)
        mask[0, 0, 3, 3] = 0.0
        self.assertLess(global_pool(features, mask)[0, 2].item(), 9.0)

    def test_module_reports_its_output_width(self) -> None:
        pool = GlobalPool(channels=12)
        self.assertEqual(pool.output_width, 36)
        self.assertEqual(pool(torch.randn(2, 12, 5, 5)).shape, (2, 36))

    def test_module_rejects_the_wrong_channel_count(self) -> None:
        with self.assertRaises(ValueError):
            GlobalPool(channels=4)(torch.randn(1, 7, 3, 3))


class PooledModelTests(unittest.TestCase):
    def test_default_is_unchanged(self) -> None:
        torch.manual_seed(0)
        plain = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8,
        )
        torch.manual_seed(0)
        flagged = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8, global_pooling=False,
        )
        self.assertEqual(set(plain.state_dict()), set(flagged.state_dict()))

    def test_pooled_heads_are_wider(self) -> None:
        pooled = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8, global_pooling=True,
        )
        plain = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8,
        )
        self.assertEqual(
            pooled.pass_head.in_features, 3 * plain.pass_head.in_features
        )

    def test_shapes_and_gradients_survive(self) -> None:
        model = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8, global_pooling=True,
        )
        model(torch.randn(2, 5, 64, 64))[0].sum().backward()
        self.assertIsNotNone(model.pass_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
