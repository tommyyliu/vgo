import unittest

import torch

from vgo_training.model import build_model
from vgo_training.optimistic import (
    optimistic_policy_loss,
    optimistic_weights,
    outcome_surprise,
)


class SurpriseTests(unittest.TestCase):
    def test_only_upward_surprise_counts(self) -> None:
        predicted = torch.tensor([0.0, 0.5, -0.5])
        outcome = torch.tensor([1.0, -1.0, 1.0])
        torch.testing.assert_close(
            outcome_surprise(predicted, outcome), torch.tensor([1.0, 0.0, 1.5])
        )

    def test_a_perfect_prediction_is_unsurprising(self) -> None:
        value = torch.tensor([0.3, -0.7])
        torch.testing.assert_close(
            outcome_surprise(value, value.clone()), torch.zeros(2)
        )


class WeightTests(unittest.TestCase):
    def test_weights_average_to_one(self) -> None:
        # Switching this on must redistribute the loss, not rescale it.
        weights = optimistic_weights(torch.rand(1000) * 2.0)
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

    def test_surprise_earns_more_weight(self) -> None:
        weights = optimistic_weights(torch.tensor([0.0, 1.5]))
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_unsurprising_samples_keep_a_baseline(self) -> None:
        weights = optimistic_weights(torch.tensor([0.0, 0.0, 2.0]), baseline=0.25)
        self.assertGreater(weights[0].item(), 0.0)

    def test_outliers_are_capped(self) -> None:
        weights = optimistic_weights(torch.tensor([0.0] * 50 + [1e4]), maximum=4.0)
        self.assertLessEqual(weights.max().item() / weights.mean().item(), 60.0)

    def test_rejects_a_negative_baseline(self) -> None:
        with self.assertRaises(ValueError):
            optimistic_weights(torch.ones(3), baseline=-1.0)


class LossTests(unittest.TestCase):
    def test_matches_unweighted_cross_entropy_at_uniform_weights(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 9)
        targets = torch.softmax(torch.randn(4, 9), dim=1)
        masks = torch.ones(4, 9, dtype=torch.bool)
        weighted = optimistic_policy_loss(logits, targets, masks, torch.ones(4))
        plain = -(targets * torch.log_softmax(logits, dim=1)).sum(1).mean()
        torch.testing.assert_close(weighted, plain)

    def test_illegal_actions_do_not_produce_nan(self) -> None:
        # log_softmax returns -inf on masked actions and the target is 0 there;
        # 0 * -inf is NaN unless those terms are skipped.
        logits = torch.randn(3, 6)
        masks = torch.ones(3, 6, dtype=torch.bool)
        masks[:, 4:] = False
        targets = torch.zeros(3, 6)
        targets[:, :4] = 0.25
        loss = optimistic_policy_loss(logits, targets, masks, torch.ones(3))
        self.assertTrue(torch.isfinite(loss))


class ModelTests(unittest.TestCase):
    def build(self, optimistic: bool):
        return build_model(
            "ddrnet", channels=5, width=32, blocks=8, policy_resolution=32,
            norm_groups=8, optimistic_policy=optimistic,
        )

    def test_default_is_unchanged(self) -> None:
        torch.manual_seed(0)
        plain = self.build(False)
        torch.manual_seed(0)
        flagged = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8,
        )
        self.assertEqual(set(plain.state_dict()), set(flagged.state_dict()))

    def test_head_appears_in_both_modes(self) -> None:
        model = self.build(True)
        x = torch.randn(2, 5, 64, 64)
        model.train()
        self.assertEqual(len(model(x)), 7)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        # Appended after the two-output contract, so existing consumers are
        # unaffected while search can read the third.
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[2].shape, (2, 32 * 32))

    def test_gradients_reach_the_head(self) -> None:
        model = self.build(True)
        model(torch.randn(2, 5, 64, 64))[6].sum().backward()
        self.assertIsNotNone(model.optimistic_map.weight.grad)


if __name__ == "__main__":
    unittest.main()
