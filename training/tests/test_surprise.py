import unittest

import torch

from vgo_training.surprise import (
    policy_kl_from_prior,
    sample_counts,
    surprise_weights,
    weighted_row_index,
)


class PolicyKLTests(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self) -> None:
        target = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
        masks = torch.ones_like(target, dtype=torch.bool)
        divergence = policy_kl_from_prior(target, target.clone(), masks)
        torch.testing.assert_close(divergence, torch.zeros(2), atol=1e-6, rtol=0)

    def test_divergence_grows_with_disagreement(self) -> None:
        masks = torch.ones(1, 3, dtype=torch.bool)
        target = torch.tensor([[1.0, 0.0, 0.0]])
        near = policy_kl_from_prior(target, torch.tensor([[0.8, 0.1, 0.1]]), masks)
        far = policy_kl_from_prior(target, torch.tensor([[0.1, 0.8, 0.1]]), masks)
        self.assertGreater(far.item(), near.item())

    def test_illegal_actions_do_not_contribute(self) -> None:
        # A prior that wasted mass on an illegal action should not read as
        # disagreement -- it is renormalized away first.
        masks = torch.tensor([[True, True, False]])
        target = torch.tensor([[0.5, 0.5, 0.0]])
        prior = torch.tensor([[0.25, 0.25, 0.5]])
        divergence = policy_kl_from_prior(target, prior, masks)
        torch.testing.assert_close(divergence, torch.zeros(1), atol=1e-6, rtol=0)

    def test_is_finite_when_the_prior_has_a_hole(self) -> None:
        # The prior assigns nothing where the target has mass. True KL is
        # infinite; the clamp has to keep it large but usable.
        masks = torch.ones(1, 2, dtype=torch.bool)
        divergence = policy_kl_from_prior(
            torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]), masks
        )
        self.assertTrue(torch.isfinite(divergence).all())
        self.assertGreater(divergence.item(), 1.0)


class SurpriseWeightTests(unittest.TestCase):
    def test_weights_average_to_one(self) -> None:
        # The total sample budget must not change, or this becomes a
        # learning-rate change wearing a disguise.
        divergence = torch.rand(512) * 3.0
        weights = surprise_weights(divergence)
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_every_position_keeps_a_baseline(self) -> None:
        divergence = torch.tensor([0.0, 0.0, 5.0])
        weights = surprise_weights(divergence, uniform_share=0.5)
        self.assertTrue(bool((weights >= 0.5).all()))

    def test_surprise_gets_more_weight(self) -> None:
        weights = surprise_weights(torch.tensor([0.1, 2.0]))
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_uniform_share_of_one_is_plain_uniform(self) -> None:
        weights = surprise_weights(torch.tensor([0.0, 9.0]), uniform_share=1.0)
        torch.testing.assert_close(weights, torch.ones(2))

    def test_outliers_are_capped(self) -> None:
        weights = surprise_weights(torch.tensor([0.0] * 99 + [1e6]), maximum=8.0)
        self.assertLessEqual(weights.max().item(), 8.0)

    def test_no_disagreement_falls_back_to_uniform(self) -> None:
        weights = surprise_weights(torch.zeros(16))
        torch.testing.assert_close(weights, torch.ones(16))

    def test_rejects_an_invalid_share(self) -> None:
        with self.assertRaises(ValueError):
            surprise_weights(torch.ones(4), uniform_share=1.5)


class SampleCountTests(unittest.TestCase):
    def test_integer_weights_repeat_exactly(self) -> None:
        counts = sample_counts(torch.tensor([0.0, 1.0, 3.0]))
        self.assertEqual(counts.tolist(), [0, 1, 3])

    def test_fractional_weights_are_unbiased(self) -> None:
        generator = torch.Generator().manual_seed(0)
        weights = torch.full((20000,), 2.25)
        counts = sample_counts(weights, generator)
        self.assertAlmostEqual(counts.float().mean().item(), 2.25, places=1)

    def test_row_index_repeats_by_count(self) -> None:
        generator = torch.Generator().manual_seed(1)
        index = weighted_row_index(torch.tensor([2.0, 0.0, 1.0]), generator)
        self.assertEqual(index.tolist(), [0, 0, 2])


if __name__ == "__main__":
    unittest.main()
