import unittest

import torch

from vgo_training.model import RasterPolicyValueNet
from vgo_training.train_demo import (
    full_legal_policy_masks,
    importance_corrected_policy_targets,
    policy_cross_entropy,
    sampled_policy_loss,
)


class ModelTests(unittest.TestCase):
    def test_policy_and_value_shapes(self) -> None:
        model = RasterPolicyValueNet(channels=10, width=8, blocks=1)
        policy, value = model(torch.zeros(2, 10, 8, 8))
        self.assertEqual(tuple(policy.shape), (2, 65))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_full_legal_loss_pushes_down_unexplored_legal_cells(self) -> None:
        states = torch.zeros(1, 10, 1, 3)
        states[:, 7] = torch.tensor([[[0.5, 0.25, -0.5]]])
        logits = torch.tensor([[1.0, 2.0, 100.0, 0.5]], requires_grad=True)
        visits = torch.tensor([[3.0, 0.0, 0.0, 1.0]])
        betas = torch.tensor([[0.25, 0.0, 0.0, 0.0]])
        proposal_counts = torch.zeros_like(visits, dtype=torch.uint32)
        explored = torch.tensor([[True, False, False, True]])
        loss = sampled_policy_loss(
            logits, states, visits, betas, proposal_counts, explored
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)

    def test_full_legal_cross_entropy_matches_manual_logsumexp(self) -> None:
        logits = torch.tensor([[1.0, -0.5, 2.0, 0.25]])
        targets = torch.tensor([[0.75, 0.0, 0.0, 0.25]])
        mask = torch.tensor([[True, True, False, True]])
        actual = policy_cross_entropy(logits, targets, mask)
        denominator = torch.logsumexp(logits[0, mask[0]], dim=0)
        expected = -(
            targets[0, 0] * (logits[0, 0] - denominator)
            + targets[0, 3] * (logits[0, 3] - denominator)
        )
        torch.testing.assert_close(actual, expected)

    def test_importance_correction_uses_multiplicity_and_deterministic_pass(self) -> None:
        visits = torch.tensor([[4.0, 2.0, 0.0, 1.0]])
        betas = torch.tensor([[0.25, 0.5, 0.1, 0.0]])
        proposal_counts = torch.tensor([[2, 1, 1, 0]], dtype=torch.uint32)
        explored = torch.ones_like(visits, dtype=torch.bool)
        target = importance_corrected_policy_targets(
            visits, betas, proposal_counts, explored
        )
        weights = torch.tensor(
            [[4.0 * 2.0 / (4.0 * 0.25), 2.0 / (4.0 * 0.5), 0.0, 1.0]]
        )
        torch.testing.assert_close(target, weights / weights.sum(dim=1, keepdim=True))
        self.assertEqual(float(target[0, 2]), 0.0)

    def test_pre_v3_target_is_normalized_visits_even_when_beta_is_present(self) -> None:
        visits = torch.tensor([[2.0, 0.0, 6.0]])
        target = importance_corrected_policy_targets(
            visits,
            torch.tensor([[0.25, 0.0, 0.0]]),
            torch.zeros_like(visits, dtype=torch.uint32),
            torch.tensor([[True, False, True]]),
        )
        torch.testing.assert_close(target, torch.tensor([[0.25, 0.0, 0.75]]))

    def test_tiny_beta_and_pass_only_targets_are_finite(self) -> None:
        tiny = importance_corrected_policy_targets(
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[torch.finfo(torch.float32).tiny, 0.0]]),
            torch.tensor([[1, 0]], dtype=torch.uint32),
            torch.tensor([[True, True]]),
        )
        self.assertTrue(bool(torch.isfinite(tiny).all()))
        torch.testing.assert_close(tiny.sum(dim=1), torch.ones(1))

        pass_only = importance_corrected_policy_targets(
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.uint32),
            torch.tensor([[False, True]]),
        )
        torch.testing.assert_close(pass_only, torch.tensor([[0.0, 1.0]]))

    def test_full_legal_mask_uses_clearance_pass_and_explored_aliases(self) -> None:
        states = torch.zeros(1, 10, 1, 3)
        states[:, 7] = torch.tensor([[[-0.5, 0.0, 0.5]]])
        explored = torch.tensor([[True, False, False, False]])
        self.assertEqual(
            full_legal_policy_masks(states, explored).tolist(),
            [[True, True, True, True]],
        )

    def test_malformed_importance_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal positive-beta"):
            importance_corrected_policy_targets(
                torch.tensor([[1.0, 1.0, 0.0]]),
                torch.tensor([[0.5, 0.0, 0.0]]),
                torch.tensor([[0, 1, 0]], dtype=torch.uint32),
                torch.tensor([[True, True, True]]),
            )
        with self.assertRaisesRegex(ValueError, "at least one visit"):
            importance_corrected_policy_targets(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2, dtype=torch.uint32),
                torch.ones(1, 2, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
