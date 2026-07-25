import unittest

import torch

from vgo_training.model import RasterPolicyValueNet
from vgo_training.train_demo import (
    DIHEDRAL_TRANSFORMS,
    apply_dihedral,
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


class DihedralAugmentationTests(unittest.TestCase):
    """A wrong reindex here silently corrupts every policy target, so these check
    the state and the target move together rather than just that shapes survive."""

    height = 4
    width = 4

    def _fixture(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cells = self.height * self.width
        # Each state pixel holds its own flat index, so we can follow where it went.
        states = (
            torch.arange(cells, dtype=torch.float32)
            .reshape(1, 1, self.height, self.width)
            .repeat(2, 1, 1, 1)
        )
        policies = torch.zeros(2, cells + 1)
        policies[:, 5] = 1.0
        policies[:, -1] = 0.25
        masks = torch.ones(2, cells + 1)
        return states, policies, masks

    def test_state_and_policy_stay_aligned(self) -> None:
        states, policies, masks = self._fixture()
        cells = self.height * self.width
        for transform in range(len(DIHEDRAL_TRANSFORMS)):
            moved_states, moved_policies, _ = apply_dihedral(
                states, policies, masks, transform, self.height, self.width
            )
            cell = int(moved_policies[0, :cells].argmax())
            row, column = divmod(cell, self.width)
            self.assertEqual(
                float(moved_states[0, 0, row, column]),
                5.0,
                f"transform {transform} moved the policy and the state differently",
            )

    def test_pass_is_invariant_and_mass_conserved(self) -> None:
        states, policies, masks = self._fixture()
        for transform in range(len(DIHEDRAL_TRANSFORMS)):
            _, moved, _ = apply_dihedral(
                states, policies, masks, transform, self.height, self.width
            )
            self.assertAlmostEqual(float(moved[0, -1]), 0.25, places=6)
            self.assertAlmostEqual(float(moved[0].sum()), 1.25, places=6)

    def test_transforms_are_distinct(self) -> None:
        # Cell (0, 1) is off both diagonals and off centre, so its orbit under the
        # dihedral group has all eight elements. A cell on a diagonal (such as the
        # (1, 1) used by the other tests) is fixed by a reflection and would only
        # produce four images -- correct behaviour, but useless for this check.
        states, _, masks = self._fixture()
        cells = self.height * self.width
        policies = torch.zeros(2, cells + 1)
        policies[:, 1] = 1.0
        seen = set()
        for transform in range(len(DIHEDRAL_TRANSFORMS)):
            _, moved, _ = apply_dihedral(
                states, policies, masks, transform, self.height, self.width
            )
            seen.add(int(moved[0, :cells].argmax()))
        self.assertEqual(len(seen), 8, "the eight symmetries must map a cell eight ways")

    def test_identity_transform_is_a_passthrough(self) -> None:
        states, policies, masks = self._fixture()
        moved_states, moved_policies, _ = apply_dihedral(
            states, policies, masks, 0, self.height, self.width
        )
        self.assertTrue(torch.equal(moved_states, states))
        self.assertTrue(torch.equal(moved_policies, policies))

    def test_non_square_raster_is_rejected(self) -> None:
        states, policies, masks = self._fixture()
        with self.assertRaises(ValueError):
            apply_dihedral(states, policies, masks, 1, 2, 8)


if __name__ == "__main__":
    unittest.main()


class SelectionMetricTests(unittest.TestCase):
    """Checkpoint selection must not let a term that carries no signal outvote
    one that does.

    Measured on a real iteration: policy_kl sat at ~1.79 and jittered +/-0.008
    across 80 epochs while the whole value signal spanned 0.019. Summing raw
    terms meant KL noise outranked genuine value learning, epoch-zero weights
    were saved, and that checkpoint then lost its promotion arena.
    """

    @staticmethod
    def _score(current, initial, value_weight):
        # Mirrors train_demo.selection_score, which is a closure over the run's
        # initial validation and so cannot be imported directly.
        policy = current["policy_kl"] / max(initial["policy_kl"], 1e-9)
        value = current["value_mae"] / max(initial["value_mae"], 1e-9)
        return policy + value_weight * value

    def test_flat_policy_does_not_outvote_improving_value(self) -> None:
        initial = {"policy_kl": 1.7842, "value_mae": 0.5452}
        # Real epoch-80 numbers from the iteration that regressed.
        final = {"policy_kl": 1.7904, "value_mae": 0.3570}
        raw_initial = initial["policy_kl"] + 0.1 * initial["value_mae"]
        raw_final = final["policy_kl"] + 0.1 * final["value_mae"]
        self.assertLess(
            raw_final,
            raw_initial,
            "sanity: on these numbers the raw metric does prefer the final epoch",
        )
        # The failure mode is a *lucky early* KL sample beating everything later.
        lucky = {"policy_kl": 1.7700, "value_mae": 0.5400}
        self.assertLess(
            lucky["policy_kl"] + 0.1 * lucky["value_mae"],
            raw_final,
            "raw metric prefers the lucky early epoch over trained weights",
        )
        self.assertLess(
            self._score(final, initial, 0.1),
            self._score(lucky, initial, 0.1),
            "relative metric must prefer the trained weights",
        )

    def test_improving_policy_still_wins(self) -> None:
        initial = {"policy_kl": 2.0, "value_mae": 1.0}
        better = {"policy_kl": 1.5, "value_mae": 1.0}
        worse = {"policy_kl": 2.2, "value_mae": 0.9}
        self.assertLess(
            self._score(better, initial, 0.1), self._score(worse, initial, 0.1)
        )
