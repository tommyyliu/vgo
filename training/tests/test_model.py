from pathlib import Path
import argparse
import copy
import tempfile
import unittest

import torch

from vgo_training.model import (
    DDRNetPolicyValueNet,
    RasterPolicyValueNet,
    build_model,
)
from vgo_training.serve import load_model
from vgo_training.train_demo import (
    DIHEDRAL_TRANSFORMS,
    apply_dihedral,
    build_scheduler,
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

    def test_ddrnet_handles_odd_rasters_and_backpropagates_both_fusions(self) -> None:
        model = DDRNetPolicyValueNet(channels=10, width=16, blocks=2)
        policy, value = model(torch.randn(2, 10, 19, 23))
        self.assertEqual(tuple(policy.shape), (2, 19 * 23 + 1))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

        (policy.square().mean() + value.square().mean()).backward()
        for parameter in (
            model.context_to_detail1.weight,
            model.detail_to_context1.weight,
            model.context_to_detail2.weight,
            model.detail_to_context2[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_ddrnet_emits_decoupled_policy_grid(self) -> None:
        model = build_model(
            "ddrnet",
            channels=10,
            width=16,
            blocks=2,
            policy_resolution=7,
        )
        policy, value = model(torch.zeros(1, 10, 31, 31))
        self.assertIsInstance(model, DDRNetPolicyValueNet)
        self.assertEqual(tuple(policy.shape), (1, 7 * 7 + 1))
        self.assertEqual(tuple(value.shape), (1,))

    def test_ddrnet_averages_when_policy_grid_is_smaller(self) -> None:
        logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        resized = DDRNetPolicyValueNet._resize_policy(logits, (2, 2))
        expected = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]])
        torch.testing.assert_close(resized, expected)

    def test_ddrnet_pools_before_mixed_axis_policy_resize(self) -> None:
        logits = torch.zeros(1, 1, 4, 8)
        logits[0, 0, 1, 2] = 1.0
        logits[0, 0, 2, 5] = 1.0
        resized = DDRNetPolicyValueNet._resize_policy(logits, (8, 3))
        pooled = torch.nn.functional.adaptive_avg_pool2d(logits, (4, 3))
        expected = torch.nn.functional.interpolate(
            pooled, size=(8, 3), mode="bilinear", align_corners=False
        )
        torch.testing.assert_close(resized, expected)
        self.assertGreater(float(resized.abs().sum()), 0.0)

    def test_ddrnet_checkpoint_round_trips_through_loader(self) -> None:
        torch.manual_seed(42)
        model = DDRNetPolicyValueNet(
            channels=10, width=16, blocks=2, policy_resolution=7
        ).eval()
        states = torch.randn(2, 10, 16, 16)
        expected = model(states)
        checkpoint = {
            "schema": "vgo.raster-policy-value.v1",
            "architecture": "ddrnet",
            "channels": 10,
            "height": 16,
            "width": 16,
            "policy_resolution": 7,
            "model_width": 16,
            "blocks": 2,
            "state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ddrnet.pt"
            torch.save(checkpoint, path)
            loaded, metadata = load_model(path)
        actual = loaded(states)
        self.assertIsInstance(loaded, DDRNetPolicyValueNet)
        self.assertEqual(metadata["architecture"], "ddrnet")
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_unknown_architecture_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model architecture"):
            build_model("unknown", channels=10, width=8, blocks=1)

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
    """Checkpoint selection must let a genuinely better model win.

    This is scored as `policy_kl + value_weight * value_mae`, which only works
    when the two terms can outvote each other. At value_weight 0.1 it discarded
    whole training runs twice: policy_kl sits near 1.8 and drifts while
    value_mae improves 30-50%, but a 0.1 weight caps the value term's total
    contribution at 0.1, so any KL drift above ~10% of the value gain makes the
    untrained epoch-zero weights win. Both resulting checkpoints lost their
    promotion arenas. These tests use the real numbers from those runs.
    """

    @staticmethod
    def _score(current, value_weight):
        return current["policy_kl"] + value_weight * current["value_mae"]

    def test_rising_kl_does_not_veto_a_large_value_gain(self) -> None:
        # decoupled-run3 iteration 1: KL degraded 6% while value improved 42%.
        initial = {"policy_kl": 1.7654, "value_mae": 0.4374}
        trained = {"policy_kl": 1.8945, "value_mae": 0.2872}
        self.assertLess(
            self._score(initial, 0.1),
            self._score(trained, 0.1),
            "sanity: at 0.1 the untrained weights win, which is the bug",
        )
        self.assertLess(
            self._score(trained, 1.0),
            self._score(initial, 1.0),
            "at 1.0 the trained weights must win",
        )

    def test_flat_kl_does_not_veto_value_learning(self) -> None:
        # decoupled-run2 iteration 2: KL flat within noise, value improved 35%.
        initial = {"policy_kl": 1.7842, "value_mae": 0.5452}
        trained = {"policy_kl": 1.7904, "value_mae": 0.3570}
        self.assertLess(self._score(trained, 1.0), self._score(initial, 1.0))

    def test_policy_improvement_still_counts(self) -> None:
        initial = {"policy_kl": 2.0, "value_mae": 1.0}
        better_policy = {"policy_kl": 1.5, "value_mae": 1.0}
        worse_policy = {"policy_kl": 2.5, "value_mae": 0.9}
        self.assertLess(
            self._score(better_policy, 1.0), self._score(worse_policy, 1.0)
        )


class ScheduleTests(unittest.TestCase):
    """WSD exists so that training longer does not reshape the curve.

    Cosine ties its shape to `--epochs`, so a run that is still improving at
    the end has already annealed its rate away, and extending it re-tunes
    every epoch's rate. WSD only moves the boundary between a constant
    stable phase and a fixed trailing decay.
    """

    @staticmethod
    def _rates(epochs: int, **overrides: object) -> list[float]:
        arguments = argparse.Namespace(
            learning_rate=2e-3,
            epochs=epochs,
            schedule="wsd",
            warmup_epochs=5,
            decay_fraction=0.2,
            final_learning_rate_fraction=0.01,
        )
        for name, value in overrides.items():
            setattr(arguments, name, value)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([parameter], lr=arguments.learning_rate)
        scheduler = build_scheduler(optimizer, arguments)
        rates = []
        for _ in range(epochs):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return rates

    def test_wsd_warms_up_holds_then_decays(self) -> None:
        rates = self._rates(150)
        self.assertTrue(all(a < b for a, b in zip(rates[:5], rates[1:6])))
        self.assertEqual(len({round(rate, 12) for rate in rates[5:120]}), 1)
        self.assertAlmostEqual(rates[60], 2e-3)
        self.assertTrue(all(a > b for a, b in zip(rates[120:-1], rates[121:])))

    def test_wsd_never_exceeds_base_rate_or_drops_below_floor(self) -> None:
        rates = self._rates(150)
        self.assertLessEqual(max(rates), 2e-3)
        self.assertGreaterEqual(min(rates), 2e-5 - 1e-15)

    def test_longer_runs_only_extend_the_stable_phase(self) -> None:
        # The point of WSD: the decay window keeps its shape and the extra
        # epochs all land at the full rate, so --epochs is a free knob.
        for epochs, expected_stable in ((50, 35), (300, 235), (600, 475)):
            rates = self._rates(epochs)
            stable = sum(1 for rate in rates if abs(rate - 2e-3) < 1e-12)
            self.assertEqual(stable, expected_stable)
            self.assertAlmostEqual(rates[-1], 2e-5)

    def test_degenerate_settings_still_produce_a_usable_rate(self) -> None:
        for epochs, warmup, decay in ((1, 5, 0.2), (3, 5, 0.2), (10, 0, 1.0)):
            rates = self._rates(epochs, warmup_epochs=warmup, decay_fraction=decay)
            self.assertEqual(len(rates), epochs)
            self.assertGreater(min(rates), 0.0)
            self.assertLessEqual(max(rates), 2e-3)

    def test_cosine_remains_available_unchanged(self) -> None:
        rates = self._rates(150, schedule="cosine")
        self.assertAlmostEqual(rates[0], 2e-3)
        self.assertTrue(all(a >= b for a, b in zip(rates[:-1], rates[1:])))


class CompileTests(unittest.TestCase):
    def test_in_place_compile_leaves_state_dict_keys_unchanged(self) -> None:
        """`Module.compile()` must not rename parameters.

        `torch.compile(model)` returns a wrapper whose state_dict prefixes every
        key with `_orig_mod.`. Saving that would produce checkpoints neither
        `serve.load_model` nor the ONNX exporter can read, so training compiles
        in place instead.
        """
        model = build_model("ddrnet", channels=10, width=8, blocks=1, policy_resolution=5)
        before = set(model.state_dict())
        self.assertIsNone(model.compile())
        self.assertEqual(set(model.state_dict()), before)

        wrapped = torch.compile(build_model("flat", channels=10, width=8, blocks=1))
        self.assertTrue(any(key.startswith("_orig_mod") for key in wrapped.state_dict()))


class OptimizerStateTests(unittest.TestCase):
    """Adam moments must survive an iteration boundary.

    They start at zero and beta2=0.999 needs ~2000 steps of history; a 10-epoch
    RL iteration is only ~4300 steps, so a cold start wastes a large share of
    every short run. The 150-epoch iterations amortized this away, which is why
    it went unnoticed.
    """

    @staticmethod
    def _stepped_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = build_model("flat", channels=10, width=8, blocks=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            policy, value = model(torch.randn(2, 10, 8, 8))
            (policy.square().mean() + value.square().mean()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_round_trip_preserves_moments(self) -> None:
        model, optimizer = self._stepped_optimizer()
        saved = copy.deepcopy(optimizer.state_dict())
        restored = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.assertEqual(len(restored.state_dict()["state"]), 0)
        restored.load_state_dict(saved)
        self.assertEqual(
            len(restored.state_dict()["state"]), len(saved["state"])
        )
        first = next(iter(saved["state"].values()))
        other = next(iter(restored.state_dict()["state"].values()))
        torch.testing.assert_close(first["exp_avg"], other["exp_avg"])
        torch.testing.assert_close(first["exp_avg_sq"], other["exp_avg_sq"])
        self.assertGreater(int(first["step"]), 0)

    def test_mismatched_state_is_rejected_not_silently_applied(self) -> None:
        # A different architecture has a different parameter count, so restoring
        # must raise rather than quietly produce nonsense; train_demo catches
        # this and falls back to a cold optimizer.
        _, optimizer = self._stepped_optimizer()
        wider = build_model("flat", channels=10, width=16, blocks=2)
        target = torch.optim.Adam(wider.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            target.load_state_dict(optimizer.state_dict())
