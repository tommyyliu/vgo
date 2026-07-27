import gc
import unittest
import weakref

import torch

from vgo_training.dataset import PreparedRasterDataset, RasterDataset
from vgo_training.model import RasterPolicyValueNet
from vgo_training.train_demo import metrics, prepare_policy_supervision, subset


def raw_fixture(
    samples: int, channels: int = 10, height: int = 8, width: int = 8
) -> RasterDataset:
    generator = torch.Generator().manual_seed(4242)
    policy_size = height * width + 1
    policies = torch.rand(samples, policy_size, generator=generator)
    masks = torch.rand(samples, policy_size, generator=generator) > 0.25
    masks[:, 0] = True                                    # never mask every action
    policies = policies * masks
    policies = policies / policies.sum(dim=1, keepdim=True)
    betas = torch.rand(samples, policy_size, generator=generator) * masks
    betas[:, -1] = 0.0
    proposal_counts = masks.to(torch.uint32)
    proposal_counts[:, -1] = 0
    return RasterDataset(
        states=torch.rand(samples, channels, height, width, generator=generator),
        policies=policies,
        policy_masks=masks,
        visits=policies.clone(),
        betas=betas,
        proposal_counts=proposal_counts,
        values=torch.rand(samples, generator=generator) * 2 - 1,
        selected_actions=torch.zeros(samples, dtype=torch.int64),
        game_ids=torch.zeros(samples, dtype=torch.int64),
        plies=torch.zeros(samples, dtype=torch.int64),
        seeds=torch.zeros(samples, dtype=torch.int64),
        height=height,
        width=width,
        sources=(),
    )


def fixture(
    samples: int, channels: int = 10, height: int = 8, width: int = 8
) -> PreparedRasterDataset:
    return prepare_policy_supervision(
        raw_fixture(samples, channels, height, width),
        batch_size=16,
    )


class MetricsBatchingTests(unittest.TestCase):
    """Batched evaluation must not change any reported number.

    Evaluating a split in one forward pass made evaluation memory scale with the
    replay window instead of the batch size, which capped the window regardless
    of device capacity. Batching is only safe because every metric is a mean over
    samples, so these tests pin that equivalence.
    """

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = RasterPolicyValueNet(channels=10, width=8, blocks=1)
        self.device = torch.device("cpu")
        self.value_weight = 0.25

    def test_batched_matches_single_pass(self) -> None:
        dataset = fixture(64)
        whole = metrics(
            self.model,
            dataset,
            self.device,
            dataset.samples,
            value_weight=self.value_weight,
        )
        for batch_size in (1, 3, 7, 16, 63, 128):
            batched = metrics(
                self.model,
                dataset,
                self.device,
                batch_size,
                value_weight=self.value_weight,
            )
            self.assertEqual(sorted(batched), sorted(whole))
            for key, expected in whole.items():
                self.assertAlmostEqual(
                    batched[key],
                    expected,
                    places=5,
                    msg=f"{key} drifted at batch size {batch_size}",
                )

    def test_ragged_final_batch_is_weighted_correctly(self) -> None:
        # 65 samples at batch 16 leaves a final batch of 1; an unweighted mean of
        # per-batch means would visibly disagree with the whole-set value.
        dataset = fixture(65)
        whole = metrics(
            self.model,
            dataset,
            self.device,
            dataset.samples,
            value_weight=self.value_weight,
        )
        batched = metrics(
            self.model,
            dataset,
            self.device,
            16,
            value_weight=self.value_weight,
        )
        for key, expected in whole.items():
            self.assertAlmostEqual(batched[key], expected, places=5, msg=key)

    def test_preparation_drops_raw_supervision_and_metrics_stays_cached(self) -> None:
        raw = raw_fixture(8)
        policy_storage = raw.policies
        mask_storage = raw.policy_masks
        raw_refs = tuple(
            weakref.ref(tensor)
            for tensor in (raw.visits, raw.betas, raw.proposal_counts)
        )
        dataset = prepare_policy_supervision(raw, batch_size=3)
        self.assertIs(dataset.policies, policy_storage)
        self.assertIs(dataset.policy_masks, mask_storage)
        prepared_refs = tuple(
            weakref.ref(tensor)
            for tensor in (
                dataset.states,
                dataset.policies,
                dataset.policy_masks,
                dataset.values,
            )
        )
        for field in ("visits", "betas", "proposal_counts"):
            self.assertFalse(hasattr(dataset, field))
        prepared_subset = subset(dataset, torch.tensor([0, 2, 4]))
        for field in ("visits", "betas", "proposal_counts"):
            self.assertFalse(hasattr(prepared_subset, field))

        expected = metrics(
            self.model,
            dataset,
            self.device,
            3,
            value_weight=self.value_weight,
        )
        raw.visits.fill_(torch.nan)
        raw.betas.fill_(torch.nan)
        raw.proposal_counts.zero_()
        self.assertEqual(
            metrics(
                self.model,
                dataset,
                self.device,
                3,
                value_weight=self.value_weight,
            ),
            expected,
        )

        # `prepared_subset` is a view and deliberately keeps the base tensors
        # alive -- that is how a split avoids copying the replay window. So the
        # prepared storage is released once no view refers to it, not merely
        # when the dataset name goes out of scope.
        del raw, dataset, policy_storage, mask_storage
        gc.collect()
        self.assertTrue(all(reference() is None for reference in raw_refs))
        self.assertTrue(all(reference() is not None for reference in prepared_refs))

        del prepared_subset
        gc.collect()
        self.assertTrue(all(reference() is None for reference in prepared_refs))

    def test_metric_loss_uses_training_value_weight(self) -> None:
        dataset = fixture(8)
        policy_only = metrics(
            self.model,
            dataset,
            self.device,
            3,
            value_weight=0.0,
        )
        weighted = metrics(
            self.model,
            dataset,
            self.device,
            3,
            value_weight=self.value_weight,
        )
        with torch.no_grad():
            _, predictions = self.model(dataset.states)
            value_mse = torch.nn.functional.mse_loss(predictions, dataset.values)
        self.assertAlmostEqual(
            weighted["loss"] - policy_only["loss"],
            self.value_weight * float(value_mse),
            places=6,
        )

    def test_rejects_degenerate_arguments(self) -> None:
        dataset = fixture(4)
        with self.assertRaises(ValueError):
            metrics(
                self.model,
                dataset,
                self.device,
                0,
                value_weight=self.value_weight,
            )
        with self.assertRaises(ValueError):
            metrics(
                self.model,
                fixture(0),
                self.device,
                8,
                value_weight=self.value_weight,
            )
        with self.assertRaises(ValueError):
            metrics(self.model, dataset, self.device, 8, value_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
