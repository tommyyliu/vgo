import unittest

import torch

from vgo_training.dataset import RasterDataset
from vgo_training.model import RasterPolicyValueNet
from vgo_training.train_demo import metrics


def fixture(samples: int, channels: int = 4, height: int = 8, width: int = 8) -> RasterDataset:
    generator = torch.Generator().manual_seed(4242)
    policy_size = height * width + 1
    policies = torch.rand(samples, policy_size, generator=generator)
    masks = torch.rand(samples, policy_size, generator=generator) > 0.25
    masks[:, 0] = True                                    # never mask every action
    policies = policies * masks
    policies = policies / policies.sum(dim=1, keepdim=True)
    return RasterDataset(
        states=torch.rand(samples, channels, height, width, generator=generator),
        policies=policies,
        policy_masks=masks,
        visits=policies.clone(),
        betas=torch.zeros(samples, policy_size),
        values=torch.rand(samples, generator=generator) * 2 - 1,
        selected_actions=torch.zeros(samples, dtype=torch.int64),
        game_ids=torch.zeros(samples, dtype=torch.int64),
        plies=torch.zeros(samples, dtype=torch.int64),
        seeds=torch.zeros(samples, dtype=torch.int64),
        height=height,
        width=width,
        sources=[],
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
        self.model = RasterPolicyValueNet(channels=4, width=8, blocks=1)
        self.device = torch.device("cpu")

    def test_batched_matches_single_pass(self) -> None:
        dataset = fixture(64)
        whole = metrics(self.model, dataset, self.device, dataset.samples)
        for batch_size in (1, 3, 7, 16, 63, 128):
            batched = metrics(self.model, dataset, self.device, batch_size)
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
        whole = metrics(self.model, dataset, self.device, dataset.samples)
        batched = metrics(self.model, dataset, self.device, 16)
        for key, expected in whole.items():
            self.assertAlmostEqual(batched[key], expected, places=5, msg=key)

    def test_rejects_degenerate_arguments(self) -> None:
        dataset = fixture(4)
        with self.assertRaises(ValueError):
            metrics(self.model, dataset, self.device, 0)
        with self.assertRaises(ValueError):
            metrics(self.model, fixture(0), self.device, 8)


if __name__ == "__main__":
    unittest.main()
