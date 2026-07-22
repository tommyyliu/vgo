import unittest

import torch

from vgo_training.model import RasterPolicyValueNet
from vgo_training.train_demo import policy_cross_entropy


class ModelTests(unittest.TestCase):
    def test_policy_and_value_shapes(self) -> None:
        model = RasterPolicyValueNet(channels=10, width=8, blocks=1)
        policy, value = model(torch.zeros(2, 10, 8, 8))
        self.assertEqual(tuple(policy.shape), (2, 65))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_policy_loss_ignores_unsampled_pixels(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 100.0]], requires_grad=True)
        targets = torch.tensor([[0.25, 0.75, 0.0]])
        mask = torch.tensor([[True, True, False]])
        loss = policy_cross_entropy(logits, targets, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
