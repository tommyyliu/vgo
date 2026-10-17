from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from vgo_training.model import build_model, load_model


class NormCountTests(unittest.TestCase):
    def test_one_norm_round_trips_checkpoint_and_default_remains_two(self):
        torch.set_num_threads(1)
        def model(count=2):
            return build_model(7, 8, 4, policy_resolution=16, norm_groups=4,
                               raster_resolution=32, norms_per_block=count).eval()
        default = model()
        one = model(1)
        groups = lambda net: sum(isinstance(module, nn.GroupNorm) for module in net.modules())
        self.assertEqual(groups(default), 2 * groups(one))
        states = torch.randn(2, 7, 32, 32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "one.pt"
            torch.save({"architecture": "ddrnet", "channels": 7, "height": 32, "width": 32,
                        "model_width": 8, "blocks": 4, "policy_resolution": 16,
                        "norm_groups": 4, "norms_per_block": 1, "state_dict": one.state_dict()}, path)
            restored, _ = load_model(path)
            self.assertEqual(restored.norms_per_block, 1)
            for original, rebuilt in zip(one(states), restored(states)):
                self.assertTrue(torch.equal(original, rebuilt))
            # Old checkpoints have no count field and must preserve the old function.
            checkpoint = torch.load(path)
            checkpoint.pop("norms_per_block")
            checkpoint["state_dict"] = default.state_dict()
            torch.save(checkpoint, path)
            restored, _ = load_model(path)
            self.assertEqual(restored.norms_per_block, 2)
            checkpoint["norms_per_block"] = 1
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_model(path)

    def test_invalid_count_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model(7, 8, 4, norm_groups=4, norms_per_block=0)
