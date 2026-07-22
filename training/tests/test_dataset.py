from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from vgo_training.dataset import HEADER, MAGIC, VERSION, load_dataset


class DatasetTests(unittest.TestCase):
    def test_loads_versioned_record(self) -> None:
        samples, channels, height, width = 2, 3, 2, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        records = np.zeros((samples, state_size + 2 * policy_size + 1), dtype="<f4")
        records[:, state_size] = 1.0
        records[:, state_size + policy_size] = 1.0
        records[:, -1] = np.array([-1.0, 1.0], dtype="<f4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        MAGIC,
                        VERSION,
                        samples,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            dataset = load_dataset(path)

        self.assertEqual(tuple(dataset.states.shape), (2, 3, 2, 2))
        self.assertEqual(tuple(dataset.policies.shape), (2, 5))
        self.assertEqual(tuple(dataset.policy_masks.shape), (2, 5))
        self.assertEqual(dataset.values.tolist(), [-1.0, 1.0])

    def test_rejects_bad_magic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.vgo"
            path.write_bytes(struct.pack("<8s6I", b"NOTVGO!!", 1, 1, 1, 1, 1, 2))
            with self.assertRaisesRegex(ValueError, "magic"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
