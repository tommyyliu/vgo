from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from vgo_training.dataset import (
    HEADER,
    MAGIC,
    REPLAY_MAGIC,
    REPLAY_VERSION,
    VERSION,
    file_sha256,
    load_dataset,
    load_datasets,
)


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

    def test_loads_and_combines_audited_replay_shards(self) -> None:
        samples, channels, height, width = 2, 3, 2, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        dtype = np.dtype(
            [
                ("state", "<f4", (state_size,)),
                ("policy", "<f4", (policy_size,)),
                ("mask", "<f4", (policy_size,)),
                ("value", "<f4"),
                ("selected_action", "<u4"),
                ("game", "<u8"),
                ("ply", "<u4"),
                ("seed", "<u8"),
            ],
            align=False,
        )
        records = np.zeros(samples, dtype=dtype)
        records["policy"][:, 1] = 1.0
        records["mask"][:, 1] = 1.0
        records["selected_action"] = 1
        records["game"] = [4, 5]
        records["ply"] = [2, 3]
        records["seed"] = [11, 12]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        REPLAY_MAGIC,
                        1,  # this record uses the v1 layout (no visits/beta)
                        samples,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            manifest = {
                "schema": "vgo.replay-shard.v1",
                "dataset_sha256": file_sha256(path),
            }
            path.with_name("manifest.json").write_text(json.dumps(manifest), encoding="ascii")
            dataset = load_dataset(path)
            combined = load_datasets([path, path])

        self.assertEqual(dataset.selected_actions.tolist(), [1, 1])
        self.assertEqual(dataset.game_ids.tolist(), [4, 5])
        self.assertEqual(dataset.plies.tolist(), [2, 3])
        self.assertEqual(dataset.seeds.tolist(), [11, 12])
        self.assertEqual(combined.samples, 4)


if __name__ == "__main__":
    unittest.main()
