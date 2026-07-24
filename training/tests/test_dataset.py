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
        self.assertEqual(dataset.proposal_counts.tolist(), [[0] * 5, [0] * 5])
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
        self.assertEqual(dataset.proposal_counts.tolist(), [[0] * 5, [0] * 5])
        self.assertEqual(combined.samples, 4)
        self.assertEqual(tuple(combined.proposal_counts.shape), (4, 5))

    def test_loads_version_two_visits_and_sampling_probabilities(self) -> None:
        samples, channels, height, width = 1, 10, 1, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        dtype = np.dtype(
            [
                ("state", "<f4", (state_size,)),
                ("policy", "<f4", (policy_size,)),
                ("mask", "<f4", (policy_size,)),
                ("visits", "<f4", (policy_size,)),
                ("beta", "<f4", (policy_size,)),
                ("value", "<f4"),
                ("selected_action", "<u4"),
                ("game", "<u8"),
                ("ply", "<u4"),
                ("seed", "<u8"),
            ],
            align=False,
        )
        records = np.zeros(samples, dtype=dtype)
        records["visits"][0] = [2.0, 1.0, 1.0]
        records["policy"][0] = [0.5, 0.25, 0.25]
        records["mask"][0] = [1.0, 1.0, 1.0]
        records["beta"][0] = [0.25, 0.5, 0.0]
        records["selected_action"][0] = 0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        REPLAY_MAGIC,
                        2,
                        samples,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            path.with_name("manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "vgo.replay-shard.v1",
                        "dataset_sha256": file_sha256(path),
                    }
                ),
                encoding="ascii",
            )
            dataset = load_dataset(path)

        self.assertEqual(dataset.visits.tolist(), [[2.0, 1.0, 1.0]])
        self.assertEqual(dataset.betas.tolist(), [[0.25, 0.5, 0.0]])
        self.assertEqual(dataset.proposal_counts.tolist(), [[0, 0, 0]])

    def test_rejects_inconsistent_version_two_sampling_fields(self) -> None:
        channels, height, width = 10, 1, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        dtype = np.dtype(
            [
                ("state", "<f4", (state_size,)),
                ("policy", "<f4", (policy_size,)),
                ("mask", "<f4", (policy_size,)),
                ("visits", "<f4", (policy_size,)),
                ("beta", "<f4", (policy_size,)),
                ("value", "<f4"),
                ("selected_action", "<u4"),
                ("game", "<u8"),
                ("ply", "<u4"),
                ("seed", "<u8"),
            ],
            align=False,
        )

        def assert_invalid(
            records: np.ndarray, message: str, directory: Path
        ) -> None:
            directory.mkdir()
            path = directory / "dataset.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        REPLAY_MAGIC,
                        2,
                        1,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            path.with_name("manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "vgo.replay-shard.v1",
                        "dataset_sha256": file_sha256(path),
                    }
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, message):
                load_dataset(path)

        base = np.zeros(1, dtype=dtype)
        base["policy"][0] = [0.5, 0.5, 0.0]
        base["mask"][0] = [1.0, 1.0, 1.0]
        base["visits"][0] = [1.0, 1.0, 0.0]
        base["selected_action"][0] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_beta = base.copy()
            bad_beta["beta"][0, 0] = 1.5
            assert_invalid(bad_beta, r"in \[0, 1\]", root / "bad-beta")

            bad_policy = base.copy()
            bad_policy["visits"][0] = [3.0, 1.0, 0.0]
            assert_invalid(bad_policy, "normalized visit", root / "bad-policy")

    def test_loads_version_three_proposal_multiplicities(self) -> None:
        samples, channels, height, width = 1, 10, 1, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        dtype = np.dtype(
            [
                ("state", "<f4", (state_size,)),
                ("policy", "<f4", (policy_size,)),
                ("mask", "<f4", (policy_size,)),
                ("visits", "<f4", (policy_size,)),
                ("beta", "<f4", (policy_size,)),
                ("proposal_counts", "<u4", (policy_size,)),
                ("value", "<f4"),
                ("selected_action", "<u4"),
                ("game", "<u8"),
                ("ply", "<u4"),
                ("seed", "<u8"),
            ],
            align=False,
        )
        records = np.zeros(samples, dtype=dtype)
        records["visits"][0] = [2.0, 1.0, 1.0]
        records["policy"][0] = [0.5, 0.25, 0.25]
        records["mask"][0] = [1.0, 1.0, 1.0]
        records["beta"][0] = [0.25, 0.5, 0.0]
        records["proposal_counts"][0] = [2, 1, 0]
        records["selected_action"][0] = 0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        REPLAY_MAGIC,
                        REPLAY_VERSION,
                        samples,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            path.with_name("manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "vgo.replay-shard.v1",
                        "dataset_sha256": file_sha256(path),
                    }
                ),
                encoding="ascii",
            )
            dataset = load_dataset(path)

        self.assertEqual(dataset.proposal_counts.tolist(), [[2, 1, 0]])

    def test_rejects_invalid_version_three_proposal_counts(self) -> None:
        channels, height, width = 10, 1, 2
        policy_size = height * width + 1
        state_size = channels * height * width
        dtype = np.dtype(
            [
                ("state", "<f4", (state_size,)),
                ("policy", "<f4", (policy_size,)),
                ("mask", "<f4", (policy_size,)),
                ("visits", "<f4", (policy_size,)),
                ("beta", "<f4", (policy_size,)),
                ("proposal_counts", "<u4", (policy_size,)),
                ("value", "<f4"),
                ("selected_action", "<u4"),
                ("game", "<u8"),
                ("ply", "<u4"),
                ("seed", "<u8"),
            ],
            align=False,
        )

        def assert_invalid(
            records: np.ndarray, message: str, directory: Path
        ) -> None:
            directory.mkdir()
            path = directory / "dataset.vgo"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(
                        REPLAY_MAGIC,
                        REPLAY_VERSION,
                        1,
                        channels,
                        height,
                        width,
                        policy_size,
                    )
                )
                stream.write(records.tobytes())
            path.with_name("manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "vgo.replay-shard.v1",
                        "dataset_sha256": file_sha256(path),
                    }
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, message):
                load_dataset(path)

        base = np.zeros(1, dtype=dtype)
        base["policy"][0] = [0.5, 0.5, 0.0]
        base["mask"][0] = [1.0, 1.0, 1.0]
        base["visits"][0] = [1.0, 1.0, 0.0]
        base["beta"][0] = [0.25, 0.5, 0.0]
        base["proposal_counts"][0] = [1, 1, 0]
        base["selected_action"][0] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            unsupported_count = base.copy()
            unsupported_count["mask"][0, 1] = 0.0
            unsupported_count["policy"][0] = [1.0, 0.0, 0.0]
            unsupported_count["visits"][0] = [1.0, 0.0, 0.0]
            unsupported_count["beta"][0, 1] = 0.0
            assert_invalid(
                unsupported_count,
                "proposal counts must be included",
                root / "unsupported-count",
            )

            pass_count = base.copy()
            pass_count["proposal_counts"][0, -1] = 1
            assert_invalid(
                pass_count,
                "pass action must have proposal count zero",
                root / "pass-count",
            )

            mismatched_support = base.copy()
            mismatched_support["proposal_counts"][0, 1] = 0
            assert_invalid(
                mismatched_support,
                "support must equal",
                root / "mismatched-support",
            )


if __name__ == "__main__":
    unittest.main()
