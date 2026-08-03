from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vgo_training.dataset import RasterDataset
from vgo_training.dataset import file_sha256
from vgo_training.learner import (
    BatchStager,
    LearnerConfig,
    LearnerUpdate,
    PersistentLearner,
    ReplayCache,
    serve_json_lines,
)


def raw_fixture(
    samples: int = 12,
    *,
    height: int = 4,
    width: int = 4,
) -> RasterDataset:
    generator = torch.Generator().manual_seed(123)
    channels = 10
    policy_size = height * width + 1
    masks = torch.ones(samples, policy_size, dtype=torch.bool)
    visits = torch.zeros(samples, policy_size)
    visits[:, 0] = 3.0
    visits[:, -1] = 1.0
    policies = visits / visits.sum(dim=1, keepdim=True)
    states = torch.rand(
        samples, channels, height, width, generator=generator
    )
    states[:, 7] = 1.0
    # Two plies per game exercise the game-stable validation partition.
    games = torch.arange(samples, dtype=torch.int64) // 2
    return RasterDataset(
        states=states,
        policies=policies,
        policy_masks=masks,
        visits=visits,
        betas=torch.zeros(samples, policy_size),
        proposal_counts=torch.zeros(
            samples, policy_size, dtype=torch.uint32
        ),
        values=torch.linspace(-1.0, 1.0, samples),
        selected_actions=torch.zeros(samples, dtype=torch.int64),
        game_ids=games,
        plies=torch.arange(samples, dtype=torch.int64) % 2,
        seeds=games + 100,
        height=height,
        width=width,
        sources=("fixture",),
    )


def cache_fixture(directory: Path) -> tuple[ReplayCache, Path, list[int]]:
    path = directory / "dataset.vgo"
    path.write_bytes(b"immutable replay fixture")
    path.with_name("manifest.json").write_text(
        json.dumps({"dataset_sha256": "ab" * 32}), encoding="utf-8"
    )
    loads: list[int] = []

    def load(_: str | Path) -> RasterDataset:
        loads.append(1)
        return raw_fixture()

    return ReplayCache(loader=load), path, loads


class ReplayCacheTests(unittest.TestCase):
    def test_unchanged_shard_is_prepared_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, path, loads = cache_fixture(Path(temporary))
            first = cache.window([path], 4)
            second = cache.window([path], 2)

        self.assertIs(first.shards[0], second.shards[0])
        self.assertEqual(len(loads), 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(first.shards[0].digest, "ab" * 32)
        for field in ("visits", "betas", "proposal_counts"):
            self.assertFalse(hasattr(first.shards[0].dataset, field))

    def test_validation_split_keeps_every_game_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, path, _ = cache_fixture(Path(temporary))
            window = cache.window([path], 4)
            split = window.split(0.5)
            training_rows = set(split.training.selections[0].rows.tolist())
            validation_rows = set(split.validation.selections[0].rows.tolist())

        self.assertTrue(training_rows)
        self.assertTrue(validation_rows)
        for first in range(0, 12, 2):
            pair = {first, first + 1}
            self.assertTrue(
                pair <= training_rows or pair <= validation_rows,
                f"game rows {pair} crossed the split",
            )

    def test_sliding_window_evicts_before_loading_the_entering_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first" / "dataset.vgo"
            second = root / "second" / "dataset.vgo"
            observed_entries: list[int] = []
            cache: ReplayCache

            for index, path in enumerate((first, second)):
                path.parent.mkdir()
                path.write_bytes(f"shard-{index}".encode())
                path.with_name("manifest.json").write_text(
                    json.dumps({"dataset_sha256": f"{index + 1:02x}" * 32}),
                    encoding="utf-8",
                )

            def load(_: str | Path) -> RasterDataset:
                observed_entries.append(len(cache._entries))
                return raw_fixture()

            cache = ReplayCache(loader=load)
            cache.window([first], 4)
            cache.window([second], 4)

        self.assertEqual(observed_entries, [0, 0])
        self.assertEqual(
            [entry["path"] for entry in cache.status()["entries"]],
            [str(second.resolve())],
        )

    def test_cpu_stager_reuses_two_buffers_for_many_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, path, _ = cache_fixture(Path(temporary))
            window = cache.window([path], 4)
            view = window.split(0.0).training
            stager = BatchStager(
                batch_size=2,
                channels=window.channels,
                height=window.height,
                width=window.width,
                policy_size=window.policy_size,
                device=torch.device("cpu"),
                state_dtype=window.shards[0].dataset.states.dtype,
            )
            pointers = []
            samples = 0
            try:
                for states, policies, masks, values, _ownership in stager.batches(
                    view.batches(2, shuffle=False)
                ):
                    pointers.append(states.untyped_storage().data_ptr())
                    samples += states.shape[0]
                    self.assertEqual(policies.shape[1], window.policy_size)
                    self.assertEqual(masks.dtype, torch.bool)
                    self.assertEqual(values.ndim, 1)
            finally:
                stager.close()

        self.assertEqual(samples, window.samples)
        self.assertLessEqual(len(set(pointers)), 2)

    def test_batches_pack_across_shard_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(2):
                path = root / str(index) / "dataset.vgo"
                path.parent.mkdir()
                path.write_bytes(f"shard-{index}".encode())
                path.with_name("manifest.json").write_text(
                    json.dumps({"dataset_sha256": f"{index + 1:02x}" * 32}),
                    encoding="utf-8",
                )
                paths.append(path)
            cache = ReplayCache(loader=lambda _: raw_fixture(samples=3))
            view = cache.window(paths, 4).split(0.0).training

            batches = view.batches(4, shuffle=False)

        self.assertEqual([batch.samples for batch in batches], [4, 2])
        self.assertEqual(len(batches[0].parts), 2)


class PersistentLearnerTests(unittest.TestCase):
    def test_two_updates_reuse_runtime_cache_and_stager(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, loads = cache_fixture(root)
            config = LearnerConfig(
                epochs=1,
                batch_size=3,
                learning_rate=1e-3,
                model_width=4,
                blocks=1,
                threads=1,
                device="cpu",
                compile=False,
                report_every=1,
                validation_fraction=0.25,
                augment=False,
            )
            learner = PersistentLearner(
                defaults=config, replay_cache=cache, log=lambda _: None
            )
            first_output = root / "first.pt"
            second_output = root / "second.pt"
            try:
                first = learner.update(
                    LearnerUpdate((path,), first_output, None, config)
                )
                model_identity = id(learner.model)
                optimizer_identity = id(learner.optimizer)
                second = learner.update(
                    LearnerUpdate(
                        (path,), second_output, first_output, config
                    )
                )
                status = learner.status()
                output_exists = second_output.exists()
                report_exists = second_output.with_suffix(".pt.json").exists()
                gradients_released = all(
                    parameter.grad is None
                    for parameter in learner.model.parameters()
                )
            finally:
                learner.close()

        self.assertEqual(len(loads), 1)
        self.assertEqual(id(learner.model), model_identity)
        self.assertEqual(id(learner.optimizer), optimizer_identity)
        self.assertEqual(first["replay_cache_misses"], 1)
        self.assertEqual(second["replay_cache_hits"], 1)
        self.assertTrue(second["stager_reused"])
        self.assertTrue(second["optimizer_restored"])
        self.assertEqual(second["parent_checkpoint"], str(first_output.resolve()))
        self.assertEqual(status["updates"], 2)
        self.assertEqual(
            status["current_checkpoint"], str(second_output.resolve())
        )
        self.assertTrue(output_exists)
        self.assertTrue(report_exists)
        self.assertTrue(gradients_released)

    def test_explicit_different_parent_reloads_authoritative_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, _ = cache_fixture(root)
            config = LearnerConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                model_width=4,
                blocks=1,
                threads=1,
                device="cpu",
                compile=False,
                report_every=1,
                validation_fraction=0.0,
                augment=False,
            )
            learner = PersistentLearner(
                defaults=config, replay_cache=cache, log=lambda _: None
            )
            accepted = root / "accepted.pt"
            candidate = root / "candidate.pt"
            next_output = root / "next.pt"
            try:
                learner.update(LearnerUpdate((path,), accepted, None, config))
                # Produce a different resident candidate, then explicitly point
                # back to the accepted checkpoint as promotion rejection does.
                learner.update(
                    LearnerUpdate((path,), candidate, accepted, config)
                )
                resident = id(learner.model)
                accepted_digest = file_sha256(accepted)
                report = learner.update(
                    LearnerUpdate((path,), next_output, accepted, config)
                )
            finally:
                learner.close()

        self.assertNotEqual(id(learner.model), resident)
        self.assertEqual(report["parent_checkpoint"], str(accepted.resolve()))
        self.assertEqual(report["parent_checkpoint_sha256"], accepted_digest)

    def test_failed_update_discards_partial_runtime_before_implicit_retry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, _ = cache_fixture(root)
            config = LearnerConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                model_width=4,
                blocks=1,
                threads=1,
                device="cpu",
                compile=False,
                report_every=1,
                validation_fraction=0.0,
                augment=False,
            )
            learner = PersistentLearner(
                defaults=config, replay_cache=cache, log=lambda _: None
            )
            original_step = torch.optim.Adam.step

            def fail_after_step(optimizer, closure=None):
                original_step(optimizer, closure)
                raise RuntimeError("injected optimizer failure")

            retry_output = root / "retry.pt"
            clean_output = root / "clean.pt"
            try:
                with patch.object(torch.optim.Adam, "step", fail_after_step):
                    with self.assertRaisesRegex(
                        RuntimeError, "injected optimizer failure"
                    ):
                        learner.update(
                            LearnerUpdate(
                                (path,), root / "failed.pt", None, config
                            )
                        )
                self.assertIsNone(learner.model)
                self.assertIsNone(learner.optimizer)
                self.assertIsNone(learner.current_checkpoint)
                self.assertEqual(learner.status()["updates"], 0)

                retry = learner.update(
                    LearnerUpdate((path,), retry_output, None, config)
                )

                clean_cache = ReplayCache(loader=lambda _: raw_fixture())
                clean = PersistentLearner(
                    defaults=config,
                    replay_cache=clean_cache,
                    log=lambda _: None,
                )
                try:
                    clean.update(
                        LearnerUpdate((path,), clean_output, None, config)
                    )
                finally:
                    clean.close()
                retried_state = torch.load(
                    retry_output, map_location="cpu"
                )["state_dict"]
                clean_state = torch.load(
                    clean_output, map_location="cpu"
                )["state_dict"]
            finally:
                learner.close()

        self.assertIsNone(retry["parent_checkpoint"])
        self.assertEqual(retried_state.keys(), clean_state.keys())
        for name in retried_state:
            torch.testing.assert_close(retried_state[name], clean_state[name])

    def test_same_resident_checkpoint_still_validates_replay_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first" / "dataset.vgo"
            second = root / "second" / "dataset.vgo"
            for index, path in enumerate((first, second)):
                path.parent.mkdir()
                path.write_bytes(f"shard-{index}".encode())
                path.with_name("manifest.json").write_text(
                    json.dumps({"dataset_sha256": f"{index + 1:02x}" * 32}),
                    encoding="utf-8",
                )

            def load(path: str | Path) -> RasterDataset:
                height = 4 if Path(path).parent.name == "first" else 5
                return raw_fixture(height=height, width=height)

            config = LearnerConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                model_width=4,
                blocks=1,
                threads=1,
                device="cpu",
                compile=False,
                report_every=1,
                validation_fraction=0.0,
                augment=False,
            )
            learner = PersistentLearner(
                defaults=config,
                replay_cache=ReplayCache(loader=load),
                log=lambda _: None,
            )
            checkpoint = root / "model.pt"
            try:
                learner.update(
                    LearnerUpdate((first,), checkpoint, None, config)
                )
                with self.assertRaisesRegex(
                    ValueError, "does not match replay tensor shape"
                ):
                    learner.update(
                        LearnerUpdate(
                            (second,),
                            root / "mismatched.pt",
                            checkpoint,
                            config,
                        )
                    )
            finally:
                learner.close()


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_one_json_line_per_command(self) -> None:
        class FakeLearner:
            def __init__(self) -> None:
                self.closed = False

            def update_from_mapping(self, message):
                return {"checkpoint": message["output"]}

            def status(self):
                return {"closed": self.closed}

            def close(self):
                self.closed = True

        requests = StringIO(
            "\n".join(
                (
                    '{"command":"status","request_id":1}',
                    '{"command":"unknown","request_id":2}',
                    '{"command":"shutdown","request_id":3}',
                )
            )
            + "\n"
        )
        responses = StringIO()
        errors = StringIO()
        serve_json_lines(
            FakeLearner(),
            input_stream=requests,
            output_stream=responses,
            error_stream=errors,
        )
        messages = [
            json.loads(line) for line in responses.getvalue().splitlines()
        ]

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["event"], "ready")
        self.assertTrue(messages[1]["ok"])
        self.assertFalse(messages[2]["ok"])
        self.assertIn("unknown learner command", errors.getvalue())
        self.assertTrue(messages[3]["ok"])
        self.assertEqual(messages[3]["result"], {"closed": True})


if __name__ == "__main__":
    unittest.main()
