from __future__ import annotations

from dataclasses import replace
from io import StringIO
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vgo_training.dataset import PreparedRasterDataset, RasterDataset
from vgo_training.dataset import file_sha256
from vgo_training.learner import (
    BatchStager,
    LearnerConfig,
    LearnerUpdate,
    PersistentLearner,
    ReplayCache,
    evaluate,
    masked_value_cross_entropy,
    thin_value_labels,
    value_cross_entropy,
)


def raw_fixture(
    samples: int = 12,
    *,
    # 32, not smaller: DDRNet's context branch is 1/16 of the raster, and its
    # training-only BatchNorm needs more than one value per channel when the
    # last batch of an epoch holds a single sample.
    height: int = 32,
    width: int = 32,
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


def scanned_batches(view, batch_size, *, shuffle, generator=None, augment=False):
    """`ReplayView.batches` as it was: every shard scanned for every batch.

    Kept as the reference the grouped version must reproduce exactly. It costs
    batches x shards tensor ops, which is why it was replaced.
    """
    from vgo_training.learner import DIHEDRAL_TRANSFORMS, BatchPart, BatchSpec

    shard_ids = torch.cat([
        torch.full((selection.rows.numel(),), index, dtype=torch.int64)
        for index, selection in enumerate(view.selections)
    ])
    rows = torch.cat([selection.rows for selection in view.selections])
    if shuffle and rows.numel() > 1:
        order = torch.randperm(rows.numel(), generator=generator)
        shard_ids, rows = shard_ids[order], rows[order]
    batches = []
    for start in range(0, rows.numel(), batch_size):
        batch_shards, batch_rows = shard_ids[start:start + batch_size], rows[start:start + batch_size]
        parts = tuple(
            BatchPart(selection.shard, batch_rows[batch_shards == index])
            for index, selection in enumerate(view.selections)
            if bool((batch_shards == index).any())
        )
        transform = (
            int(torch.randint(len(DIHEDRAL_TRANSFORMS), (1,), generator=generator).item())
            if augment else 0
        )
        batches.append(BatchSpec(parts, transform))
    return batches


class ReplayViewBatchTests(unittest.TestCase):
    def test_batches_match_the_per_shard_scan(self) -> None:
        from vgo_training.learner import ReplayView, ShardSelection

        sizes = torch.Generator().manual_seed(7)
        for case in range(40):
            shards = int(torch.randint(1, 30, (1,), generator=sizes).item())
            selections = tuple(
                # Distinct objects stand in for shards: batches() only carries them.
                ShardSelection(object(), torch.randperm(50, generator=sizes)[
                    : int(torch.randint(0, 12, (1,), generator=sizes).item())
                ])
                for _ in range(shards)
            )
            view = ReplayView(selections)
            for batch_size in (1, 5, 64):
                for shuffle, augment in ((False, False), (True, False), (True, True)):
                    seed = 1000 * case + batch_size
                    expected = scanned_batches(
                        view, batch_size, shuffle=shuffle, augment=augment,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    actual = view.batches(
                        batch_size, shuffle=shuffle, augment=augment,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    self.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        self.assertEqual(got.transform, want.transform)
                        self.assertEqual(len(got.parts), len(want.parts))
                        for a, b in zip(got.parts, want.parts):
                            self.assertIs(a.shard, b.shard)
                            self.assertTrue(torch.equal(a.rows, b.rows))


class ReplayCacheTests(unittest.TestCase):
    def test_spilled_shard_loads_identical_in_a_new_process_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, path, loads = cache_fixture(root)
            spill = root / "spill"

            def load(_: str | Path) -> RasterDataset:
                loads.append(1)
                return raw_fixture()

            first = ReplayCache(loader=load, spill_directory=spill).window([path], 4)

            def refuse(_: str | Path) -> RasterDataset:
                raise AssertionError("a spilled shard must not be rendered again")

            second_cache = ReplayCache(loader=refuse, spill_directory=spill)
            second = second_cache.window([path], 4)
            self.assertEqual(len(loads), 1)
            self.assertEqual(second_cache.spill_hits, 1)
            a, b = first.shards[0], second.shards[0]
            self.assertEqual(a.digest, b.digest)
            self.assertTrue(torch.equal(a.split_hashes, b.split_hashes))
            for field in ("values", "policies", "policy_masks"):
                self.assertTrue(torch.equal(getattr(a.dataset, field), getattr(b.dataset, field)))

            # A new manifest digest is new content: render it, do not reuse.
            path.with_name("manifest.json").write_text(
                json.dumps({"dataset_sha256": "cd" * 32}), encoding="utf-8"
            )
            ReplayCache(loader=load, spill_directory=spill).window([path], 4)
            self.assertEqual(len(loads), 2)

            # Only the second digest's file was asked for, so the first is pruned.
            pruner = ReplayCache(loader=refuse, spill_directory=spill)
            pruner.window([path], 4)
            self.assertEqual(pruner.prune_spill(), 1)
            self.assertEqual(len(list(spill.glob("*.pt"))), 1)

    def test_mapped_spill_holds_shards_as_file_maps(self) -> None:
        # Both a shard this process renders and one it finds spilled must be
        # backed by the spill file, or a window larger than RAM still loads
        # into RAM: the rendered ones are most of a new window.
        # Storages loaded with mmap=True report no filename, so ask the kernel:
        # a mapped tensor's memory lies in a range /proc/self/maps names a
        # spill file for.
        def mapped(tensor: torch.Tensor) -> bool:
            address = tensor.data_ptr()
            for line in Path("/proc/self/maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) < 6 or not fields[5].startswith(str(spill)):
                    continue
                low, high = (int(bound, 16) for bound in fields[0].split("-"))
                if low <= address < high:
                    return True
            return False

        def file_backed(dataset) -> bool:
            return all(mapped(getattr(dataset, field)) for field in ("states", "values"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, path, loads = cache_fixture(root)
            spill = root / "spill"

            def load(_: str | Path) -> RasterDataset:
                loads.append(1)
                return raw_fixture()

            plain = ReplayCache(loader=load).window([path], 4).shards[0]
            rendered = ReplayCache(
                loader=load, spill_directory=spill, map_spill=True
            ).window([path], 4).shards[0]
            reloaded = ReplayCache(
                loader=load, spill_directory=spill, map_spill=True
            ).window([path], 4).shards[0]
            self.assertEqual(len(loads), 2)  # the plain cache and the first mapped one
            for shard in (rendered, reloaded):
                self.assertTrue(file_backed(shard.dataset))
                self.assertTrue(mapped(shard.split_hashes))
                self.assertTrue(torch.equal(shard.split_hashes, plain.split_hashes))
                for field in ("values", "policies", "policy_masks"):
                    self.assertTrue(
                        torch.equal(getattr(shard.dataset, field), getattr(plain.dataset, field))
                    )
            self.assertFalse(file_backed(plain.dataset))

        with self.assertRaises(ValueError):
            ReplayCache(map_spill=True)

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

    def test_value_thinning_keeps_one_label_per_stride_of_each_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, path, _ = cache_fixture(Path(temporary))
            window = cache.window([path], 4)
            original = window.shards[0].dataset.values.clone()
            view = window.split(0.0).training
            thinned, share = thin_value_labels(view, 2, seed=5)
            again, _ = thin_value_labels(view, 2, seed=5)
            values = thinned.selections[0].shard.dataset.values

        # Two plies per game: exactly one keeps its label, and it is the same one
        # on a second call with the same seed.
        for first in range(0, 12, 2):
            kept = (~torch.isnan(values[first : first + 2])).sum().item()
            self.assertEqual(kept, 1, f"game at rows {first}-{first + 1}")
        self.assertTrue(
            torch.equal(
                torch.isnan(values),
                torch.isnan(again.selections[0].shard.dataset.values),
            )
        )
        known = ~torch.isnan(values)
        self.assertTrue(torch.equal(values[known], original[known]))
        # The cached shard, which validation reads, keeps every label.
        self.assertTrue(torch.equal(window.shards[0].dataset.values, original))
        self.assertEqual(share, 0.5)
        self.assertEqual(thin_value_labels(view, 1, seed=5), (view, 1.0))

    def test_labels_per_game_spreads_a_fixed_count_over_each_game(self) -> None:
        # Games of 40, 17 and 3 plies, interleaved as rows of one shard.
        games = torch.tensor([0] * 40 + [1] * 17 + [2] * 3, dtype=torch.int64)
        games = games[torch.randperm(60, generator=torch.Generator().manual_seed(2))]

        def load(_: str | Path) -> RasterDataset:
            return replace(raw_fixture(60), game_ids=games)

        with tempfile.TemporaryDirectory() as temporary:
            _, path, _ = cache_fixture(Path(temporary))
            cache = ReplayCache(loader=load)
            window = cache.window([path], 4)
            view = window.split(0.0).training
            for seed in (1, 2, 3):
                thinned, share = thin_value_labels(view, 1, seed=seed, per_game=4)
                values = thinned.selections[0].shard.dataset.values
                kept = ~torch.isnan(values)
                counts = {g: int(kept[games == g].sum()) for g in (0, 1, 2)}
                # Four per game, every row of the 3-ply game.
                self.assertEqual(counts, {0: 4, 1: 4, 2: 3}, f"seed {seed}")
                self.assertAlmostEqual(share, 11 / 60)
                # Spread: in the 40-ply game the kept ranks are 10 apart.
                ranks = kept[games == 0].nonzero().flatten()
                self.assertEqual(set(torch.diff(ranks).tolist()), {10})

    def test_masked_value_loss_drops_rather_than_reweights(self) -> None:
        logits = torch.randn(6, 2, generator=torch.Generator().manual_seed(4))
        targets = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(
                masked_value_cross_entropy(logits, targets),
                value_cross_entropy(logits, targets),
            )
        )
        self.assertTrue(
            torch.allclose(
                masked_value_cross_entropy(logits, targets, scale=3.0),
                3.0 * value_cross_entropy(logits, targets),
            )
        )
        thinned = targets.clone()
        thinned[[1, 3, 4]] = float("nan")
        kept = value_cross_entropy(logits[[0, 2, 5]], targets[[0, 2, 5]])
        loss = masked_value_cross_entropy(logits, thinned)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss, kept * 3 / 6))

    def test_evaluate_reports_proper_value_scores(self) -> None:
        predicted = torch.linspace(-0.9, 0.3, 12)

        class Fixed(torch.nn.Module):
            """Uniform policy, and a value that rises with the row, as the targets do."""

            def forward(self, states):
                return torch.zeros(states.shape[0], 32 * 32 + 1), predicted[: states.shape[0]]

        with tempfile.TemporaryDirectory() as temporary:
            cache, path, _ = cache_fixture(Path(temporary))
            window = cache.window([path], 12)
            view = window.split(0.0).training
            stager = BatchStager(batch_size=12, channels=window.channels, height=32, width=32,
                                 policy_size=32 * 32 + 1, device=torch.device("cpu"),
                                 state_dtype=window.shards[0].dataset.states.dtype)
            result = evaluate(Fixed(), view, stager, batch_size=12, value_weight=1.0,
                              precision="float32")
            stager.close()
        values = torch.linspace(-1.0, 1.0, 12)  # the fixture's targets, in row order
        self.assertAlmostEqual(result["value_mse"], float(((predicted - values) ** 2).mean()), places=5)
        self.assertAlmostEqual(result["value_mae"], float((predicted - values).abs().mean()), places=5)
        win, outcome = (predicted + 1) / 2, (values + 1) / 2
        log_loss = -(outcome * win.log() + (1 - outcome) * (1 - win).log())
        self.assertAlmostEqual(result["value_log_loss"], float(log_loss.mean()), places=5)

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
            optimizer_class = torch.optim.Adam
            original_step = optimizer_class.step

            def fail_after_step(optimizer, closure=None):
                original_step(optimizer, closure)
                raise RuntimeError("injected optimizer failure")

            retry_output = root / "retry.pt"
            clean_output = root / "clean.pt"
            try:
                with patch.object(optimizer_class, "step", fail_after_step):
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


class OwnershipWeightTests(unittest.TestCase):
    """Zero ownership weight must skip both head sets, not just the first.

    The normed head's ownership term read a `present` mask that only the plain
    head's branch assigned, so disabling ownership raised UnboundLocalError one
    update into a live run rather than at import or in a unit test.
    """

    def test_zero_weight_skips_both_ownership_terms(self) -> None:
        from vgo_training.learner import LearnerConfig

        source = inspect.getsource(PersistentLearner._update_once)
        guarded = [
            line
            for line in source.splitlines()
            if "ownership_loss(" in line
        ]
        self.assertEqual(len(guarded), 2, "expected a plain and a normed term")
        # Both must sit under the same flag; a branch that computes `present`
        # for one and not the other is the bug this covers.
        self.assertEqual(source.count("train_ownership"), 4)
        self.assertEqual(LearnerConfig().ownership_weight, 1.5)
        LearnerConfig(ownership_weight=0.0).validate()
        with self.assertRaises(ValueError):
            LearnerConfig(ownership_weight=-1.0).validate()

    def test_dropping_ownership_releases_the_targets(self) -> None:
        from vgo_training.learner import _drop_ownership

        dataset = PreparedRasterDataset(
            states=torch.zeros((2, 5, 4, 4), dtype=torch.float16),
            policies=torch.zeros((2, 17)),
            policy_masks=torch.ones((2, 17)),
            values=torch.zeros((2,)),
            height=4,
            width=4,
            sources=("test",),
            ownerships=torch.zeros((2, 16)),
        )
        self.assertIsNone(_drop_ownership(dataset).ownerships)
        # Idempotent: a shard already without targets is returned unchanged.
        self.assertIs(
            _drop_ownership(_drop_ownership(dataset)).ownerships, None
        )


if __name__ == "__main__":
    unittest.main()
