import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from vgo_training.pipeline import (
    CommandResult,
    ModelArtifact,
    Pipeline,
    PipelineConfig,
    ReplayArtifact,
    _compress_shard,
    atomic_json,
    config_from_arguments,
    file_sha256,
)
from vgo_training.rl_loop import parse_arguments


def replay(sequence: int) -> ReplayArtifact:
    return ReplayArtifact(
        sequence=sequence,
        path=f"/replay/{sequence}/dataset.vgo",
        manifest=f"/replay/{sequence}/manifest.json",
        samples=64,
        behavior_model_sha256=None,
        dataset_sha256=f"digest-{sequence}",
        seed=sequence,
    )


class PipelineConfigurationTests(unittest.TestCase):
    def test_defaults_select_the_utilization_oriented_path(self) -> None:
        arguments = parse_arguments(["--output", "run"])
        self.assertEqual(arguments.architecture, "ddrnet")
        self.assertEqual(arguments.coarse_pool, 4)
        self.assertTrue(arguments.overlap_actor_learner)
        self.assertTrue(arguments.warm_inference)
        self.assertEqual(arguments.inference_slots, 2)
        self.assertEqual(arguments.samples_per_shard, 1024)
        PipelineConfig(output="run").validate()

    def test_invalid_resource_and_gate_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefetch"):
            PipelineConfig(
                output="run", maximum_prefetch_shards=-1
            ).validate()
        with self.assertRaisesRegex(ValueError, "nonzero"):
            PipelineConfig(output="run", promotion_arena=True).validate()
        with self.assertRaisesRegex(ValueError, "requires"):
            PipelineConfig(output="run", promotion_score=0.52).validate()
        with self.assertRaisesRegex(ValueError, "inference slots"):
            PipelineConfig(output="run", inference_slots=0).validate()

    def test_json_normalization_makes_a_run_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = PipelineConfig(
                output=directory, initial_replay=(), updates=2
            )
            first = Pipeline(config)
            second = Pipeline(config)
            self.assertEqual(first.config_digest, second.config_digest)
            self.assertEqual(first.state.to_json(), second.state.to_json())

    def test_telemetry_controls_do_not_change_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = parse_arguments(
                ["--output", directory, "--drain-telemetry"]
            )
            config = config_from_arguments(arguments)
            original = Pipeline(config)
            resumed = Pipeline.resume(Path(directory))
            self.assertEqual(original.config_digest, resumed.config_digest)

    def test_operational_controls_can_change_and_targets_can_extend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Pipeline(
                PipelineConfig(output=directory, updates=2, actors=64)
            )
            tuned = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=5,
                    actors=12,
                    inference_slots=3,
                    maximum_prefetch_shards=2,
                )
            )
            history = json.loads(
                (
                    Path(directory) / "pipeline-config-history.json"
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(original.config_digest, tuned.config_digest)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["config"]["updates"], 5)
            self.assertEqual(history[-1]["config"]["actors"], 12)
            self.assertEqual(history[-1]["config"]["inference_slots"], 3)

    def test_learning_semantics_cannot_change_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Pipeline(PipelineConfig(output=directory))
            with self.assertRaisesRegex(ValueError, "learning configuration"):
                Pipeline(
                    PipelineConfig(
                        output=directory, generation_simulations=257
                    )
                )

    def test_target_cannot_be_reduced_below_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Pipeline(
                PipelineConfig(output=directory, updates=3)
            )
            original.state.updates_completed = 2
            original._save_state()
            with self.assertRaisesRegex(ValueError, "cannot be reduced"):
                Pipeline(PipelineConfig(output=directory, updates=1))


class PipelineCommandTests(unittest.TestCase):
    @patch("vgo_training.pipeline.cargo_executable", return_value="cargo")
    def test_actor_and_arena_commands_share_search_contract(
        self, _cargo: object
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    coarse_pool=8,
                    leaf_batch=4,
                    inference_batch=32,
                    inference_slots=3,
                )
            )
            model = ModelArtifact(
                version=3,
                checkpoint="/models/3.pt",
                onnx="/models/3.onnx",
                checkpoint_sha256="checkpoint",
                onnx_sha256="onnx",
                parent_version=2,
            )
            generation = pipeline.generation_command(
                output=Path("/staging"), sequence=7, model=model
            )
            arena = pipeline.arena_command(
                candidate=Path("/models/4.onnx"),
                opponents=[Path("/models/3.onnx"), Path("/models/1.onnx")],
                seed=9,
                pairs=5,
            )

        for command in (generation, arena):
            self.assertEqual(
                command[command.index("--coarse-pool") + 1], "8"
            )
            self.assertEqual(command[command.index("--leaf-batch") + 1], "4")
            self.assertEqual(
                command[command.index("--maximum-batch") + 1], "32"
            )
        self.assertEqual(generation[generation.index("--runtime") + 1], "onnx")
        self.assertEqual(
            generation[generation.index("--writer-queue-games") + 1], "2"
        )
        self.assertEqual(
            generation[generation.index("--inference-slots") + 1], "3"
        )
        self.assertEqual(arena.count("--opponent"), 2)
        self.assertEqual(arena[arena.index("--pairs") + 1], "5")

    def test_command_result_reads_every_json_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            stdout.write_text(
                'progress\n{"step":1}\nmore\n{"step":2}\n',
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")
            result = CommandResult((), 0, 1.0, stdout, stderr)
            self.assertEqual(
                result.json_documents(), [{"step": 1}, {"step": 2}]
            )
            self.assertEqual(result.final_json(), {"step": 2})

    def test_tensorrt_warmup_uses_the_actor_runtime_contract(self) -> None:
        class Result:
            @staticmethod
            def final_json() -> dict[str, object]:
                return {
                    "provider": "tensorrt",
                    "resolution": 96,
                    "policy_resolution": 32,
                    "batch": 24,
                    "fp16": True,
                }

        class Runner:
            def __init__(self) -> None:
                self.command: list[str] | None = None

            async def run(self, command, **_kwargs):
                self.command = list(command)
                return Result()

        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    inference_batch=24,
                    inference_device_id=2,
                )
            )
            runner = Runner()
            pipeline.runner = runner  # type: ignore[assignment]
            report = asyncio.run(
                pipeline._warm_inference(
                    3, Path(directory) / "updates" / "update-000003"
                )
            )

        assert runner.command is not None
        self.assertEqual(report["batch"], 24)
        self.assertEqual(
            runner.command[runner.command.index("--device-id") + 1], "2"
        )
        self.assertEqual(
            runner.command[
                runner.command.index("--policy-resolution") + 1
            ],
            "32",
        )
        self.assertIn("--compare-python", runner.command)


class PipelineSchedulingTests(unittest.TestCase):
    def test_scheduler_fills_one_update_then_bounds_stale_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=3,
                    shards_per_update=2,
                    maximum_prefetch_shards=2,
                )
            )
            self.assertTrue(
                pipeline._should_start_generation(
                    [], learning_through=None, generation_active=False
                )
            )
            first = [replay(0)]
            self.assertTrue(
                pipeline._should_start_generation(
                    first, learning_through=None, generation_active=False
                )
            )
            ready = first + [replay(1)]
            self.assertFalse(
                pipeline._should_start_generation(
                    ready, learning_through=None, generation_active=False
                )
            )
            self.assertTrue(
                pipeline._should_start_generation(
                    ready, learning_through=1, generation_active=False
                )
            )
            self.assertTrue(
                pipeline._should_start_generation(
                    ready + [replay(2)],
                    learning_through=1,
                    generation_active=False,
                )
            )
            self.assertFalse(
                pipeline._should_start_generation(
                    ready + [replay(2), replay(3)],
                    learning_through=1,
                    generation_active=False,
                )
            )

    def test_remaining_demand_prevents_a_useless_final_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(output=directory, updates=1)
            )
            self.assertFalse(
                pipeline._should_start_generation(
                    [replay(0)],
                    learning_through=0,
                    generation_active=False,
                )
            )

    def test_concurrent_generators_run_at_once_with_distinct_sequences(self) -> None:
        """Two generators must overlap, and must not claim the same shard.

        `state.next_shard` only advances on commit, so a second task reading it
        while the first is still running would take the same number and race for
        the same staging directory. The scheduler assigns sequences instead.
        """

        class FakeLearner:
            async def close(self, *, force: bool = False) -> None:
                del force

        async def exercise(directory: str) -> tuple[int, list[int]]:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=4,
                    maximum_prefetch_shards=4,
                    concurrent_generators=2,
                    telemetry_opponents=0,
                )
            )
            claimed: list[int] = []
            active = 0
            maximum_active = 0

            async def generate(
                _model: ModelArtifact | None, sequence: int | None = None
            ) -> ReplayArtifact:
                nonlocal active, maximum_active
                if sequence is None:
                    sequence = pipeline.state.next_shard
                claimed.append(sequence)
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    await asyncio.sleep(0.01)
                    return replay(sequence)
                finally:
                    active -= 1

            async def learn(update: int, spec: dict, path: Path) -> None:
                del spec, path
                await asyncio.sleep(0.001)
                pipeline.state.updates_completed = update + 1

            pipeline.learner = FakeLearner()
            pipeline._generate_shard = generate
            pipeline._learn_and_publish = learn
            await pipeline.run()
            return maximum_active, claimed

        with tempfile.TemporaryDirectory() as directory:
            maximum_active, claimed = asyncio.run(exercise(directory))
        self.assertGreater(maximum_active, 1, "generators never overlapped")
        self.assertEqual(
            len(claimed), len(set(claimed)), f"duplicate sequences: {claimed}"
        )

    def test_one_generator_is_the_serial_path(self) -> None:
        # The default must behave exactly as before, or every existing run
        # changes shape when this lands.
        pipeline = Pipeline(
            PipelineConfig(output="/tmp/x", updates=3, telemetry_opponents=0)
        )
        self.assertEqual(pipeline.config.concurrent_generators, 1)
        self.assertFalse(
            pipeline._should_start_generation(
                [], learning_through=None, generation_active=True
            )
        )

    def test_async_state_machine_overlaps_both_completion_orders(self) -> None:
        class FakeLearner:
            async def close(self, *, force: bool = False) -> None:
                del force

        async def exercise(
            directory: str, generation_delay: float, learning_delay: float
        ) -> tuple[dict[str, object], list[int], list[int], int]:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=3,
                    maximum_prefetch_shards=1,
                    telemetry_opponents=0,
                )
            )
            generated: list[int] = []
            learned: list[int] = []
            active = 0
            maximum_active = 0

            async def generate(
                _model: ModelArtifact | None, sequence: int | None = None
            ) -> ReplayArtifact:
                nonlocal active, maximum_active
                # The scheduler assigns a sequence when generators may overlap;
                # fall back to the committed one for the serial path.
                if sequence is None:
                    sequence = pipeline.state.next_shard
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    await asyncio.sleep(generation_delay)
                    generated.append(sequence)
                    return replay(sequence)
                finally:
                    active -= 1

            async def learn(
                update: int,
                spec: dict[str, object] | None = None,
                update_path: Path | None = None,
            ) -> ModelArtifact:
                nonlocal active, maximum_active
                del update_path
                assert spec is not None
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    await asyncio.sleep(learning_delay)
                    learned.append(update)
                    parent = pipeline.incumbent
                    model = ModelArtifact(
                        version=update,
                        checkpoint=f"/models/{update}.pt",
                        onnx=f"/models/{update}.onnx",
                        checkpoint_sha256=f"{update + 1:064x}",
                        onnx_sha256=f"{update + 2:064x}",
                        parent_version=None if parent is None else parent.version,
                    )
                    pipeline._commit_publication(
                        {
                            "schema": "vgo.pipeline-publication.v1",
                            "update": update,
                            "through_shard": int(spec["through_shard"]),
                            "accepted": True,
                            "model": asdict(model),
                        }
                    )
                    return model
                finally:
                    active -= 1

            pipeline._generate_shard = generate  # type: ignore[method-assign]
            pipeline._learn_and_publish = learn  # type: ignore[method-assign]
            with patch(
                "vgo_training.pipeline.LearnerService.start",
                new=AsyncMock(return_value=FakeLearner()),
            ):
                report = await pipeline.run()
            return report, generated, learned, maximum_active

        for generation_delay, learning_delay in ((0.01, 0.001), (0.001, 0.01)):
            with self.subTest(
                generation_delay=generation_delay,
                learning_delay=learning_delay,
            ):
                with tempfile.TemporaryDirectory() as directory:
                    report, generated, learned, maximum_active = asyncio.run(
                        exercise(
                            directory, generation_delay, learning_delay
                        )
                    )
                self.assertEqual(report["updates_completed"], 3)
                self.assertEqual(generated, [0, 1, 2])
                self.assertEqual(learned, [0, 1, 2])
                self.assertEqual(maximum_active, 2)


class PipelineRecoveryTests(unittest.TestCase):
    def test_run_lease_excludes_a_second_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Pipeline(PipelineConfig(output=directory))
            second = Pipeline(PipelineConfig(output=directory))
            with first._run_lease():
                with self.assertRaisesRegex(RuntimeError, "another pipeline"):
                    with second._run_lease():
                        self.fail("the second lease must not be acquired")

    def test_publication_recovery_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory, updates=1, telemetry_opponents=0
                )
            )
            checkpoint = Path(directory) / "candidate.pt"
            onnx = Path(directory) / "candidate.onnx"
            checkpoint.write_bytes(b"checkpoint")
            onnx.write_bytes(b"onnx")
            model = ModelArtifact(
                version=0,
                checkpoint=str(checkpoint),
                onnx=str(onnx),
                checkpoint_sha256=file_sha256(checkpoint),
                onnx_sha256=file_sha256(onnx),
                parent_version=None,
            )
            report = {
                "schema": "vgo.pipeline-publication.v1",
                "update": 0,
                "through_shard": 0,
                "accepted": True,
                "model": asdict(model),
            }
            update_path = Path(directory) / "updates" / "update-000000"
            atomic_json(update_path / "publication.json", report)
            recovered = asyncio.run(
                pipeline._learn_and_publish(
                    0,
                    {
                        "schema": "vgo.pipeline-update.v1",
                        "update": 0,
                        "through_shard": 0,
                    },
                    update_path,
                )
            )
            pipeline._commit_publication(report)

            self.assertEqual(recovered, model)
            self.assertEqual(len(pipeline.state.models), 1)
            self.assertEqual(pipeline.state.updates_completed, 1)
            self.assertEqual(pipeline.state.consumed_through_shard, 0)

    def test_preconstructed_pipeline_reloads_state_after_acquiring_lease(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = PipelineConfig(
                output=directory, updates=1, telemetry_opponents=0
            )
            first = Pipeline(config)
            stale = Pipeline(config)
            model = ModelArtifact(
                version=0,
                checkpoint="/models/0.pt",
                onnx="/models/0.onnx",
                checkpoint_sha256="01" * 32,
                onnx_sha256="02" * 32,
                parent_version=None,
            )
            first._commit_publication(
                {
                    "schema": "vgo.pipeline-publication.v1",
                    "update": 0,
                    "through_shard": 0,
                    "accepted": True,
                    "model": asdict(model),
                }
            )
            with patch(
                "vgo_training.pipeline.LearnerService.start",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "a completed run must not start the learner"
                    )
                ),
            ):
                report = asyncio.run(stale.run())

            self.assertEqual(report["updates_completed"], 1)
            self.assertEqual(stale.state.models, [asdict(model)])

    def test_existing_replay_is_rehashed_before_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    samples_per_shard=1,
                    telemetry_opponents=0,
                )
            )
            shard = (
                Path(directory) / "replay" / "shard-000000"
            )
            shard.mkdir(parents=True)
            (shard / "dataset.vgo").write_bytes(b"corrupt")
            atomic_json(
                shard / "manifest.json",
                {
                    "schema": "vgo.replay-shard.v1",
                    "samples": 1,
                    "dataset_sha256": "00" * 32,
                    "dataset_bytes": 7,
                    "behavior_model_sha256": None,
                    "seed": 1,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "checksum"):
                asyncio.run(pipeline._generate_shard(None))

    def test_durable_update_spec_is_reconciled_with_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=1,
                    telemetry_opponents=0,
                )
            )
            pipeline._commit_replay(replay(0))
            spec_path = (
                Path(directory)
                / "updates"
                / "update-000000"
                / "update-spec.json"
            )
            atomic_json(
                spec_path,
                {
                    "schema": "vgo.pipeline-update.v1",
                    "update": 0,
                    "through_shard": 1,
                    "active_replay": [asdict(replay(0))],
                    "parent_model": None,
                },
            )

            with self.assertRaisesRegex(
                RuntimeError, "different replay boundary"
            ):
                pipeline._update_spec(0)

    def test_completed_state_reconstructs_a_missing_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(output=directory, updates=1)
            )
            pipeline.state.updates_completed = 1
            pipeline._save_state()
            report = asyncio.run(pipeline.run())
            self.assertEqual(report["updates_completed"], 1)
            self.assertTrue((Path(directory) / "run.json").exists())

    def test_run_report_exposes_overlap_and_batch_fill_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=1,
                    inference_batch=16,
                )
            )
            manifest = Path(directory) / "manifest.json"
            atomic_json(
                manifest,
                {
                    "generation_metrics": {
                        "wall_seconds": 8.0,
                        "summed_game_seconds": 40.0,
                        "writer_backpressure_seconds": 1.5,
                    },
                    "broker_metrics": {
                        "positions": 24,
                        "batches": 2,
                        "inference_seconds": 3.0,
                    },
                },
            )
            pipeline.state.replay = [
                {
                    **asdict(replay(0)),
                    "manifest": str(manifest),
                }
            ]
            pipeline.state.updates_completed = 1
            pipeline.state.active_wall_seconds = 10.0
            atomic_json(
                Path(directory)
                / "updates"
                / "update-000000"
                / "publication.json",
                {
                    "wall_seconds": 4.0,
                    "training": {
                        "wall_seconds": 3.0,
                        "optimization_seconds": 2.0,
                        "replay_cache_hits": 7,
                        "replay_cache_misses": 1,
                    },
                },
            )

            utilization = pipeline.report()["utilization"]

            self.assertEqual(utilization["pipeline_overlap_factor"], 1.2)
            self.assertEqual(utilization["average_active_games"], 5.0)
            self.assertEqual(utilization["inference_batch_fill"], 0.75)
            self.assertEqual(utilization["replay_cache_hit_ratio"], 0.875)

    def test_ratings_ignore_their_own_materialized_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(output=directory, updates=1)
            )
            telemetry = Path(directory) / "telemetry"
            atomic_json(
                telemetry / "v000001-vs-v000000.json",
                {
                    "schema": "vgo.telemetry-result.v1",
                    "candidate_version": 1,
                    "opponent_version": 0,
                    "arena": {
                        "candidate_wins": 7,
                        "candidate_losses": 1,
                        "draws": 0,
                    },
                },
            )
            atomic_json(telemetry / "ratings.json", {"0": 0.0})
            pipeline._refresh_ratings()
            ratings = json.loads(
                (telemetry / "ratings.json").read_text(encoding="utf-8")
            )
            self.assertIn("1", ratings)

    def test_telemetry_amortizes_one_candidate_load_across_opponents(
        self,
    ) -> None:
        class Result:
            @staticmethod
            def json_documents() -> list[dict[str, object]]:
                return [
                    {
                        "opponent_model": f"/models/{version}.onnx",
                        "games": 4,
                        "completed": 4,
                        "candidate_wins": 3,
                        "candidate_losses": 1,
                        "draws": 0,
                    }
                    for version in (0, 1)
                ]

        class Runner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            async def run(self, command, **_kwargs):
                self.commands.append(list(command))
                return Result()

        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(
                    output=directory,
                    updates=1,
                    telemetry_pairs=2,
                )
            )
            group_seed = 1_000
            pipeline.state.telemetry_pending = [
                {
                    "schema": "vgo.telemetry-job.v1",
                    "id": f"v000002-vs-v{opponent:06d}",
                    "candidate_version": 2,
                    "candidate": "/models/2.onnx",
                    "opponent_version": opponent,
                    "opponent": f"/models/{opponent}.onnx",
                    "pairs": 2,
                    "group_seed": group_seed,
                    "group_index": index,
                    "seed": group_seed + index * 1_000_003,
                }
                for index, opponent in enumerate((0, 1))
            ]
            pipeline._save_state()
            runner = Runner()
            pipeline.runner = runner  # type: ignore[assignment]

            asyncio.run(pipeline.drain_telemetry())

            self.assertEqual(len(runner.commands), 1)
            self.assertEqual(runner.commands[0].count("--opponent"), 2)
            self.assertEqual(pipeline.state.telemetry_pending, [])
            self.assertEqual(len(pipeline.state.telemetry_completed), 2)

    def test_promotion_rejects_excessive_truncation(self) -> None:
        arena = {"games": 100, "completed": 50, "candidate_score": 1.0}
        self.assertFalse(Pipeline._promotion_decision(arena, 0.52, 0.02))
        arena["completed"] = 99
        self.assertTrue(Pipeline._promotion_decision(arena, 0.52, 0.02))


class ShardRetirementTests(unittest.TestCase):
    def _pipeline(self, directory: str, **overrides: object) -> Pipeline:
        return Pipeline(
            PipelineConfig(output=directory, replay_window=3, **overrides)
        )

    def test_compression_round_trips_before_the_original_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard-000000"
            shard.mkdir()
            source = shard / "dataset.vgo"
            payload = bytes(range(256)) * 4096
            source.write_bytes(payload)

            path, before, after = _compress_shard(source)

            self.assertEqual(path, source)
            self.assertEqual(before, len(payload))
            self.assertFalse(source.exists(), "original should be removed")
            archive = source.with_suffix(source.suffix + ".zst")
            self.assertTrue(archive.exists())
            self.assertEqual(after, archive.stat().st_size)
            restored = subprocess.run(
                ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, check=True
            ).stdout
            self.assertEqual(restored, payload, "archive must restore exactly")

    def test_retirement_spares_every_shard_the_window_still_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self._pipeline(directory)
            pipeline.state.replay = [asdict(replay(index)) for index in range(6)]
            retired: list[Path] = []

            def record(path: Path) -> tuple[Path, int, int]:
                retired.append(path)
                return path, 1, 1

            # through_shard 5 with window 3 means the learner read 3, 4, and 5.
            # Without a running loop the retirement runs inline, which is what
            # makes this assertable without driving an event loop.
            with patch("vgo_training.pipeline._compress_shard", side_effect=record):
                with patch.object(Path, "exists", return_value=True):
                    pipeline._retire_aged_shards(5)

            sequences = sorted(int(path.parent.name) for path in retired)
            self.assertEqual(
                sequences,
                [0, 1, 2],
                "only shards below the window may be retired",
            )

    def test_retirement_can_be_switched_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self._pipeline(directory, retire_shards=False)
            pipeline.state.replay = [asdict(replay(index)) for index in range(6)]
            with patch("vgo_training.pipeline._compress_shard") as compress:
                pipeline._retire_aged_shards(5)
            compress.assert_not_called()
            self.assertFalse(pipeline._retirements)


class TelemetrySubsetTests(unittest.TestCase):
    def _model(self, version: int) -> ModelArtifact:
        return ModelArtifact(
            version=version,
            checkpoint=f"/models/{version}/candidate.pt",
            onnx=f"/models/{version}/candidate.onnx",
            checkpoint_sha256=f"checkpoint-{version}",
            onnx_sha256=f"onnx-{version}",
            parent_version=version - 1 if version else None,
            accepted=True,
        )

    def test_every_nth_checkpoint_is_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(
                PipelineConfig(output=directory, telemetry_every=3)
            )
            pipeline.state.models = [asdict(self._model(v)) for v in range(7)]
            for version in range(7):
                pipeline._queue_telemetry(self._model(version))
            rated = sorted(
                {int(job["candidate_version"]) for job in pipeline.state.telemetry_pending}
            )
            # 0 has no prior opponent to play, so the first rated point is 3.
            self.assertEqual(rated, [3, 6])

    def test_default_rates_every_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(PipelineConfig(output=directory))
            pipeline.state.models = [asdict(self._model(v)) for v in range(4)]
            for version in range(4):
                pipeline._queue_telemetry(self._model(version))
            rated = sorted(
                {int(job["candidate_version"]) for job in pipeline.state.telemetry_pending}
            )
            self.assertEqual(rated, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
