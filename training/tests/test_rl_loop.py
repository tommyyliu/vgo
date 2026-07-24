import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vgo_training.rl_loop import (
    arena_command,
    generation_command,
    json_from_log,
    parse_arguments,
    promotion_decision,
    recover_progress,
    require_artifacts,
    validate_arguments,
)


class RlLoopTests(unittest.TestCase):
    def test_json_from_log_uses_final_json_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.log"
            path.write_text(
                "$ command\n\nSTDOUT\nepoch=1 loss=1.0\n{\n  \"schema\": \"test\"\n}\n\nSTDERR\nwarning\n",
                encoding="utf-8",
            )
            self.assertEqual(json_from_log(path), {"schema": "test"})

    def test_recovery_ignores_an_incomplete_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generate.log").write_text(
                "$ command\n\nSTDOUT\npartial\n\nSTDERR\n",
                encoding="utf-8",
            )
            (root / "export.log").write_text(
                "$ command\n\nSTDOUT\n{\"schema\": \"export\"}\n\nSTDERR\n",
                encoding="utf-8",
            )
            self.assertEqual(
                recover_progress(root), {"export": {"schema": "export"}}
            )
            self.assertEqual(
                json.loads((root / "progress.json").read_text(encoding="ascii")),
                {"export": {"schema": "export"}},
            )

    def test_progress_requires_every_declared_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            present = Path(directory) / "present"
            present.touch()
            missing = Path(directory) / "missing"
            progress: dict[str, object] = {"training": {"schema": "test"}}
            self.assertFalse(
                require_artifacts(progress, "training", present, missing)
            )
            self.assertNotIn("training", progress)

    def test_promotion_rejects_excessive_truncation(self) -> None:
        arena: dict[str, object] = {
            "games": 100,
            "completed": 50,
            "candidate_score": 1.0,
        }
        self.assertFalse(promotion_decision(arena, 0.52, 0.02))
        arena["completed"] = 99
        self.assertTrue(promotion_decision(arena, 0.52, 0.02))

    def test_coarse_sampling_defaults_and_validation(self) -> None:
        default = parse_arguments(["--output", "run"])
        self.assertEqual(default.coarse_pool, 0)
        validate_arguments(default)

        invalid = parse_arguments(["--output", "run", "--coarse-pool", "-1"])
        with self.assertRaisesRegex(ValueError, "coarse pool must be nonnegative"):
            validate_arguments(invalid)

        oversized_pool = parse_arguments(
            ["--output", "run", "--resolution", "16", "--coarse-pool", "17"]
        )
        with self.assertRaisesRegex(ValueError, "coarse pool must not exceed resolution"):
            validate_arguments(oversized_pool)

    @patch("vgo_training.rl_loop.cargo_executable", return_value="cargo")
    def test_coarse_sampling_is_forwarded_to_generation_and_all_arenas(
        self, _cargo: object
    ) -> None:
        arguments = parse_arguments(
            [
                "--output",
                "run",
                "--coarse-pool",
                "8",
                "--elo-pool-pairs",
                "3",
            ]
        )
        root = Path("/repo")

        generation = generation_command(
            arguments,
            root,
            Path("/replay"),
            123,
            Path("/incumbent.onnx"),
        )
        self.assertEqual(generation[generation.index("--coarse-pool") + 1], "8")

        baseline = arena_command(
            arguments,
            root,
            Path("/candidate.onnx"),
            None,
            234,
        )
        self.assertEqual(baseline[baseline.index("--coarse-pool") + 1], "8")
        self.assertNotIn("--opponent", baseline)

        promotion = arena_command(
            arguments,
            root,
            Path("/candidate.onnx"),
            Path("/incumbent.onnx"),
            456,
        )
        self.assertEqual(promotion[promotion.index("--coarse-pool") + 1], "8")
        self.assertEqual(promotion[promotion.index("--pairs") + 1], "16")

        elo = arena_command(
            arguments,
            root,
            Path("/candidate.onnx"),
            Path("/past.onnx"),
            789,
            pairs=arguments.elo_pool_pairs,
        )
        self.assertEqual(elo[elo.index("--coarse-pool") + 1], "8")
        self.assertEqual(elo[elo.index("--pairs") + 1], "3")


if __name__ == "__main__":
    unittest.main()
