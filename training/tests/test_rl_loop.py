import json
from pathlib import Path
import tempfile
import unittest

from vgo_training.rl_loop import (
    json_from_log,
    promotion_decision,
    recover_progress,
    require_artifacts,
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


if __name__ == "__main__":
    unittest.main()
