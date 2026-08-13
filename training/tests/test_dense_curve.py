import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dense-curve.py"
SPEC = importlib.util.spec_from_file_location("dense_curve", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dense_curve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense_curve)


class DenseCurveBatchTests(unittest.TestCase):
    def test_round_clamps_to_smallest_exported_batch(self) -> None:
        old = Path("old.onnx")
        new = Path("new.onnx")
        group = [("run", 1, old), ("run", 2, new), ("naive", -1, None)]

        maximum = dense_curve.round_maximum_batch(
            group, {old: 32, new: 64}, requested=64
        )

        self.assertEqual(maximum, 32)

    def test_reads_maximum_batch_from_export_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            onnx = Path(directory) / "candidate.onnx"
            manifest = onnx.with_suffix(".onnx.json")
            manifest.write_text(
                json.dumps({"input": {"maximum_batch": 64}}),
                encoding="utf-8",
            )

            self.assertEqual(dense_curve.exported_maximum_batch(onnx), 64)


if __name__ == "__main__":
    unittest.main()
