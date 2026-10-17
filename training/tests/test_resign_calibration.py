import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/resign-calibration.py'
spec = importlib.util.spec_from_file_location('resign_calibration', SCRIPT)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationIdentityTests(unittest.TestCase):
    def test_picker_uses_only_the_requested_minimum_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = root / 'gen-test' / 'game-000000001'
            game.mkdir(parents=True)
            rows = [
                dict(threshold=.70, window=5, fired=40, wrong=0),  # Legacy: minimum zero.
                dict(threshold=.80, window=5, minimum_ply=10, fired=40, wrong=0),
                dict(threshold=.90, window=5, minimum_ply=20, fired=40, wrong=0),
                dict(threshold=.95, window=8, minimum_ply=20, fired=40, wrong=0),
            ]
            (game / 'resign-calibration.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(calibration.pick(root, 400, 5, .03, .99, minimum_ply=20), .90)
            self.assertEqual(calibration.pick(root, 400, 5, .03, .99, minimum_ply=0), .70)
            self.assertEqual(calibration.pick(root, 400, 5, .03, .99, minimum_ply=30), .99)


if __name__ == '__main__':
    unittest.main()
