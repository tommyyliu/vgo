import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "training/loop/komi-calibration.py"
spec = importlib.util.spec_from_file_location("komi_calibration", SCRIPT)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)

START = {"centre_points": 7.0, "points_per_unit": 0.0, "spread_points": 5.46}


def games(count, fair, slope=-0.6, seed=1):
    """Games whose Black win chance is logistic in komi around `fair(size)` points."""
    generator = random.Random(seed)
    out = []
    for _ in range(count):
        size = generator.choice((18.0, 26.0, 38.0))
        points = generator.uniform(1.5, 12.5)
        black = 1.0 / (1.0 + math.exp(-slope * (points - fair(size))))
        radius = 1.0 / size
        out.append({"radius": radius, "komi": points * 4.0 * radius * radius,
                    "black_utility": 1.0 if generator.random() < black else -1.0})
    return out


class KomiCalibrationTests(unittest.TestCase):
    def test_the_fit_finds_the_even_point_on_each_board(self):
        built = calibration.recentre(games(6000, lambda s: 2.0 + 0.1 * s), START,
                                     minimum_games=300, max_step=100.0, minimum_z=2.0)
        self.assertAlmostEqual(built["fair_points"]["18"], 3.8, delta=0.4)
        self.assertAlmostEqual(built["fair_points"]["38"], 5.8, delta=0.4)
        self.assertAlmostEqual(built["centre_points"] + 18 * built["points_per_unit"],
                               built["centre_at"]["18"], places=9)
        self.assertLess(built["slope_per_point"], 0)

    def test_the_centre_moves_at_most_one_step(self):
        built = calibration.recentre(games(6000, lambda s: -4.0), START,
                                     minimum_games=300, max_step=1.0, minimum_z=2.0)
        self.assertEqual(built["centre_at"], {"18": 6.0, "38": 6.0})
        self.assertEqual(built["spread_points"], 5.46)

    def test_no_move_without_a_clear_komi_effect_or_enough_games(self):
        flat = calibration.recentre(games(3000, lambda s: 0.0, slope=0.0), START,
                                    minimum_games=300, max_step=1.0, minimum_z=2.0)
        self.assertEqual(flat["centre_at"], {"18": 7.0, "38": 7.0})
        self.assertIn("does not yet move", flat["reason"])
        few = calibration.recentre(games(100, lambda s: 0.0), START,
                                   minimum_games=300, max_step=1.0, minimum_z=2.0)
        self.assertEqual(few["reason"], "too few games")
        self.assertEqual(few["centre_at"], {"18": 7.0, "38": 7.0})

    def test_the_output_is_a_schedule_the_generator_reads(self):
        built = calibration.recentre(games(2000, lambda s: 5.0), START,
                                     minimum_games=300, max_step=1.0, minimum_z=2.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "komi.json"
            path.write_text(json.dumps(built))
            read = json.loads(path.read_text())
        for key in ("centre_points", "points_per_unit", "spread_points"):
            self.assertIsInstance(read[key], float)


if __name__ == "__main__":
    unittest.main()
