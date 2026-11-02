import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "training/loop/tail-calibration.py"
spec = importlib.util.spec_from_file_location("tail_calibration", SCRIPT)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def game(values, utility, radius=1 / 18, maximum_plies=100):
    return {
        "radius": radius, "maximum_plies": maximum_plies, "black_utility": utility,
        "plies": len(values), "root_values": values,
    }


def trailing_black(plies=60):
    """Black reads -1 throughout, and it's Black's own turns that count."""
    return [-1.0] * plies


class FirstEntryTests(unittest.TestCase):
    def test_a_streak_counts_own_turns_from_the_first_stage(self):
        # Black's turns at plies 26, 28 and so on: 25 is White's.
        entries = calibration.first_entries(trailing_black(), 100, 0.9, 2)
        self.assertEqual(entries[(0, 0)], 28)
        self.assertEqual(entries[(0, 1)], 50)

    def test_the_first_quarter_never_counts(self):
        self.assertEqual(calibration.first_entries(trailing_black(25), 100, 0.9, 2), {})

    def test_one_good_turn_resets_the_streak(self):
        values = [-1.0] * 60
        for ply in range(26, 60, 4):
            values[ply] = 0.0
        self.assertNotIn((0, 0), calibration.first_entries(values, 100, 0.9, 2))

    def test_seats_are_counted_apart(self):
        entries = calibration.first_entries([1.0] * 60, 100, 0.9, 2)
        self.assertEqual({seat for seat, _ in entries}, {1})


class SharedCaseTests(unittest.TestCase):
    def test_the_replay_matches_the_live_rule(self):
        """The Rust rule reads these same cases, so the two can't drift apart."""
        fixture = json.loads((Path(__file__).parent / "fixtures/tail-rule.json").read_text())
        for case in fixture["cases"]:
            values = []
            for start, end, value in case["spans"]:
                values += [value] * (end - start)
            entries = calibration.first_entries(
                values, fixture["maximum_plies"], fixture["threshold"], fixture["window"]
            )
            first = min(entries.values()) + 1 if entries else None
            self.assertEqual(first, case["first_tail_ply"], case["name"])


class TableTests(unittest.TestCase):
    def test_a_comeback_is_a_win_or_tie_for_the_trailing_seat(self):
        games = [game(trailing_black(), 1.0), game(trailing_black(), 0.0), game(trailing_black(), -1.0)]
        table = calibration.comeback_table(games, 2)
        self.assertEqual(table[(0, 0, 0.99)], [2, 3])

    def test_a_comeback_is_read_from_the_trailing_seat(self):
        # Black leads throughout, so White is the seat that trails, and White won.
        table = calibration.comeback_table([game([1.0] * 60, -1.0)], 2)
        self.assertEqual(table[(0, 0, 0.99)], [1, 1])

    def test_a_game_counts_once_per_cell(self):
        table = calibration.comeback_table([game(trailing_black(), -1.0)], 2)
        self.assertEqual(table[(0, 0, 0.8)], [0, 1])

    def test_boards_land_in_their_bands(self):
        self.assertEqual(calibration.board_band(1 / 18), 0)
        self.assertEqual(calibration.board_band(1 / 26), 1)
        self.assertEqual(calibration.board_band(1 / 38), 2)


class ScheduleTests(unittest.TestCase):
    def test_an_empty_table_leaves_every_cell_closed(self):
        built = calibration.schedule({}, 0.05)
        self.assertTrue(all(t is None for board in built["boards"] for t in board["thresholds"]))

    def test_a_cell_needs_enough_games_even_with_no_comebacks(self):
        self.assertGreater(calibration.upper_bound(0, 72), 0.05)
        self.assertLess(calibration.upper_bound(0, 74), 0.05)

    def test_a_small_clean_sample_stays_closed(self):
        built = calibration.schedule({(0, 0, 0.9): [0, 10]}, 0.05)
        self.assertEqual(built["boards"][0]["thresholds"], [None, None, None])

    def test_the_loosest_passing_value_wins(self):
        table = {
            (0, 0, 0.8): [40, 100],  # Fails.
            (0, 0, 0.9): [0, 200],
            (0, 0, 0.95): [0, 200],
        }
        built = calibration.schedule(table, 0.05)
        self.assertEqual(built["boards"][0]["thresholds"], [0.9, None, None])

    def test_the_schedule_reads_as_the_generator_expects(self):
        built = calibration.schedule({}, 0.05)
        self.assertEqual(built["schema"], "vgo.tail-schedule.v1")
        self.assertIsNone(built["boards"][-1]["up_to_units"])
        for board in built["boards"]:
            self.assertEqual(len(board["thresholds"]), len(built["stages"]))


class RecentGameTests(unittest.TestCase):
    def write(self, root, generation, number, manifest, record):
        path = root / generation / f"game-{number:09d}"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps(manifest))
        (path / "games.jsonl").write_text(json.dumps(record) + "\n")

    def test_naive_games_and_older_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {"behavior_model_sha256": "abc"}
            self.write(root, "gen-0", 1, {}, game([0.0], 1.0))
            self.write(root, "gen-1", 2, model, {"radius": 0.05})
            self.write(root, "gen-1", 3, model, game([0.5], 1.0))
            games = calibration.recent_games(root, 10)
            self.assertEqual([g["root_values"] for g in games], [[0.5]])

    def test_only_the_newest_games_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {"behavior_model_sha256": "abc"}
            for number in (1, 2, 3):
                self.write(root, "gen-1", number, model, game([float(number)], 1.0))
            games = calibration.recent_games(root, 2)
            self.assertEqual([g["root_values"] for g in games], [[2.0], [3.0]])


if __name__ == "__main__":
    unittest.main()
