import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vgo_training.dataset import _select_rows, kept_rows
from vgo_training.window import finished_games, kept_samples, window_from_games


def write_game(root, generation, number, samples, tail_samples=0):
    game = root / generation / f"game-{number:09d}"
    game.mkdir(parents=True)
    manifest = {"samples": samples, "tail_samples": tail_samples}
    (game / "manifest.json").write_text(json.dumps(manifest))
    (game / "dataset.vgo").write_bytes(b"")
    return game


class WindowTests(unittest.TestCase):
    def test_tail_samples_count_at_the_rate_they_are_kept(self):
        manifest = {"samples": 100, "tail_samples": 40}
        self.assertEqual(kept_samples(manifest, 1.0), 100)
        self.assertEqual(kept_samples(manifest, 0.25), 70)

    def test_older_manifests_have_no_tail(self):
        self.assertEqual(kept_samples({"samples": 100}, 0.25), 100)

    def test_a_thinned_window_reaches_further_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (1, 2, 3):
                write_game(root, "gen", number, 100, 80)
            self.assertEqual(len(window_from_games(root, 200, 1.0)[0]), 2)
            self.assertEqual(len(window_from_games(root, 200, 0.25)[0]), 3)

    def test_recency_is_the_game_number_and_not_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_game(root, "gen-1", 5, 10)
            write_game(root, "gen-9", 2, 10)
            chosen, total = window_from_games(root, 10)
            self.assertEqual(chosen[0].parent.name, "game-000000005")
            self.assertEqual(total, 10)

    def test_a_game_still_being_written_does_not_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_game(root, "gen", 1, 10)
            (root / "gen" / "game-000000002.staging").mkdir()
            self.assertEqual(len(list(finished_games(root))), 1)


def shard(directory, spans, plies=100):
    (directory / "games.jsonl").write_text(json.dumps({
        "game": 7, "first_sample": 0, "sample_count": plies, "plies": plies, "tail_spans": spans,
    }) + "\n")
    return directory / "dataset.vgo"


class ThinningTests(unittest.TestCase):
    def test_every_play_ply_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[60, 100]])
            keep = kept_rows(path, np.arange(100), 0.25)
            self.assertTrue(keep[:60].all())

    def test_tail_plies_thin_to_about_the_keep_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[0, 4000]], plies=4000)
            kept = kept_rows(path, np.arange(4000), 0.25).mean()
            self.assertAlmostEqual(kept, 0.25, delta=0.03)

    def test_thinning_repeats_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[50, 100]])
            first = kept_rows(path, np.arange(100), 0.5)
            np.testing.assert_array_equal(first, kept_rows(path, np.arange(100), 0.5))

    def test_keeping_everything_touches_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[0, 100]])
            self.assertTrue(kept_rows(path, np.arange(100), 1.0).all())

    def test_a_game_with_several_spans_thins_all_of_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[10, 20], [50, 100]])
            keep = kept_rows(path, np.arange(100), 0.0001)
            self.assertEqual(int(keep[10:20].sum()) + int(keep[50:].sum()), 0)
            self.assertTrue(keep[20:50].all())

    def test_rows_follow_the_ply_when_fast_turns_left_gaps(self):
        # With fast turns skipped, row 2 is ply 4, and spans are in plies.
        with tempfile.TemporaryDirectory() as directory:
            path = shard(Path(directory), [[4, 5]], plies=6)
            keep = kept_rows(path, np.array([0, 2, 4, 5]), 0.0001)
            self.assertEqual(keep.tolist(), [True, True, False, True])

    def test_rows_are_selected_through_nested_records(self):
        from dataclasses import dataclass

        @dataclass
        class Inner:
            rows: torch.Tensor

        @dataclass
        class Outer:
            values: torch.Tensor
            inner: Inner
            empty_view: torch.Tensor
            size: int

        outer = Outer(torch.arange(4), Inner(torch.arange(8).reshape(4, 2)), torch.zeros(0), 16)
        kept = _select_rows(outer, torch.tensor([True, False, True, False]), 4)
        self.assertEqual(kept.values.tolist(), [0, 2])
        self.assertEqual(kept.inner.rows.tolist(), [[0, 1], [4, 5]])
        self.assertEqual(kept.empty_view.shape, (0,))
        self.assertEqual(kept.size, 16)


if __name__ == "__main__":
    unittest.main()
