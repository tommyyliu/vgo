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
        self.assertEqual(default.architecture, "flat")
        validate_arguments(default)

        invalid = parse_arguments(["--output", "run", "--coarse-pool", "-1"])
        with self.assertRaisesRegex(ValueError, "coarse pool must be nonnegative"):
            validate_arguments(invalid)

        # The pool counts fine cells per coarse region on the placement grid,
        # which is independent of the render resolution.
        oversized_pool = parse_arguments(
            ["--output", "run", "--policy-resolution", "16", "--coarse-pool", "17"]
        )
        with self.assertRaisesRegex(ValueError, "coarse pool must not exceed policy resolution"):
            validate_arguments(oversized_pool)

        decoupled = parse_arguments(
            [
                "--output",
                "run",
                "--resolution",
                "16",
                "--policy-resolution",
                "32",
                "--coarse-pool",
                "17",
            ]
        )
        validate_arguments(decoupled)

        ddrnet = parse_arguments(
            ["--output", "run", "--architecture", "ddrnet"]
        )
        self.assertEqual(ddrnet.architecture, "ddrnet")
        validate_arguments(ddrnet)

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


class EloPoolTests(unittest.TestCase):
    """The Elo pool replaces the vs-incumbent arena once gating is off.

    A promotion arena only ever answers "did N beat N-1", which is the least
    informative comparison available: consecutive generations are nearly
    identical, so a 60-pair result is mostly sampling noise. Bradley-Terry over
    an accumulating match history pools every game ever played into every
    rating, for a fraction of the games per iteration.
    """

    def test_ratings_recover_a_known_ladder(self) -> None:
        import random

        from vgo_training.bradley_terry import fit_ratings

        rng = random.Random(7)
        truth = {generation: generation * 15.0 for generation in range(30)}
        matches = []
        for generation in range(1, 30):
            for opponent in rng.sample(range(generation), min(4, generation)):
                wins = losses = 0
                for _ in range(8):
                    probability = 1.0 / (
                        1.0 + 10.0 ** ((truth[opponent] - truth[generation]) / 400.0)
                    )
                    if rng.random() < probability:
                        wins += 1
                    else:
                        losses += 1
                matches.append(
                    {"a": generation, "b": opponent, "a_wins": wins, "b_wins": losses}
                )
        ratings = fit_ratings(matches, anchor=0, prior_games=0.25)
        self.assertAlmostEqual(ratings[0], 0.0)
        # Monotone ladder: the last generation must rate well above the first.
        self.assertGreater(ratings[29], 300.0)
        errors = [abs(ratings[g] - truth[g]) for g in range(30)]
        self.assertLess(sum(errors) / len(errors), 120.0)

    def test_heavy_prior_shrinks_ratings_toward_the_anchor(self) -> None:
        from vgo_training.bradley_terry import fit_ratings

        matches = [{"a": 1, "b": 0, "a_wins": 16, "b_wins": 4}]
        light = fit_ratings(matches, anchor=0, prior_games=0.25)
        heavy = fit_ratings(matches, anchor=0, prior_games=2.0)
        self.assertGreater(light[1], heavy[1])

    def test_promotion_gate_and_arena_must_agree(self) -> None:
        # Default: no arena, no gate, every candidate accepted.
        default = parse_arguments(["--output", "out"])
        self.assertFalse(default.promotion_arena)
        self.assertEqual(default.promotion_score, 0.0)
        validate_arguments(default)

        # A score without the arena that measures it is a silent no-op.
        orphan = parse_arguments(["--output", "out", "--promotion-score", "0.52"])
        with self.assertRaisesRegex(ValueError, "no effect without"):
            validate_arguments(orphan)

        # An arena with a zero gate would run 120 games and accept regardless.
        toothless = parse_arguments(["--output", "out", "--promotion-arena"])
        with self.assertRaisesRegex(ValueError, "nonzero"):
            validate_arguments(toothless)

        gated = parse_arguments(
            ["--output", "out", "--promotion-arena", "--promotion-score", "0.52"]
        )
        validate_arguments(gated)


class BatchedArenaTests(unittest.TestCase):
    def test_arena_command_repeats_the_opponent_flag(self) -> None:
        arguments = parse_arguments(["--output", "out"])
        single = arena_command(
            arguments, Path("/root"), Path("cand.onnx"), Path("a.onnx"), 1
        )
        self.assertEqual(single.count("--opponent"), 1)
        several = arena_command(
            arguments,
            Path("/root"),
            Path("cand.onnx"),
            [Path("a.onnx"), Path("b.onnx"), Path("c.onnx")],
            1,
        )
        self.assertEqual(several.count("--opponent"), 3)
        self.assertIn("b.onnx", several)

    def test_every_record_is_read_from_a_batched_log(self) -> None:
        from vgo_training.rl_loop import json_documents_from_log

        body = "\n".join(
            json.dumps({"candidate_score": score, "opponent_model": name})
            for score, name in ((0.5, "a"), (0.75, "b"), (0.25, "c"))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.log"
            path.write_text(f"COMMAND x\nSTDOUT\n{body}\nSTDERR\n", encoding="utf-8")
            documents = json_documents_from_log(path)
        self.assertEqual([d["opponent_model"] for d in documents], ["a", "b", "c"])
        # The single-document reader keeps only the last, which is why the
        # batched path needs its own reader.
        self.assertEqual(len(documents), 3)


class LeafBatchTests(unittest.TestCase):
    def test_leaf_batch_reaches_generation_and_arenas(self) -> None:
        """Both seats of a comparison must search the same way.

        Above 1, leaf parallelization changes which nodes get explored, so a
        generation run and the arenas judging it have to agree. The flag existed
        in both Rust binaries but rl_loop forwarded it to neither, pinning every
        run to the sequential path.
        """
        arguments = parse_arguments(["--output", "run", "--leaf-batch", "8"])
        root = Path("/repo")
        generation = generation_command(
            arguments, root, Path("/replay"), 1, Path("/model.onnx")
        )
        self.assertEqual(generation[generation.index("--leaf-batch") + 1], "8")
        arena = arena_command(
            arguments, root, Path("/candidate.onnx"), Path("/opponent.onnx"), 2
        )
        self.assertEqual(arena[arena.index("--leaf-batch") + 1], "8")

    def test_leaf_batch_defaults_to_the_sequential_path(self) -> None:
        self.assertEqual(parse_arguments(["--output", "run"]).leaf_batch, 1)
