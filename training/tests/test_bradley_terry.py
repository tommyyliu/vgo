import unittest

from vgo_training.bradley_terry import fit_ratings


def match(a: int, b: int, a_wins: int, b_wins: int, draws: int = 0) -> dict:
    return {"a": a, "b": b, "a_wins": a_wins, "b_wins": b_wins, "draws": draws}


class FitRatingsTests(unittest.TestCase):
    def test_the_anchor_holds_zero_and_a_winner_rates_above_it(self) -> None:
        ratings = fit_ratings([match(1, 0, 30, 10)], anchor=0)
        self.assertEqual(ratings[0], 0)
        self.assertGreater(ratings[1], 0)

    def test_the_rating_gap_matches_the_elo_formula(self) -> None:
        # A 3:1 score is 400 * log10(3), about 191 Elo, and the large sample
        # leaves the prior almost nothing to do.
        ratings = fit_ratings([match(1, 0, 3000, 1000)], anchor=0)
        self.assertAlmostEqual(ratings[1], 190.8, delta=2.0)

    def test_draws_count_as_half_a_win_each(self) -> None:
        # 40-20 with 40 draws is a 60:40 score, not a 40:20 score.
        ratings = fit_ratings([match(1, 0, 400, 200, 400)], anchor=0)
        self.assertAlmostEqual(ratings[1], 70.4, delta=2.0)

    def test_the_prior_is_an_even_game_and_leaves_an_even_record_at_zero(self) -> None:
        ratings = fit_ratings([match(1, 0, 3, 3)], anchor=0, prior_games=8.0)
        self.assertAlmostEqual(ratings[1], 0.0, delta=1.0)

    def test_an_undefeated_model_stays_finite_and_the_prior_sets_how_far(self) -> None:
        light = fit_ratings([match(1, 0, 10, 0)], anchor=0, prior_games=2.0)[1]
        heavy = fit_ratings([match(1, 0, 10, 0)], anchor=0, prior_games=8.0)[1]
        self.assertLess(light, 1000)
        self.assertLess(heavy, light)


if __name__ == "__main__":
    unittest.main()
