import unittest

from arena import ArenaStats, play_game, run_arena
from agents import RandomAgent


class ArenaTests(unittest.TestCase):
    def test_play_game_returns_result_tuple(self):
        result, plies, maxed = play_game(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            max_plies=20,
        )

        self.assertIsInstance(result, int)
        self.assertGreaterEqual(plies, 0)
        self.assertIsInstance(maxed, bool)

    def test_arena_stats_records_maxed_game(self):
        stats = ArenaStats()

        stats.record(game_result=0, plies=20, maxed=True)

        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.max_plies_reached, 1)
        self.assertEqual(stats.avg_plies, 20.0)

    def test_run_arena_smoke(self):
        stats = run_arena(
            black_name="random",
            white_name="greedy",
            games=2,
            max_plies=50,
            log_interval=2,
            seed=1,
            material_config_path=None,
            alphabeta_config_path=None,
        )

        self.assertEqual(stats.games, 2)


if __name__ == "__main__":
    unittest.main()
