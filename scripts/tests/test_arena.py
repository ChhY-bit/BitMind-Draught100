import unittest
import random
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from arena import ArenaStats, apply_random_opening, load_arena_config, play_game, run_arena
from draughts.agents import RandomAgent
from draughts.env import StandardDraughtsEnv
from draughts.model import CandidateMoveNet, ModelConfig


def create_zero_supervised_checkpoint(directory: Path) -> Path:
    """Create a tiny zero-initialised checkpoint for integration tests."""
    config = ModelConfig(
        conv_channels=4,
        conv_layers=1,
        board_feature_dim=8,
        move_embedding_dim=6,
        hidden_dim=5,
    )
    model = CandidateMoveNet(config=config)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    path = directory / "supervised.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "board_channels": config.board_channels,
                "board_size": config.board_size,
                "conv_channels": config.conv_channels,
                "conv_layers": config.conv_layers,
                "conv_kernel_size": config.conv_kernel_size,
                "conv_padding": config.conv_padding,
                "board_feature_dim": config.board_feature_dim,
                "move_feature_dim": config.move_feature_dim,
                "move_embedding_dim": config.move_embedding_dim,
                "hidden_dim": config.hidden_dim,
            },
        },
        path,
    )
    return path


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
            white_name="alphabeta",
            games=2,
            max_plies=50,
            log_interval=2,
            seed=1,
            alphabeta_config_path=None,
            alphabeta_depth=1,
        )

        self.assertEqual(stats.games, 2)

    def test_random_opening_is_reproducible_from_seed(self):
        first = StandardDraughtsEnv()
        second = StandardDraughtsEnv()
        third = StandardDraughtsEnv()
        first.reset()
        second.reset()
        third.reset()

        first_plies, first_finished = apply_random_opening(first, 8, 40, random.Random(7))
        second_plies, second_finished = apply_random_opening(second, 8, 40, random.Random(7))
        apply_random_opening(third, 8, 40, random.Random(8))

        self.assertEqual(first_plies, 8)
        self.assertFalse(first_finished)
        self.assertEqual(second_plies, 8)
        self.assertFalse(second_finished)
        self.assertEqual(first.raw_observation()["squares"], second.raw_observation()["squares"])
        self.assertNotEqual(first.raw_observation()["squares"], third.raw_observation()["squares"])

    def test_arena_config_loads_random_opening(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "arena.yaml"
            path.write_text(
                "black: random\nwhite: alphabeta\ngames: 3\nmax_plies: 40\n"
                "log_interval: 1\nseed: 9\nopening_random_plies: 6\n"
                "alphabeta_config: none\n"
                "black_checkpoint: none\nwhite_checkpoint: none\ndevice: none\n",
                encoding="utf-8",
            )

            config = load_arena_config(path)

        self.assertEqual(config.games, 3)
        self.assertEqual(config.opening_random_plies, 6)
        self.assertIsNone(config.device)

    def test_run_arena_with_supervised_checkpoint(self):
        with TemporaryDirectory() as tmp_dir:
            checkpoint = create_zero_supervised_checkpoint(Path(tmp_dir))
            stats = run_arena(
                black_name="supervised",
                white_name="supervised",
                games=1,
                max_plies=4,
                log_interval=1,
                seed=1,
                black_checkpoint_path=str(checkpoint),
                white_checkpoint_path=str(checkpoint),
                device="cpu",
                opening_random_plies=2,
            )

        self.assertEqual(stats.games, 1)


if __name__ == "__main__":
    unittest.main()
