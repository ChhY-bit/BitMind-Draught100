import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from agents import (
    AlphaBetaAgent,
    AlphaBetaConfig,
    GreedyCaptureAgent,
    MaterialConfig,
    MaterialAgent,
    RandomAgent,
    SupervisedAgent,
    evaluate_enhanced_position,
    load_alphabeta_config,
    load_material_config,
    make_agent,
    material_score_for_side,
)
from draughts_env import Move
from model import CandidateMoveNet, ModelConfig


class AgentTests(unittest.TestCase):
    def test_random_agent_returns_legal_action_id(self):
        actions = [
            Move(action_id=3, path=(16, 21), captures=()),
            Move(action_id=7, path=(17, 22), captures=()),
        ]
        agent = RandomAgent(seed=1)

        self.assertIn(agent.select_action({}, actions), {3, 7})

    def test_greedy_capture_prefers_most_captures(self):
        actions = [
            Move(action_id=0, path=(16, 21), captures=()),
            Move(action_id=1, path=(21, 32), captures=(27,)),
            Move(action_id=2, path=(22, 31, 42), captures=(27, 37)),
        ]
        observation = {"squares": [0] * 50, "side_to_move": 0}
        observation["squares"][21] = 1
        agent = GreedyCaptureAgent(seed=1)

        self.assertEqual(agent.select_action(observation, actions), 2)

    def test_make_agent(self):
        self.assertIsInstance(make_agent("random", seed=1), RandomAgent)
        self.assertIsInstance(make_agent("greedy", seed=1), GreedyCaptureAgent)
        self.assertIsInstance(make_agent("material", seed=1), MaterialAgent)
        self.assertIsInstance(make_agent("alphabeta", seed=1), AlphaBetaAgent)

        with self.assertRaises(ValueError):
            make_agent("unknown")

    def test_supervised_agent_loads_checkpoint_and_returns_legal_action(self):
        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = create_zero_supervised_checkpoint(Path(tmp_dir))
            agent = SupervisedAgent(str(checkpoint_path), device="cpu")

            action_id = agent.select_action(
                {"side_to_move": 0, "squares": [0] * 50},
                [
                    Move(action_id=3, path=(16, 21), captures=()),
                    Move(action_id=7, path=(17, 22), captures=()),
                ],
                FakeObservationEnv(),
            )

        self.assertEqual(action_id, 3)

    def test_make_supervised_agent_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            make_agent("supervised")

    def test_material_score_for_side(self):
        observation = {"squares": [0] * 50}
        observation["squares"][0] = 1
        observation["squares"][1] = 2
        observation["squares"][2] = 3

        self.assertEqual(material_score_for_side(observation, 0), 250)
        self.assertEqual(material_score_for_side(observation, 1), -250)
        self.assertEqual(
            material_score_for_side(observation, 0, MaterialConfig(man_value=10, king_value=30)),
            30,
        )

    def test_material_agent_prefers_best_material_after_move(self):
        actions = [
            Move(action_id=0, path=(16, 21), captures=()),
            Move(action_id=1, path=(21, 32), captures=(27,)),
        ]
        observation = {"squares": [0] * 50, "side_to_move": 0}
        agent = MaterialAgent(seed=1)
        env = FakeMaterialEnv(
            {
                0: {"squares": [1, 3], "game_result": 0},
                1: {"squares": [1], "game_result": 0},
            }
        )

        self.assertEqual(agent.select_action(observation, actions, env), 1)

    def test_load_material_config_from_yaml_subset(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "material.yaml"
            path.write_text(
                "man_value: 10\nking_value: 40\nwin_value: 999\ndraw_value: 5\n",
                encoding="utf-8",
            )

            config = load_material_config(str(path))

        self.assertEqual(config.man_value, 10)
        self.assertEqual(config.king_value, 40)
        self.assertEqual(config.win_value, 999)
        self.assertEqual(config.draw_value, 5)

    def test_load_alphabeta_config_from_yaml(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "alphabeta.yaml"
            path.write_text(
                "search_depth: 3\nman_value: 10\nking_value: 40\n"
                "win_value: 999\ndraw_value: 5\nadvancement_value: 2\n"
                "center_value: 7\nedge_penalty: 3\nback_row_guard_value: 4\n",
                encoding="utf-8",
            )

            config = load_alphabeta_config(str(path))

        self.assertEqual(config.search_depth, 3)
        self.assertEqual(config.man_value, 10)
        self.assertEqual(config.king_value, 40)
        self.assertEqual(config.win_value, 999)
        self.assertEqual(config.draw_value, 5)
        self.assertEqual(config.advancement_value, 2)
        self.assertEqual(config.center_value, 7)
        self.assertEqual(config.edge_penalty, 3)
        self.assertEqual(config.back_row_guard_value, 4)

    def test_enhanced_evaluation_scores_advancement_and_center(self):
        less_active = make_observation(0, [6], [46])
        more_active = make_observation(0, [26], [46])
        config = AlphaBetaConfig(
            man_value=100,
            king_value=250,
            advancement_value=5,
            center_value=10,
            edge_penalty=0,
            back_row_guard_value=0,
        )

        self.assertGreater(
            evaluate_enhanced_position(more_active, 0, config),
            evaluate_enhanced_position(less_active, 0, config),
        )

    def test_enhanced_evaluation_penalizes_edge_piece(self):
        edge_piece = make_observation(0, [6], [])
        central_piece = make_observation(0, [7], [])
        config = AlphaBetaConfig(
            man_value=100,
            king_value=250,
            advancement_value=0,
            center_value=0,
            edge_penalty=10,
            back_row_guard_value=0,
        )

        self.assertLess(
            evaluate_enhanced_position(edge_piece, 0, config),
            evaluate_enhanced_position(central_piece, 0, config),
        )

    def test_alphabeta_depth_one_uses_leaf_evaluation(self):
        actions = [
            Move(action_id=0, path=(16, 21), captures=()),
            Move(action_id=1, path=(21, 32), captures=(27,)),
        ]
        observation = {"squares": [0] * 50, "side_to_move": 0}
        env = FakeMaterialEnv(
            {
                0: {"squares": [1, 3], "game_result": 0},
                1: {"squares": [1], "game_result": 0},
            }
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "alphabeta.yaml"
            path.write_text("search_depth: 1\n", encoding="utf-8")
            agent = AlphaBetaAgent(seed=1, config_path=str(path))

            self.assertEqual(agent.select_action(observation, actions, env), 1)

    def test_alphabeta_depth_two_avoids_immediate_material_trap(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "alphabeta.yaml"
            path.write_text("search_depth: 2\n", encoding="utf-8")
            agent = AlphaBetaAgent(seed=1, config_path=str(path))

            action_id = agent.select_action(
                SEARCH_TREE["root"]["observation"],
                SEARCH_TREE["root"]["actions"],
                FakeSearchEnv("root"),
            )

        self.assertEqual(action_id, 1)


class FakeMaterialEnv:
    def __init__(self, observations):
        self._observations = observations
        self._selected_action = None

    def clone(self):
        return FakeMaterialEnv(self._observations)

    def step(self, action_id):
        self._selected_action = action_id

    def raw_observation(self):
        return self._observations[self._selected_action]

    def legal_actions(self):
        return []


class FakeObservationEnv:
    def observation(self):
        return [[[0 for _col in range(10)] for _row in range(10)] for _channel in range(6)]


def create_zero_supervised_checkpoint(directory: Path) -> Path:
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


def make_observation(side_to_move, black_men, white_men, game_result=0):
    squares = [0] * 50
    for square_id in black_men:
        squares[square_id - 1] = 1
    for square_id in white_men:
        squares[square_id - 1] = 3
    return {
        "squares": squares,
        "side_to_move": side_to_move,
        "game_result": game_result,
    }


SEARCH_TREE = {
    "root": {
        "observation": make_observation(0, [16, 17], [31]),
        "actions": [
            Move(action_id=0, path=(16, 21), captures=()),
            Move(action_id=1, path=(17, 22), captures=()),
        ],
        "next": {0: "trap_reply", 1: "quiet_reply"},
    },
    "trap_reply": {
        "observation": make_observation(1, [16, 17, 18], [31]),
        "actions": [Move(action_id=10, path=(31, 26), captures=())],
        "next": {10: "trap_leaf"},
    },
    "quiet_reply": {
        "observation": make_observation(1, [16, 17], [31, 32]),
        "actions": [Move(action_id=11, path=(32, 27), captures=())],
        "next": {11: "quiet_leaf"},
    },
    "trap_leaf": {
        "observation": make_observation(0, [16], [31, 32, 33]),
        "actions": [],
        "next": {},
    },
    "quiet_leaf": {
        "observation": make_observation(0, [16, 17], [31, 32]),
        "actions": [],
        "next": {},
    },
}


class FakeSearchEnv:
    def __init__(self, node_name):
        self._node_name = node_name

    def clone(self):
        return FakeSearchEnv(self._node_name)

    def step(self, action_id):
        self._node_name = SEARCH_TREE[self._node_name]["next"][action_id]

    def raw_observation(self):
        return SEARCH_TREE[self._node_name]["observation"]

    def legal_actions(self):
        return SEARCH_TREE[self._node_name]["actions"]


if __name__ == "__main__":
    unittest.main()
