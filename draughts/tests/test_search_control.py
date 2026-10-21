from __future__ import annotations

from threading import Event
import unittest
from unittest.mock import Mock, patch

import torch

from draughts.agents import AlphaBetaAgent, RandomAgent
from draughts.env import StandardDraughtsEnv
from draughts.mcts import MCTSConfig, mcts_search
from draughts.search_control import SearchCancelled


class SearchCancellationTests(unittest.TestCase):
    def setUp(self):
        self.env = StandardDraughtsEnv()
        self.cancel_event = Event()

    def test_cancelled_search_does_not_run_network(self):
        self.cancel_event.set()
        model = Mock()
        with self.assertRaises(SearchCancelled):
            mcts_search(self.env, model, MCTSConfig(), torch.device("cpu"),
                        cancel_event=self.cancel_event)
        model.assert_not_called()

    def test_mcts_cancellation_after_inference_preserves_root(self):
        cancel_event = self.cancel_event

        class CancellingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, boards, moves, mask):
                self.calls += 1
                if self.calls == 2:
                    cancel_event.set()
                return {"policy_logits": torch.zeros(moves.shape[:2]),
                        "value": torch.zeros(boards.shape[0])}

        model = CancellingModel()
        before = self.env.raw_observation()
        with self.assertRaises(SearchCancelled):
            mcts_search(self.env, model, MCTSConfig(num_simulations=100, batch_size=4),
                        torch.device("cpu"), cancel_event=cancel_event)
        self.assertEqual(model.calls, 2)
        self.assertEqual(self.env.raw_observation(), before)

    def test_alphabeta_checks_cancellation_inside_search(self):
        before = self.env.raw_observation()

        def cancel_on_leaf(*_args):
            self.cancel_event.set()
            return 0

        with patch("draughts.agents.evaluate_enhanced_position", side_effect=cancel_on_leaf) as evaluate:
            with self.assertRaises(SearchCancelled):
                AlphaBetaAgent(depth=3).select_action(
                    before, self.env.legal_actions(), self.env,
                    cancel_event=self.cancel_event,
                )
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(self.env.raw_observation(), before)

    def test_random_agent_honours_cancel_before_move_selection(self):
        self.cancel_event.set()
        with self.assertRaises(SearchCancelled):
            RandomAgent().select_action({}, self.env.legal_actions(),
                                        cancel_event=self.cancel_event)


if __name__ == "__main__":
    unittest.main()
