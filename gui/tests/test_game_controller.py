from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
import re
from unittest.mock import Mock, patch

from draughts.search_control import SearchCancelled
from draughts_env import Move
from gui.game_controller import GameController
from gui.i18n import _tr_static
from gui.protocol import PlayerAnalysis


class FakeVariable:
    def set(self, value):
        self.value = value


class FakeApp:
    """A main-thread event loop without a display or Tk dependency."""

    def __init__(self):
        self.main_thread = threading.get_ident()
        self.jobs = {}
        self.next_job = 0
        self.logs = []
        self.dialogs = []
        self.confirm_forfeit = False
        self.status_vars = {key: FakeVariable() for key in ("side", "result")}
        self.curve = Mock()

    def _tr(self, key, **kwargs):
        return _tr_static(key, "zh").format(**kwargs)

    def after(self, delay, callback):
        self.assert_main_thread()
        self.next_job += 1
        job = str(self.next_job)
        self.jobs[job] = delay, callback
        return job

    def after_cancel(self, job):
        self.assert_main_thread()
        self.jobs.pop(job, None)

    def assert_main_thread(self):
        if threading.get_ident() != self.main_thread:
            raise AssertionError("worker called the GUI")

    def pump(self):
        for job, (delay, callback) in list(self.jobs.items()):
            if delay <= 25 and job in self.jobs:
                del self.jobs[job]
                callback()

    def log(self, *args):
        self.assert_main_thread()
        self.logs.append(args)

    def show_forfeit_dialog(self, reason, side):
        self.assert_main_thread()
        self.dialogs.append((reason, side))
        return self.confirm_forfeit

    def show_illegal_move_dialog(self, square):
        return self.show_forfeit_dialog(str(square), 0)

    def _update(self, *args, **kwargs):
        self.assert_main_thread()

    update_pause_button = update_board = update_status = _update
    update_side_move_list = update_notation = update_search_stats = _update
    update_eval_bar = update_curve = clear_curve = clear_log = _update
    _highlight_active_side = _update


class Player:
    name = "test player"
    kind = "random"

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = 0
        self.last_analysis = PlayerAnalysis(value=0.25)

    def is_human(self):
        return False

    def choose_move(self, moves, env, *, cancel_event=None):
        self.calls += 1
        result = self.results.pop(0) if self.results else moves[0]
        if isinstance(result, Exception):
            raise result
        return result


class BlockingPlayer(Player):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.active = 0
        self.max_active = 0
        self.snapshot = None
        self.cancel_event = None

    def choose_move(self, moves, env, *, cancel_event=None):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.snapshot = env
        self.cancel_event = cancel_event
        self.started.set()
        try:
            # Deliberately ignore cancellation to simulate a forward pass
            # that must finish before a replacement search can start.
            if not self.release.wait(3):
                raise RuntimeError("test worker was not released")
            return moves[0]
        finally:
            self.active -= 1
            self.finished.set()


class GameControllerTests(unittest.TestCase):
    def setUp(self):
        self.app = FakeApp()
        self.game = GameController(self.app)
        self.game.ai_paused = True

    def tearDown(self):
        self.game.close()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.app.pump()
            threading.Event().wait(0.001)
        self.assertTrue(predicate(), "GUI task did not finish")

    def start_blocking_player(self):
        player = BlockingPlayer()
        self.addCleanup(player.release.set)
        self.game.players[0] = player
        self.game.ai_step_once()
        self.assertTrue(player.started.wait(1))
        return player

    def test_single_step_works_while_paused_and_does_not_resume_autoplay(self):
        self.game.players[0] = Player()
        self.game.ai_step_once()
        self.wait_until(lambda: self.game.turn_index == 1)
        self.assertTrue(self.game.ai_paused)
        self.assertIsNone(self.game.ai_job)
        self.assertEqual(len(self.game.action_history), 1)

    def test_pause_cancels_search_and_discards_its_result(self):
        self.game.ai_paused = False
        player = self.start_blocking_player()
        before = self.game.env.observation()
        self.game.toggle_pause()
        self.assertTrue(player.cancel_event.is_set())
        player.release.set()
        self.wait_until(lambda: self.game._ai_task is None)
        self.assertEqual(self.game.env.observation(), before)
        self.assertFalse(self.app.dialogs)
        self.assertEqual(self.game.last_player_analysis(), PlayerAnalysis())

    def test_replacement_waits_for_cancelled_worker_to_finish(self):
        player = self.start_blocking_player()
        self.game.ai_step_once()
        self.assertTrue(player.cancel_event.is_set())
        self.assertEqual(player.calls, 1)
        player.release.set()
        self.wait_until(lambda: self.game.turn_index == 1)
        self.assertEqual(player.calls, 2)
        self.assertEqual(player.max_active, 1)

    def test_reset_discards_result_even_if_initial_position_matches(self):
        player = self.start_blocking_player()
        self.game.reset_current_game()
        player.release.set()
        self.wait_until(lambda: self.game._ai_task is None)
        self.assertEqual(self.game.turn_index, 0)
        self.assertFalse(self.game.action_history)

    def test_search_uses_independent_snapshot_and_checks_current_position(self):
        player = self.start_blocking_player()
        snapshot = player.snapshot.observation()
        self.assertIsNot(player.snapshot, self.game.env)
        self.game.env.step(self.game.env.legal_moves()[0].action_id)
        self.assertEqual(player.snapshot.observation(), snapshot)
        player.release.set()
        self.wait_until(lambda: self.game._ai_task is None)
        self.assertEqual(self.game.turn_index, 0)

    def test_cancelled_search_is_not_reported_as_program_failure(self):
        self.game.players[0] = Player([SearchCancelled()])
        self.game.ai_step_once()
        self.wait_until(lambda: self.game._ai_task is None)
        self.assertFalse(self.app.dialogs)
        self.assertFalse(self.game.is_terminal())

    def test_error_confirm_forfeits_without_counting_a_move(self):
        self.game.players[0] = Player([RuntimeError("broken opponent")])
        self.app.confirm_forfeit = True
        before = self.game.env.observation()
        self.game.ai_step_once()
        self.wait_until(self.game.is_terminal)
        self.assertEqual(self.game._forfeit_winner, 1)
        self.assertEqual(self.game.env.observation(), before)
        self.assertEqual(self.game.turn_index, 0)
        self.assertFalse(self.game.action_history)

    def test_error_cancel_keeps_position_and_allows_manual_retry(self):
        self.game.players[0] = Player([RuntimeError("retry")])
        self.game.ai_paused = False
        before = self.game.env.observation()
        self.game.perform_ai_move()
        self.wait_until(lambda: len(self.app.dialogs) == 1)
        self.assertTrue(self.game.ai_paused)
        self.assertFalse(self.game.is_terminal())
        self.assertEqual(self.game.env.observation(), before)
        self.assertFalse(self.app.jobs)
        self.game.ai_step_once()
        self.wait_until(lambda: self.game.turn_index == 1)

    def test_missing_or_illegal_ai_move_opens_confirmation(self):
        for result in (None, Move(0, (1, 50), ()), object()):
            with self.subTest(result=result):
                self.game.players[0] = Player([result])
                before = self.game.env.observation()
                count = len(self.app.dialogs)
                self.game.ai_step_once()
                self.wait_until(lambda: len(self.app.dialogs) == count + 1)
                self.assertEqual(self.game.env.observation(), before)
                self.assertFalse(self.game.is_terminal())

    def test_illegal_human_click_cancel_requires_reselection(self):
        self.game.selected_square = 16
        self.game.candidate_moves = self.game.env.legal_moves()
        before = self.game.env.observation()
        self.game._handle_illegal_piece(1)
        self.assertFalse(self.game.is_terminal())
        self.assertIsNone(self.game.selected_square)
        self.assertFalse(self.game.candidate_moves)
        self.assertEqual(self.game.env.observation(), before)
        self.game.execute_move(self.game.env.legal_moves()[0])
        self.assertEqual(self.game.turn_index, 1)

    def test_illegal_human_click_confirm_blocks_further_moves(self):
        self.app.confirm_forfeit = True
        move = self.game.env.legal_moves()[0]
        before = self.game.env.observation()
        self.game.handle_square_click(1)
        self.assertTrue(self.game.is_terminal())
        self.game.execute_move(move)
        self.assertEqual(self.game.env.observation(), before)

    def test_move_with_reused_action_id_but_wrong_path_is_rejected(self):
        move = self.game.env.legal_moves()[0]
        before = self.game.env.observation()
        self.game.execute_move(Move(move.action_id, (1, 50), ()))
        self.assertEqual(len(self.app.dialogs), 1)
        self.assertEqual(self.game.env.observation(), before)

    def test_close_cancels_worker_and_all_gui_callbacks(self):
        player = self.start_blocking_player()
        self.game.close()
        self.assertTrue(player.cancel_event.is_set())
        self.assertFalse(self.app.jobs)
        player.release.set()
        self.assertTrue(player.finished.wait(1))
        self.assertEqual(self.game.turn_index, 0)

    def test_inference_is_skipped_while_old_search_finishes(self):
        player = self.start_blocking_player()
        player.infer_value = Mock(return_value=0.0)
        self.game._record_position_value(0)
        self.game._update_eval_bar_from_inference()
        player.infer_value.assert_not_called()

    def test_unavailable_device_is_rejected_before_replacing_players(self):
        players = self.game.players
        with patch("gui.game_controller.available_devices", return_value=["cpu"]):
            with self.assertRaisesRegex(ValueError, "cuda:1"):
                self.game.start_new_game("mcts", "human", device="cuda:1")
        self.assertIs(self.game.players, players)

    def test_selected_device_reaches_model_player(self):
        with patch("gui.game_controller.SupervisedPlayer") as player:
            self.game._make_player(0, "supervised", "model.pt", device="cuda:1")
        self.assertEqual(player.call_args.kwargs["device"], "cuda:1")

    def play_path(self, path):
        move = next(move for move in self.game.env.legal_moves() if move.path == path)
        self.game.execute_move(move)

    def test_display_export_undo_and_rewind_share_complete_move_history(self):
        for path in ((20, 24), (33, 29), (24, 33), (38, 29), (14, 20), (29, 24),
                     (20, 29), (34, 23, 14)):
            self.play_path(path)
        self.assertEqual(self.game.notation_history, [
            "1. 20-24 33-29", "2. 24x33 38x29", "3. 14-20 29-24", "4. 20x29 34x23x14",
        ])
        snapshot = self.game.record_snapshot()
        self.game.undo_last_move()
        self.assertEqual(self.game.notation_history[-1], "4. 20x29")
        self.assertEqual(len(self.game.record.moves), 7)
        self.game.rewind_to_ply(3)
        self.assertEqual(self.game.record.move_lines(), ["1. 20-24 33-29", "2. 24x33"])
        self.assertEqual(len(self.game.action_history), len(self.game.record.moves))
        self.assertEqual(len(snapshot.moves), 8)

    def test_natural_game_result_is_exported_and_cleared_by_undo(self):
        fixture = Path(__file__).resolve().parents[2] / "draughts/tests/fixtures/notation_example_1.txt"
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\d+\. ", line):
                for token in line.split()[1:]:
                    self.play_path(tuple(int(square) for square in re.split("[-x]", token)))
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.record_snapshot().result, "0-1")
        self.assertIn("54. 12-18 04x22x06", self.game.record_snapshot().to_text())
        self.game.undo_last_move()
        self.assertEqual(self.game.record_snapshot().result, "*")
        self.assertFalse(self.game.is_terminal())

    def test_forfeit_exports_actual_winner_without_adding_a_move(self):
        for loser, result in ((0, "0-1"), (1, "1-0")):
            with self.subTest(loser=loser):
                self.game.reset_current_game()
                self.play_path((20, 24))
                self.game.forfeit(loser)
                snapshot = self.game.record_snapshot()
                self.assertEqual(snapshot.result, result)
                self.assertEqual(snapshot.move_lines(), ["1. 20-24"])
                self.assertEqual(self.game.turn_index, 1)

    def test_metadata_is_customizable_and_reset_preserves_names(self):
        self.game.start_new_game("human", "human", first_player="甲队", second_player="乙队")
        self.play_path((20, 24))
        self.game.reset_current_game()
        self.assertEqual(self.game.record.first_player, "甲队")
        self.assertEqual(self.game.record.second_player, "乙队")
        self.assertFalse(self.game.record.moves)
        self.assertEqual(self.game.record.result, "*")


if __name__ == "__main__":
    unittest.main()
