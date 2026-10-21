from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence

from draughts_env import DraughtsEnv, Move

from .agents_local import (
    AlphaBetaPlayer,
    HumanPlayer,
    MCTSPlayer,
    RandomPlayer,
    SupervisedPlayer,
    available_devices,
)
from draughts.search_control import SearchCancelled
from draughts.notation import GameRecord, result_code
from .protocol import PlayerAnalysis
from .utils import (
    BLACK_KING,
    BLACK_MAN,
    BLACK_SIDE,
    WHITE_KING,
    WHITE_MAN,
    WHITE_SIDE,
    find_move_by_action_id,
    format_move,
    group_moves_by_start,
    make_notation,
    opponent_side,
)


@dataclass
class _AITask:
    generation: int
    side: int
    position: tuple
    player: object
    single_step: bool
    cancel_event: threading.Event = field(default_factory=threading.Event)
    results: Queue = field(default_factory=Queue)


class GameController:
    def __init__(self, app: "DraughtsApp") -> None:
        self.app = app
        self.env = DraughtsEnv()
        self.rng = random.Random()
        self.players = {
            BLACK_SIDE: HumanPlayer(app._tr("player_name_human", side=app._tr("side_black"))),
            WHITE_SIDE: HumanPlayer(app._tr("player_name_human", side=app._tr("side_white"))),
        }
        self.selected_square: Optional[int] = None
        self.candidate_moves: List[Move] = []
        self.last_move: Optional[Move] = None
        self.action_history: List[int] = []
        self.record = GameRecord()
        self.turn_index = 0
        self.ai_paused = False
        self.ai_job: Optional[str] = None
        self._ai_generation: int = 0  # guards against stale thread results
        self._ai_task: Optional[_AITask] = None
        self._ai_poll_job: Optional[str] = None
        self._pending_ai_request: Optional[bool] = None
        self._closing = False
        self._accepted_analysis: Dict[int, PlayerAnalysis] = {}
        self._value_history: deque = deque(maxlen=300)
        self._forfeit_winner: Optional[int] = None

    # -- value-curve helpers -----------------------------------------------

    def _find_model_player(self):
        """Return the first player that can run inference, or None."""
        for side in (BLACK_SIDE, WHITE_SIDE):
            player = self.players.get(side)
            if player is not None and hasattr(player, "infer_value"):
                return player
        return None

    def _record_position_value(self, ply: int) -> None:
        """Infer the value of the current position and record it for the curve.

        Uses mixed data-collection: the network is evaluated after every
        move, whether the move was made by a human or an AI.  Also feeds
        the inferred value to the eval bar so it stays meaningful even
        during human turns (instead of defaulting to 50 %).
        """
        if self._ai_task is not None:
            return  # a cancelled worker may still be finishing an inference
        model_player = self._find_model_player()
        if model_player is None:
            return  # no AI in game → curve stays empty
        try:
            value = model_player.infer_value(self.env)
        except Exception:
            return
        if value is None:
            return
        notation = self.notation_history[-1] if self.notation_history else ""
        side = self.current_side()
        self._value_history.append((ply, float(value), side, notation))
        self.app.update_curve(self._value_history)
        # Also drive the eval bar from the fresh inference (human turns
        # would otherwise see a stale or neutral bar).
        self.app.update_eval_bar(PlayerAnalysis(value=float(value)), side)

    def _update_eval_bar_from_inference(self) -> None:
        """Re-infer the current position and update only the eval bar.

        Used after undo / rewind to refresh the bar without adding a
        duplicate point to the value curve.
        """
        if self._ai_task is not None:
            return
        model_player = self._find_model_player()
        if model_player is None:
            return
        try:
            value = model_player.infer_value(self.env)
        except Exception:
            return
        if value is not None:
            self.app.update_eval_bar(
                PlayerAnalysis(value=float(value)), self.current_side(),
            )

    # ------------------------------------------------------------------

    def start_new_game(
        self,
        black_kind: str,
        white_kind: str,
        checkpoint_paths: Optional[Dict[int, str]] = None,
        mcts_config_paths: Optional[Dict[int, str]] = None,
        force_fp16: bool = False,
        force_compile: bool = False,
        device: str = "cpu",
        first_player: str = "先手方",
        second_player: str = "后手方",
    ) -> None:
        self.cancel_ai_job()
        new_record = GameRecord(first_player=first_player, second_player=second_player)
        new_record.to_text()  # validate metadata before replacing the current game
        checkpoint_paths = checkpoint_paths or {}
        mcts_config_paths = mcts_config_paths or {}
        if {black_kind, white_kind} & {"mcts", "supervised"}:
            if device not in available_devices():
                raise ValueError(self.app._tr("dialog_device_unavailable", device=device))
        self.players = {
            BLACK_SIDE: self._make_player(
                BLACK_SIDE, black_kind,
                checkpoint_paths.get(BLACK_SIDE),
                mcts_config_paths.get(BLACK_SIDE),
                force_fp16, force_compile, device,
            ),
            WHITE_SIDE: self._make_player(
                WHITE_SIDE, white_kind,
                checkpoint_paths.get(WHITE_SIDE),
                mcts_config_paths.get(WHITE_SIDE),
                force_fp16, force_compile, device,
            ),
        }
        self.record = new_record
        self.ai_paused = False
        self.app.update_pause_button(self.ai_paused)
        self.reset_current_game(clear_log=True)
        self.log_model_players()

    def reset_current_game(self, clear_log: bool = True) -> None:
        self.cancel_ai_job()
        self.env.reset()
        self.selected_square = None
        self.candidate_moves = []
        self.last_move = None
        self.action_history = []
        self.record = GameRecord(
            first_player=self.record.first_player, second_player=self.record.second_player,
        )
        self.turn_index = 0
        self._value_history.clear()
        self._forfeit_winner = None
        self._accepted_analysis.clear()
        self.app.clear_curve()
        if clear_log:
            self.app.clear_log()
        self.app.log(self.app._tr("log_new_game", mode=self.mode_label()), "info")
        self.refresh()
        # Record initial position value (ply 0) for the curve
        self._record_position_value(0)
        self.schedule_ai_turn()

    def open_new_game_dialog(self) -> None:
        self.cancel_ai_job()
        self.app.show_mode_dialog(
            self.players[BLACK_SIDE].kind,
            self.players[WHITE_SIDE].kind,
        )
        self.schedule_ai_turn()

    def toggle_pause(self) -> None:
        self.ai_paused = not self.ai_paused
        self.app.update_pause_button(self.ai_paused)
        if self.ai_paused:
            self.cancel_ai_job()
            self.app.log(self.app._tr("log_ai_paused"), "warn")
        else:
            self.app.log(self.app._tr("log_ai_resumed"), "info")
            self.schedule_ai_turn()

    def ai_step_once(self) -> None:
        self.cancel_ai_job()
        if self.is_terminal():
            self.app.log(self.app._tr("log_game_ended_ai"), "warn")
            return
        player = self.current_player()
        if player.is_human():
            self.app.log(self.app._tr("log_human_no_ai_step"), "warn")
            return
        self.perform_ai_move(single_step=True)

    def refresh(self) -> None:
        observation = self.env.observation()
        legal_moves = self.env.legal_moves()
        landing_squares = [move.path[-1] for move in self.candidate_moves]
        path_squares = self.last_move.path if self.last_move is not None else []
        analysis = self.last_player_analysis()
        analysis_side = opponent_side(int(observation["side_to_move"]))
        self.app.update_board(
            observation,
            selected_square=self.selected_square,
            landing_squares=landing_squares,
            path_squares=path_squares,
        )
        self.app.update_status(observation, self.last_move)
        # Override result display when the game ended via forfeit
        if self._forfeit_winner is not None:
            winner_key = (
                "black_win" if self._forfeit_winner == BLACK_SIDE else "white_win"
            )
            self.app.status_vars["result"].set(
                self.app._tr(f"result_{winner_key}")
            )
        self.app.update_side_move_list(
            self.current_side(), legal_moves, self.candidate_moves, None,
        )
        self.app._highlight_active_side(self.current_side())
        notation = self.notation_history
        if self.record.result != "*":
            notation.append(self.record.result)
        self.app.update_notation(notation)
        # Eval bar is driven by _record_position_value (fresh inference
        # after every move) and perform_ai_move._on_done (pre-move AI
        # display).  Stored player analysis may be stale, so we skip it.
        self.app.update_search_stats(analysis, analysis_side)

    def handle_square_click(self, square_id: int) -> None:
        observation = self.env.observation()
        if observation["game_result_name"] != "ongoing" or self._forfeit_winner is not None:
            self.app.log(self.app._tr("log_game_over_btn"), "warn")
            return
        if not self.current_player().is_human():
            self.app.log(self.app._tr("log_ai_controlled"), "warn")
            return

        moves = self.env.legal_moves()
        moves_by_start = group_moves_by_start(moves)

        # ── First click: select a movable piece ──
        if self.selected_square is None:
            if square_id in moves_by_start:
                self._select_start(square_id, moves_by_start)
            elif self._is_any_piece(square_id):
                self._handle_illegal_piece(square_id)
            else:
                self._clear_selection("log_no_legal_moves")
            return

        # ── Already have a selection ──
        if square_id == self.selected_square:
            self._clear_selection("log_selection_cancel")
            return

        # Click on a landing square?
        landing_matches = [
            move for move in self.candidate_moves if move.path[-1] == square_id
        ]
        if len(landing_matches) == 1:
            self.execute_move(landing_matches[0])
            return
        if len(landing_matches) > 1:
            self.candidate_moves = landing_matches
            self.app.log(
                self.app._tr("log_ambiguous", count=len(landing_matches), sq=square_id),
                "warn",
            )
            self.refresh()
            return

        # Not a landing square — switch to a different movable piece?
        if square_id in moves_by_start:
            self._select_start(square_id, moves_by_start)
        else:
            self._handle_illegal_piece(square_id)

    def execute_move(self, move: Move, actor: Optional[str] = None) -> None:
        if self.is_terminal():
            return
        self.cancel_ai_job()
        legal_move = find_move_by_action_id(self.env.legal_moves(), move.action_id)
        if legal_move != move:
            self._handle_program_error(self.app._tr("log_action_stale", aid=move.action_id))
            return
        try:
            observation, info = self.env.step(move.action_id)
        except RuntimeError as exc:
            self._handle_program_error(self.app._tr("log_exec_fail", error=str(exc)))
            return

        self.turn_index += 1
        self.last_move = move
        self.action_history.append(move.action_id)
        self.record.moves.append(move)
        self.record.result = result_code(int(info["game_result"]))
        self.selected_square = None
        self.candidate_moves = []
        actor_label = actor or self.players[opponent_side(self.current_side())].name
        self.app.log(
            self.app._tr("log_move", turn=self.turn_index, actor=actor_label,
                         aid=move.action_id, move=format_move(move)),
            "action",
        )
        if info["done"]:
            self.app.log(
                self.app._tr("log_game_result",
                             result=self.app._tr(f"result_{observation['game_result_name']}")),
                "success",
            )
        # Record post-move evaluation for the value curve
        self._record_position_value(self.turn_index)
        self.refresh()
        self.schedule_ai_turn()

    def undo_last_move(self) -> None:
        self.cancel_ai_job()
        self._accepted_analysis.clear()
        if self._forfeit_winner is not None:
            self.app.log(self.app._tr("log_game_over_btn"), "warn")
            return
        if not self.action_history:
            self.app.log(self.app._tr("log_no_undo"), "warn")
            self.schedule_ai_turn()
            return

        removed_notation = make_notation(self.turn_index, self.record.moves[-1])
        replay_actions = self.action_history[:-1]
        rebuilt_moves: List[Move] = []

        self.env.reset()
        try:
            for action_id in replay_actions:
                move = find_move_by_action_id(self.env.legal_moves(), action_id)
                if move is None:
                    raise RuntimeError(f"重放时找不到动作 {action_id}")
                self.env.step(action_id)
                rebuilt_moves.append(move)
        except RuntimeError as exc:
            self.app.log(self.app._tr("log_undo_fail", error=str(exc)), "error")
            self.env.reset()
            self.action_history = []
            self.record.moves.clear()
            self.record.result = "*"
            self.last_move = None
            self.turn_index = 0
            self._value_history.clear()
            self.app.update_curve(self._value_history)
            self.refresh()
            return

        self.action_history = list(replay_actions)
        self.record.moves = rebuilt_moves
        self.record.result = "*"
        self.turn_index = len(self.action_history)
        self.last_move = rebuilt_moves[-1] if rebuilt_moves else None
        self.selected_square = None
        self.candidate_moves = []
        # Trim value history to current ply count
        while len(self._value_history) > self.turn_index:
            self._value_history.pop()
        self.app.update_curve(self._value_history)
        # Re-infer eval bar for the restored position
        self._update_eval_bar_from_inference()
        self.app.log(self.app._tr("log_undone", notation=removed_notation), "warn")
        self.refresh()
        self.schedule_ai_turn()

    def rewind_to_ply(self, target_ply: int) -> None:
        """Rewind the game to the position after *target_ply* plies.

        Auto-pauses AI, cancels any in-flight computation, replays the
        action history up to *target_ply*, trims the value history, and
        marks the curve at the new current ply.
        """
        if self._forfeit_winner is not None:
            self.app.log(self.app._tr("log_game_over_btn"), "warn")
            return
        target_ply = max(0, min(target_ply, len(self.action_history)))
        if target_ply == self.turn_index:
            return  # already there

        # Auto-pause so the user can inspect the position
        if not self.ai_paused:
            self.ai_paused = True
            self.app.update_pause_button(True)
        self.cancel_ai_job()

        replay_actions = self.action_history[:target_ply]
        self._accepted_analysis.clear()
        rebuilt_moves: List[Move] = []

        self.env.reset()
        try:
            for action_id in replay_actions:
                move = find_move_by_action_id(self.env.legal_moves(), action_id)
                if move is None:
                    raise RuntimeError(f"Rewind failed at action {action_id}")
                self.env.step(action_id)
                rebuilt_moves.append(move)
        except RuntimeError as exc:
            self.app.log(
                self.app._tr("log_undo_fail", error=str(exc)), "error",
            )
            self.env.reset()
            self.action_history = []
            self.record.moves.clear()
            self.record.result = "*"
            self._value_history.clear()
            self.last_move = None
            self.turn_index = 0
            self.app.update_curve(self._value_history)
            self.refresh()
            return

        self.action_history = list(replay_actions)
        self.record.moves = rebuilt_moves
        self.record.result = "*"
        self.turn_index = target_ply
        self.last_move = rebuilt_moves[-1] if rebuilt_moves else None
        self.selected_square = None
        self.candidate_moves = []

        # Trim value history
        while len(self._value_history) > target_ply:
            self._value_history.pop()
        self.app.update_curve(self._value_history)
        self.app.curve.set_current_ply(target_ply)
        self._update_eval_bar_from_inference()

        self.app.log(
            self.app._tr("log_rewind_paused", ply=target_ply), "warn",
        )
        self.refresh()

    def perform_ai_move(self, *, single_step: bool = False) -> None:
        if self._closing or self.is_terminal() or (self.ai_paused and not single_step):
            return
        player = self.current_player()
        if player.is_human():
            return

        if self._ai_task is not None:
            self._ai_task.cancel_event.set()
            self._pending_ai_request = single_step
            return

        snapshot = self.env.clone()
        legal_moves = snapshot.legal_moves()
        current_side = self.current_side()
        self._ai_generation += 1
        task = _AITask(
            self._ai_generation, current_side, self._position_key(), player, single_step
        )
        self._ai_task = task

        # Indicate thinking state
        self.app.status_vars["side"].set(
            self.app._tr("side_black" if current_side == BLACK_SIDE else "side_white")
            + " …"
        )

        def _compute() -> None:
            try:
                chosen = player.choose_move(
                    legal_moves, snapshot, cancel_event=task.cancel_event
                )
                analysis = getattr(player, "last_analysis", PlayerAnalysis())
                task.results.put((chosen, analysis, None))
            except Exception as exc:
                task.results.put((None, PlayerAnalysis(), exc))

        threading.Thread(target=_compute, daemon=True).start()
        self._ai_poll_job = self.app.after(25, self._poll_ai_result)

    def _position_key(self) -> tuple:
        observation = self.env.observation()
        return tuple(observation["squares"]), int(observation["side_to_move"])

    def _poll_ai_result(self) -> None:
        """Read worker results on the Tk thread; workers never call Tk APIs."""
        self._ai_poll_job = None
        task = self._ai_task
        if self._closing or task is None:
            return
        try:
            move, analysis, error = task.results.get_nowait()
        except Empty:
            self._ai_poll_job = self.app.after(25, self._poll_ai_result)
            return

        self._ai_task = None
        pending = self._pending_ai_request
        self._pending_ai_request = None
        valid = (
            task.generation == self._ai_generation
            and not task.cancel_event.is_set()
            and not self.is_terminal()
            and task.position == self._position_key()
            and self.players.get(task.side) is task.player
            and (not self.ai_paused or task.single_step)
        )
        if not valid or isinstance(error, SearchCancelled):
            self.refresh()
            if pending is not None:
                self.perform_ai_move(single_step=pending)
            return
        if error is not None:
            self._handle_program_error(self.app._tr("log_ai_error", error=str(error)))
            return
        if move is None:
            self._handle_program_error(self.app._tr("log_no_moves", name=task.player.name))
            return
        if move not in self.env.legal_moves():
            self._handle_program_error(self.app._tr("log_ai_invalid_move"))
            return

        self._accepted_analysis[task.side] = analysis
        self.app.update_side_move_list(
            task.side, self.env.legal_moves(), self.candidate_moves, analysis,
        )
        self.app.update_eval_bar(analysis, task.side)
        self.app.update_search_stats(analysis, task.side)
        self.app._highlight_active_side(task.side)
        self.execute_move(move, actor=task.player.name)

    def _handle_program_error(self, reason: str) -> None:
        self.cancel_ai_job()
        self.ai_paused = True
        self.app.update_pause_button(True)
        self.selected_square = None
        self.candidate_moves = []
        self.app.log(reason, "error")
        self.refresh()
        confirmed = self.app.show_forfeit_dialog(reason, self.current_side())
        if self._closing:
            return
        if confirmed:
            self.forfeit(self.current_side())
        else:
            self.app.log(self.app._tr("log_retry_move"), "warn")

    def close(self) -> None:
        self._closing = True
        self.cancel_ai_job()
        if self._ai_poll_job is not None:
            self.app.after_cancel(self._ai_poll_job)
            self._ai_poll_job = None

    def last_player_analysis(self) -> PlayerAnalysis:
        """Return the analysis data from the last AI move, if any."""
        side = opponent_side(self.current_side())
        return self._accepted_analysis.get(side, PlayerAnalysis())

    def schedule_ai_turn(self) -> None:
        self.cancel_ai_job()
        if self._closing or self.ai_paused or self.is_terminal():
            return
        if self.current_player().is_human():
            return
        self.ai_job = self.app.after(450, self._run_scheduled_ai_turn)

    def cancel_ai_job(self) -> None:
        if self.ai_job is not None:
            self.app.after_cancel(self.ai_job)
            self.ai_job = None
        self._ai_generation += 1  # invalidate any in-flight AI thread
        self._pending_ai_request = None
        if self._ai_task is not None:
            self._ai_task.cancel_event.set()

    def current_side(self) -> int:
        return int(self.env.observation()["side_to_move"])

    def current_player(self):
        return self.players[self.current_side()]

    def _is_any_piece(self, square_id: int) -> bool:
        """Return True if *square_id* holds any piece (own or opponent)."""
        piece = int(self.env.observation()["squares"][square_id - 1])
        return piece in (BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING)

    def forfeit(self, side: int) -> None:
        """Declare *side* as forfeited — the opponent wins.

        Records the forfeit in the notation, logs it, and marks the
        game as terminal so no further moves can be played.
        """
        if self.is_terminal():
            return
        self.cancel_ai_job()
        winner = opponent_side(side)
        self._forfeit_winner = winner
        self.record.result = result_code(1 if winner == BLACK_SIDE else 2)
        self.selected_square = None
        self.candidate_moves = []

        loser_name = self.app._tr(
            "side_black" if side == BLACK_SIDE else "side_white"
        )
        winner_name = self.app._tr(
            "side_black" if winner == BLACK_SIDE else "side_white"
        )

        self.app.log(
            self.app._tr("log_forfeit", loser=loser_name, winner=winner_name),
            "error",
        )
        self.refresh()

    def is_terminal(self) -> bool:
        if self._forfeit_winner is not None:
            return True
        return self.env.observation()["game_result_name"] != "ongoing"

    @property
    def notation_history(self) -> List[str]:
        return self.record.move_lines()

    def record_snapshot(self) -> GameRecord:
        return self.record.snapshot()

    def mode_label(self) -> str:
        kind_black = self.app._tr(f"kind_{self.players[BLACK_SIDE].kind}")
        kind_white = self.app._tr(f"kind_{self.players[WHITE_SIDE].kind}")
        return self.app._tr("mode_label", kind_black=kind_black, kind_white=kind_white)

    def _run_scheduled_ai_turn(self) -> None:
        self.ai_job = None
        if not self.ai_paused:
            self.perform_ai_move()

    def log_model_players(self) -> None:
        for side in (BLACK_SIDE, WHITE_SIDE):
            player = self.players[side]
            kind = getattr(player, "kind", None)
            fp16_on = getattr(player, "force_fp16", False)
            compile_on = getattr(player, "force_compile", False)
            flags = []
            if fp16_on:
                flags.append("FP16")
            if compile_on:
                flags.append("compile")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            if kind in {"supervised", "mcts"}:
                flag_str += f" [device={player.agent.device_obj}]"
            if kind == "supervised":
                self.app.log(
                    self.app._tr("log_model_loaded", name=player.name,
                                 file=Path(player.checkpoint_path).name) + flag_str,
                    "info",
                )
            elif kind == "mcts":
                cfg = Path(player.config_path).name if player.config_path else "default"
                self.app.log(
                    self.app._tr("log_model_loaded_mcts", name=player.name,
                                 file=Path(player.checkpoint_path).name, cfg=cfg) + flag_str,
                    "info",
                )

    def _make_player(self, side: int, kind: str, checkpoint_path: Optional[str] = None,
                     mcts_config_path: Optional[str] = None,
                     force_fp16: bool = False, force_compile: bool = False,
                     device: str = "cpu"):
        side_key = "side_black" if side == BLACK_SIDE else "side_white"
        side_label = self.app._tr(side_key)
        kind_label = self.app._tr(f"kind_{kind}")
        if kind == "human":
            return HumanPlayer(self.app._tr("player_name_human", side=side_label))
        if kind == "random":
            return RandomPlayer(self.app._tr("player_name", side=side_label, kind=kind_label), self.rng)
        if kind == "alphabeta":
            return AlphaBetaPlayer(self.app._tr("player_name", side=side_label, kind=kind_label))
        if kind == "supervised":
            if not checkpoint_path:
                raise ValueError(self.app._tr("log_need_checkpoint", side=side_label))
            return SupervisedPlayer(
                self.app._tr("player_name", side=side_label, kind=kind_label),
                checkpoint_path,
                force_fp16=force_fp16,
                force_compile=force_compile,
                device=device,
            )
        if kind == "mcts":
            if not checkpoint_path:
                raise ValueError(self.app._tr("log_need_mcts_ckpt", side=side_label))
            return MCTSPlayer(
                self.app._tr("player_name", side=side_label, kind=kind_label),
                checkpoint_path, mcts_config_path,
                force_fp16=force_fp16,
                force_compile=force_compile,
                device=device,
            )
        return HumanPlayer(self.app._tr("player_name_human", side=side_label))

    def _select_start(
        self,
        square_id: int,
        moves_by_start: Dict[int, List[Move]],
    ) -> None:
        self.selected_square = square_id
        self.candidate_moves = moves_by_start[square_id]
        self.app.log(
            self.app._tr("log_square_selected", sq=square_id,
                         count=len(self.candidate_moves)),
            "info",
        )
        self.refresh()

    def _handle_illegal_piece(self, square_id: int) -> None:
        """Show the illegal-move dialog and act on the user's choice."""
        forfeited = self.app.show_illegal_move_dialog(square_id)
        if self._closing:
            return
        if forfeited:
            self.forfeit(self.current_side())
            return
        # User chose to continue — clear selection and keep playing
        self._clear_selection("log_selection_cancel")

    def _clear_selection(self, key: str, **fmt) -> None:
        self.selected_square = None
        self.candidate_moves = []
        self.app.log(self.app._tr(key, **fmt))
        self.refresh()
