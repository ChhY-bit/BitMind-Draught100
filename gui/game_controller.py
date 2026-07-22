from __future__ import annotations

import random
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from draughts_env import DraughtsEnv, Move

from .agents_local import (
    AlphaBetaPlayer,
    HumanPlayer,
    MCTSPlayer,
    RandomPlayer,
    SupervisedPlayer,
)
from .protocol import PlayerAnalysis
from .utils import (
    BLACK_SIDE,
    WHITE_SIDE,
    find_move_by_action_id,
    format_move,
    group_moves_by_start,
    make_notation,
    opponent_side,
)


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
        self.notation_history: List[str] = []
        self.turn_index = 0
        self.ai_paused = False
        self.ai_job: Optional[str] = None
        self._ai_generation: int = 0  # guards against stale thread results
        self._value_history: deque = deque(maxlen=300)

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
    ) -> None:
        self.cancel_ai_job()
        checkpoint_paths = checkpoint_paths or {}
        mcts_config_paths = mcts_config_paths or {}
        self.players = {
            BLACK_SIDE: self._make_player(
                BLACK_SIDE, black_kind,
                checkpoint_paths.get(BLACK_SIDE),
                mcts_config_paths.get(BLACK_SIDE),
            ),
            WHITE_SIDE: self._make_player(
                WHITE_SIDE, white_kind,
                checkpoint_paths.get(WHITE_SIDE),
                mcts_config_paths.get(WHITE_SIDE),
            ),
        }
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
        self.notation_history = []
        self.turn_index = 0
        self._value_history.clear()
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
        self.perform_ai_move()

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
        self.app.update_side_move_list(
            self.current_side(), legal_moves, self.candidate_moves, None,
        )
        self.app._highlight_active_side(self.current_side())
        self.app.update_notation(self.notation_history)
        # Eval bar is driven by _record_position_value (fresh inference
        # after every move) and perform_ai_move._on_done (pre-move AI
        # display).  Stored player analysis may be stale, so we skip it.
        self.app.update_search_stats(analysis, analysis_side)

    def handle_square_click(self, square_id: int) -> None:
        observation = self.env.observation()
        if observation["game_result_name"] != "ongoing":
            self.app.log(self.app._tr("log_game_over_btn"), "warn")
            return
        if not self.current_player().is_human():
            self.app.log(self.app._tr("log_ai_controlled"), "warn")
            return

        moves = self.env.legal_moves()
        moves_by_start = group_moves_by_start(moves)

        if self.selected_square is None:
            self._select_start(square_id, moves_by_start)
            return

        if square_id == self.selected_square:
            self._clear_selection("log_selection_cancel")
            return

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

        if square_id in moves_by_start:
            self._select_start(square_id, moves_by_start)
            return

        self._clear_selection("log_not_landing")

    def execute_move(self, move: Move, actor: Optional[str] = None) -> None:
        try:
            observation, info = self.env.step(move.action_id)
        except RuntimeError as exc:
            self.app.log(self.app._tr("log_exec_fail", error=str(exc)), "error")
            self.refresh()
            return

        self.turn_index += 1
        self.last_move = move
        self.action_history.append(move.action_id)
        self.notation_history.append(make_notation(self.turn_index, move))
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
        if not self.action_history:
            self.app.log(self.app._tr("log_no_undo"), "warn")
            self.schedule_ai_turn()
            return

        removed_notation = self.notation_history[-1]
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
            self.notation_history = []
            self.last_move = None
            self.turn_index = 0
            self._value_history.clear()
            self.app.update_curve(self._value_history)
            self.refresh()
            return

        self.action_history = list(replay_actions)
        self.notation_history = [
            make_notation(index + 1, move) for index, move in enumerate(rebuilt_moves)
        ]
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
        target_ply = max(0, min(target_ply, len(self.action_history)))
        if target_ply == self.turn_index:
            return  # already there

        # Auto-pause so the user can inspect the position
        if not self.ai_paused:
            self.ai_paused = True
            self.app.update_pause_button(True)
        self.cancel_ai_job()

        replay_actions = self.action_history[:target_ply]
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
            self.notation_history = []
            self._value_history.clear()
            self.last_move = None
            self.turn_index = 0
            self.app.update_curve(self._value_history)
            self.refresh()
            return

        self.action_history = list(replay_actions)
        self.notation_history = [
            make_notation(i + 1, m) for i, m in enumerate(rebuilt_moves)
        ]
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

    def perform_ai_move(self) -> None:
        if self.is_terminal():
            return
        player = self.current_player()
        if player.is_human():
            return

        legal_moves = self.env.legal_moves()
        current_side = self.current_side()
        self._ai_generation += 1
        gen = self._ai_generation

        # Indicate thinking state
        self.app.status_vars["side"].set(
            self.app._tr("side_black" if current_side == BLACK_SIDE else "side_white")
            + " …"
        )

        # Run the heavy AI computation on a background thread so the
        # GUI stays responsive during search.
        result: dict = {}

        def _compute() -> None:
            try:
                chosen = player.choose_move(legal_moves, self.env)
                analysis = getattr(player, "last_analysis", PlayerAnalysis())
                result["move"] = chosen
                result["analysis"] = analysis
            except Exception as exc:
                result["error"] = exc
            self.app.after(0, _on_done)

        def _on_done() -> None:
            if gen != self._ai_generation:
                return  # stale — game was reset / paused / undone
            if self.ai_paused or self.is_terminal():
                return  # state changed while AI was thinking
            err = result.get("error")
            if err is not None:
                self.app.log(f"AI error: {err}", "error")
                return
            move = result.get("move")
            analysis = result.get("analysis", PlayerAnalysis())
            if move is None:
                self.app.log(
                    self.app._tr("log_no_moves", name=player.name), "warn",
                )
                return

            # Show the AI's analysis on the position it was computed for
            self.app.update_side_move_list(
                current_side, legal_moves, self.candidate_moves, analysis,
            )
            self.app.update_eval_bar(analysis, current_side)
            self.app.update_search_stats(analysis, current_side)
            self.app._highlight_active_side(current_side)

            self.execute_move(move, actor=player.name)

        threading.Thread(target=_compute, daemon=True).start()

    def last_player_analysis(self) -> PlayerAnalysis:
        """Return the analysis data from the last AI move, if any."""
        side = opponent_side(self.current_side())
        player = self.players.get(side)
        if player is not None:
            return getattr(player, "last_analysis", PlayerAnalysis())
        return PlayerAnalysis()

    def schedule_ai_turn(self) -> None:
        self.cancel_ai_job()
        if self.ai_paused or self.is_terminal():
            return
        if self.current_player().is_human():
            return
        self.ai_job = self.app.after(450, self._run_scheduled_ai_turn)

    def cancel_ai_job(self) -> None:
        if self.ai_job is not None:
            self.app.after_cancel(self.ai_job)
            self.ai_job = None
        self._ai_generation += 1  # invalidate any in-flight AI thread

    def current_side(self) -> int:
        return int(self.env.observation()["side_to_move"])

    def current_player(self):
        return self.players[self.current_side()]

    def is_terminal(self) -> bool:
        return self.env.observation()["game_result_name"] != "ongoing"

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
            if kind == "supervised":
                self.app.log(
                    self.app._tr("log_model_loaded", name=player.name,
                                 file=Path(player.checkpoint_path).name),
                    "info",
                )
            elif kind == "mcts":
                cfg = Path(player.config_path).name if player.config_path else "default"
                self.app.log(
                    self.app._tr("log_model_loaded_mcts", name=player.name,
                                 file=Path(player.checkpoint_path).name, cfg=cfg),
                    "info",
                )

    def _make_player(self, side: int, kind: str, checkpoint_path: Optional[str] = None,
                     mcts_config_path: Optional[str] = None):
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
            return SupervisedPlayer(self.app._tr("player_name", side=side_label, kind=kind_label), checkpoint_path)
        if kind == "mcts":
            if not checkpoint_path:
                raise ValueError(self.app._tr("log_need_mcts_ckpt", side=side_label))
            return MCTSPlayer(self.app._tr("player_name", side=side_label, kind=kind_label), checkpoint_path, mcts_config_path)
        return HumanPlayer(self.app._tr("player_name_human", side=side_label))

    def _select_start(
        self,
        square_id: int,
        moves_by_start: Dict[int, List[Move]],
    ) -> None:
        if square_id not in moves_by_start:
            self._clear_selection("log_no_legal_moves")
            return

        self.selected_square = square_id
        self.candidate_moves = moves_by_start[square_id]
        self.app.log(
            self.app._tr("log_square_selected", sq=square_id,
                         count=len(self.candidate_moves)),
            "info",
        )
        self.refresh()

    def _clear_selection(self, key: str, **fmt) -> None:
        self.selected_square = None
        self.candidate_moves = []
        self.app.log(self.app._tr(key, **fmt))
        self.refresh()
