from __future__ import annotations

import random
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_DIR = PROJECT_ROOT / "interface"
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))
TRAINER_DIR = PROJECT_ROOT / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from draughts_env import DraughtsEnv, Move  # noqa: E402
from agents import SupervisedAgent  # noqa: E402
from env import encode_observation  # noqa: E402


BOARD_SIZE = 10
PLAYABLE_SQUARES = 50
CELL_SIZE = 58
BOARD_PIXELS = BOARD_SIZE * CELL_SIZE

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
WHITE_MAN = 3
WHITE_KING = 4

BLACK_SIDE = 0
WHITE_SIDE = 1

SIDE_LABELS = {
    "black": "黑方",
    "white": "白方",
}

RESULT_LABELS = {
    "ongoing": "进行中",
    "black_win": "黑方胜",
    "white_win": "白方胜",
    "draw": "和棋",
}

PLAYER_KIND_LABELS = {
    "human": "人类",
    "random": "随机AI",
    "supervised": "监督模型AI",
}

PLAYER_LABEL_TO_KIND = {
    "人类": "human",
    "随机AI": "random",
    "监督模型AI": "supervised",
}

PLAYER_KIND_OPTIONS = ("人类", "随机AI", "监督模型AI")


@dataclass
class HumanPlayer:
    name: str
    kind: str = "human"

    def is_human(self) -> bool:
        return True


@dataclass
class RandomAgent:
    name: str
    rng: random.Random
    kind: str = "random"

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None) -> Optional[Move]:
        if not moves:
            return None
        return self.rng.choice(list(moves))


@dataclass
class SupervisedPlayer:
    name: str
    checkpoint_path: str
    kind: str = "supervised"

    def __post_init__(self) -> None:
        self.agent = SupervisedAgent(self.checkpoint_path, device="cpu")

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None) -> Optional[Move]:
        if not moves:
            return None
        if env is None:
            raise ValueError("监督模型AI需要当前棋局环境")
        action_id = self.agent.select_action(
            env.observation(),
            moves,
            EncodedObservationAdapter(env),
        )
        return find_move_by_action_id(moves, action_id)


class EncodedObservationAdapter:
    def __init__(self, env: DraughtsEnv) -> None:
        self.env = env

    def observation(self):
        return encode_observation(self.env.observation())


class BoardView(tk.Canvas):
    def __init__(self, master: tk.Widget, on_square_click) -> None:
        super().__init__(
            master,
            width=BOARD_PIXELS,
            height=BOARD_PIXELS,
            background="#d8dde3",
            highlightthickness=0,
        )
        self._on_square_click = on_square_click
        self.rotated = False
        self.bind("<Button-1>", self._handle_click)

    def _handle_click(self, event) -> None:
        view_row = event.y // CELL_SIZE
        view_col = event.x // CELL_SIZE
        if view_row < 0 or view_row >= BOARD_SIZE or view_col < 0 or view_col >= BOARD_SIZE:
            return
        row, col = self._board_coord_from_view(view_row, view_col)
        square_id = coord_to_square(row, col)
        if square_id is None:
            return
        self._on_square_click(square_id)

    def toggle_rotation(self) -> None:
        self.rotated = not self.rotated

    def draw(
        self,
        squares: Sequence[int],
        selected_square: Optional[int],
        landing_squares: Sequence[int],
        path_squares: Sequence[int],
    ) -> None:
        self.delete("all")
        landing_set = set(landing_squares)
        path_set = set(path_squares)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                view_row, view_col = self._view_coord_from_board(row, col)
                x0 = view_col * CELL_SIZE
                y0 = view_row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                playable = coord_to_square(row, col) is not None
                color = "#f3eadf" if not playable else "#49616c"
                self.create_rectangle(x0, y0, x1, y1, fill=color, outline="#26343b")

                square_id = coord_to_square(row, col)
                if square_id is not None:
                    if square_id in path_set:
                        self.create_rectangle(
                            x0 + 3,
                            y0 + 3,
                            x1 - 3,
                            y1 - 3,
                            outline="#f2c94c",
                            width=3,
                        )
                    if square_id == selected_square:
                        self.create_rectangle(
                            x0 + 4,
                            y0 + 4,
                            x1 - 4,
                            y1 - 4,
                            outline="#7bdff2",
                            width=4,
                        )
                    if square_id in landing_set:
                        self.create_oval(
                            x0 + 20,
                            y0 + 20,
                            x1 - 20,
                            y1 - 20,
                            fill="#8ee08e",
                            outline="#225522",
                            width=2,
                        )
                    self.create_text(
                        x0 + 8,
                        y0 + 8,
                        text=str(square_id),
                        anchor="nw",
                        fill="#d8e3e8",
                        font=("TkDefaultFont", 8),
                    )

        for square_id, piece in enumerate(squares, start=1):
            if piece != EMPTY:
                self._draw_piece(square_id, piece)

    def _draw_piece(self, square_id: int, piece: int) -> None:
        row, col = square_to_coord(square_id)
        view_row, view_col = self._view_coord_from_board(row, col)
        x0 = view_col * CELL_SIZE
        y0 = view_row * CELL_SIZE
        pad = 10
        fill = "#111820" if piece in (BLACK_MAN, BLACK_KING) else "#f8f4ea"
        outline = "#f8f4ea" if piece in (BLACK_MAN, BLACK_KING) else "#111820"
        text = "K" if piece in (BLACK_KING, WHITE_KING) else ""
        text_fill = "#f2c94c" if piece in (BLACK_KING, WHITE_KING) else outline

        self.create_oval(
            x0 + pad,
            y0 + pad,
            x0 + CELL_SIZE - pad,
            y0 + CELL_SIZE - pad,
            fill=fill,
            outline=outline,
            width=3,
        )
        if text:
            self.create_text(
                x0 + CELL_SIZE / 2,
                y0 + CELL_SIZE / 2,
                text=text,
                fill=text_fill,
                font=("TkDefaultFont", 18, "bold"),
            )

    def _view_coord_from_board(self, row: int, col: int) -> tuple[int, int]:
        if not self.rotated:
            return row, col
        return BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col

    def _board_coord_from_view(self, row: int, col: int) -> tuple[int, int]:
        return self._view_coord_from_board(row, col)


class GameController:
    def __init__(self, app: "DraughtsApp") -> None:
        self.app = app
        self.env = DraughtsEnv()
        self.rng = random.Random()
        self.players = {
            BLACK_SIDE: HumanPlayer("黑方玩家"),
            WHITE_SIDE: HumanPlayer("白方玩家"),
        }
        self.selected_square: Optional[int] = None
        self.candidate_moves: List[Move] = []
        self.last_move: Optional[Move] = None
        self.action_history: List[int] = []
        self.notation_history: List[str] = []
        self.turn_index = 0
        self.ai_paused = False
        self.ai_job: Optional[str] = None

    def start_new_game(
        self,
        black_kind: str,
        white_kind: str,
        checkpoint_paths: Optional[Dict[int, str]] = None,
    ) -> None:
        self.cancel_ai_job()
        checkpoint_paths = checkpoint_paths or {}
        self.players = {
            BLACK_SIDE: self._make_player(BLACK_SIDE, black_kind, checkpoint_paths.get(BLACK_SIDE)),
            WHITE_SIDE: self._make_player(WHITE_SIDE, white_kind, checkpoint_paths.get(WHITE_SIDE)),
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
        if clear_log:
            self.app.clear_log()
        self.app.log(f"新对局开始：{self.mode_label()}。", "info")
        self.refresh()
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
            self.app.log("AI 自动走棋已暂停。", "warn")
        else:
            self.app.log("AI 自动走棋已继续。", "info")
            self.schedule_ai_turn()

    def ai_step_once(self) -> None:
        self.cancel_ai_job()
        if self.is_terminal():
            self.app.log("对局已经结束，无法继续执行 AI。", "warn")
            return
        player = self.current_player()
        if player.is_human():
            self.app.log("当前行棋方是人类，不能执行 AI 走一步。", "warn")
            return
        self.perform_ai_move()

    def refresh(self) -> None:
        observation = self.env.observation()
        legal_moves = self.env.legal_moves()
        landing_squares = [move.path[-1] for move in self.candidate_moves]
        path_squares = self.last_move.path if self.last_move is not None else []
        self.app.update_board(
            observation,
            selected_square=self.selected_square,
            landing_squares=landing_squares,
            path_squares=path_squares,
        )
        self.app.update_status(observation, self.last_move)
        self.app.update_move_list(legal_moves, self.candidate_moves)
        self.app.update_notation(self.notation_history)

    def handle_square_click(self, square_id: int) -> None:
        observation = self.env.observation()
        if observation["game_result_name"] != "ongoing":
            self.app.log("对局已经结束。请点击“重新开始”。", "warn")
            return
        if not self.current_player().is_human():
            self.app.log("当前行棋方由 AI 控制。", "warn")
            return

        moves = self.env.legal_moves()
        moves_by_start = group_moves_by_start(moves)

        if self.selected_square is None:
            self._select_start(square_id, moves_by_start)
            return

        if square_id == self.selected_square:
            self._clear_selection("已取消选择。")
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
                f"有 {len(landing_matches)} 条路线都落在 {square_id} 号位，请在右侧动作列表中双击选择完整路线。",
                "warn",
            )
            self.refresh()
            return

        if square_id in moves_by_start:
            self._select_start(square_id, moves_by_start)
            return

        self._clear_selection("该位置不是当前棋子的合法落点。")

    def execute_move_by_action_id(self, action_id: int) -> None:
        if not self.current_player().is_human():
            self.app.log("当前行棋方由 AI 控制，请使用“AI走一步”或自动走棋。", "warn")
            return
        for move in self.env.legal_moves():
            if move.action_id == action_id:
                self.execute_move(move)
                return
        self.app.log(f"动作 {action_id} 已不再合法。", "warn")

    def execute_move(self, move: Move, actor: Optional[str] = None) -> None:
        try:
            observation, info = self.env.step(move.action_id)
        except RuntimeError as exc:
            self.app.log(f"执行失败：{exc}", "error")
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
            f"第 {self.turn_index} 手，{actor_label} 执行动作 {move.action_id}：{format_move(move)}",
            "action",
        )
        if info["done"]:
            self.app.log(
                f"对局结束：{localize_result(observation['game_result_name'])}。",
                "success",
            )
        self.refresh()
        self.schedule_ai_turn()

    def undo_last_move(self) -> None:
        self.cancel_ai_job()
        if not self.action_history:
            self.app.log("当前没有可悔的动作。", "warn")
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
            self.app.log(f"悔棋失败：{exc}", "error")
            self.env.reset()
            self.action_history = []
            self.notation_history = []
            self.last_move = None
            self.turn_index = 0
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
        self.app.log(f"已悔棋：{removed_notation}", "warn")
        self.refresh()
        self.schedule_ai_turn()

    def perform_ai_move(self) -> None:
        if self.is_terminal():
            return
        player = self.current_player()
        if player.is_human():
            return

        legal_moves = self.env.legal_moves()
        move = player.choose_move(legal_moves, self.env)
        if move is None:
            self.app.log(f"{player.name} 没有可执行动作。", "warn")
            return
        self.execute_move(move, actor=player.name)

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

    def current_side(self) -> int:
        return int(self.env.observation()["side_to_move"])

    def current_player(self):
        return self.players[self.current_side()]

    def is_terminal(self) -> bool:
        return self.env.observation()["game_result_name"] != "ongoing"

    def mode_label(self) -> str:
        return (
            f"黑方{PLAYER_KIND_LABELS[self.players[BLACK_SIDE].kind]} "
            f"vs 白方{PLAYER_KIND_LABELS[self.players[WHITE_SIDE].kind]}"
        )

    def _run_scheduled_ai_turn(self) -> None:
        self.ai_job = None
        if not self.ai_paused:
            self.perform_ai_move()

    def log_model_players(self) -> None:
        for side in (BLACK_SIDE, WHITE_SIDE):
            player = self.players[side]
            if getattr(player, "kind", None) == "supervised":
                self.app.log(
                    f"{player.name} 已加载模型：{Path(player.checkpoint_path).name}",
                    "info",
                )

    def _make_player(self, side: int, kind: str, checkpoint_path: Optional[str] = None):
        side_label = "黑方" if side == BLACK_SIDE else "白方"
        if kind == "random":
            return RandomAgent(f"{side_label}随机AI", self.rng)
        if kind == "supervised":
            if not checkpoint_path:
                raise ValueError(f"{side_label}监督模型AI需要选择 checkpoint 文件")
            return SupervisedPlayer(f"{side_label}监督模型AI", checkpoint_path)
        return HumanPlayer(f"{side_label}玩家")

    def _select_start(
        self,
        square_id: int,
        moves_by_start: Dict[int, List[Move]],
    ) -> None:
        if square_id not in moves_by_start:
            self._clear_selection("该棋子当前没有合法动作。")
            return

        self.selected_square = square_id
        self.candidate_moves = moves_by_start[square_id]
        self.app.log(
            f"已选择 {square_id} 号位，共 {len(self.candidate_moves)} 条合法路线。",
            "info",
        )
        self.refresh()

    def _clear_selection(self, message: str) -> None:
        self.selected_square = None
        self.candidate_moves = []
        self.app.log(message)
        self.refresh()


class DraughtsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BitMind 国际跳棋 - GUI v0")
        self.minsize(1160, 690)

        self.status_vars = {
            "mode": tk.StringVar(value="-"),
            "side": tk.StringVar(value="-"),
            "moves": tk.StringVar(value="-"),
            "result": tk.StringVar(value="-"),
            "last": tk.StringVar(value="-"),
        }
        self.pause_button: Optional[ttk.Button] = None
        self.rotate_button: Optional[ttk.Button] = None

        self._build_layout()
        self.controller = GameController(self)
        self.after(0, self.controller.open_new_game_dialog)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)

        notation_frame = ttk.LabelFrame(self, text="棋谱", padding=(8, 8, 8, 8))
        notation_frame.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=12)
        notation_frame.columnconfigure(0, weight=1)
        notation_frame.rowconfigure(0, weight=1)
        self.notation_list = tk.Listbox(notation_frame, width=24, height=30)
        self.notation_list.grid(row=0, column=0, sticky="nsew")
        notation_scrollbar = ttk.Scrollbar(
            notation_frame,
            orient="vertical",
            command=self.notation_list.yview,
        )
        notation_scrollbar.grid(row=0, column=1, sticky="ns")
        self.notation_list.configure(yscrollcommand=notation_scrollbar.set)

        self.board = BoardView(self, self._on_square_click)
        self.board.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

        sidebar = ttk.Frame(self, padding=(0, 12, 12, 12))
        sidebar.grid(row=0, column=2, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(4, weight=3)
        sidebar.rowconfigure(6, weight=2)

        status_frame = ttk.LabelFrame(sidebar, text="对局状态")
        status_frame.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        self._status_row(status_frame, 0, "模式", self.status_vars["mode"])
        self._status_row(status_frame, 1, "行棋方", self.status_vars["side"])
        self._status_row(status_frame, 2, "合法动作", self.status_vars["moves"])
        self._status_row(status_frame, 3, "局面结果", self.status_vars["result"])
        self._status_row(status_frame, 4, "最近动作", self.status_vars["last"])

        toolbar = ttk.Frame(sidebar)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        ttk.Button(toolbar, text="新建对局", command=self._on_new_game).pack(side="left")
        ttk.Button(toolbar, text="重开本局", command=self._on_restart).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="悔棋", command=self._on_undo).pack(side="left", padx=(8, 0))
        self.rotate_button = ttk.Button(toolbar, text="旋转棋盘", command=self._on_rotate_board)
        self.rotate_button.pack(side="left", padx=(8, 0))

        ai_toolbar = ttk.Frame(sidebar)
        ai_toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.pause_button = ttk.Button(
            ai_toolbar,
            text="暂停AI",
            command=self._on_toggle_pause,
        )
        self.pause_button.pack(side="left")
        ttk.Button(ai_toolbar, text="AI走一步", command=self._on_ai_step_once).pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Label(sidebar, text="合法动作").grid(row=3, column=0, sticky="w")
        self.move_tree = ttk.Treeview(
            sidebar,
            columns=("action", "path", "captures"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        self.move_tree.heading("action", text="编号")
        self.move_tree.heading("path", text="路径")
        self.move_tree.heading("captures", text="吃子")
        self.move_tree.column("action", width=68, stretch=False, anchor="center")
        self.move_tree.column("path", width=170, stretch=True)
        self.move_tree.column("captures", width=110, stretch=True)
        self.move_tree.grid(row=4, column=0, sticky="nsew")
        self.move_tree.bind("<Double-1>", self._on_move_double_click)

        ttk.Label(sidebar, text="日志").grid(row=5, column=0, sticky="w", pady=(10, 0))
        log_frame = ttk.Frame(sidebar)
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.tag_configure("info", foreground="#1f4e79")
        self.log_text.tag_configure("action", foreground="#1b6e3c")
        self.log_text.tag_configure("warn", foreground="#9a5b00")
        self.log_text.tag_configure("error", foreground="#b00020")
        self.log_text.tag_configure("success", foreground="#275d38")

    def _status_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=6,
            pady=3,
        )

    def update_board(
        self,
        observation: Dict[str, object],
        selected_square: Optional[int],
        landing_squares: Sequence[int],
        path_squares: Sequence[int],
    ) -> None:
        self.board.draw(
            observation["squares"],
            selected_square,
            landing_squares,
            path_squares,
        )

    def update_status(self, observation: Dict[str, object], last_move: Optional[Move]) -> None:
        self.status_vars["mode"].set(self.controller.mode_label())
        self.status_vars["side"].set(localize_side(observation["side_to_move_name"]))
        self.status_vars["moves"].set(str(observation["legal_move_count"]))
        self.status_vars["result"].set(localize_result(observation["game_result_name"]))
        self.status_vars["last"].set(format_move(last_move) if last_move else "-")

    def update_move_list(
        self,
        legal_moves: Sequence[Move],
        candidate_moves: Sequence[Move],
    ) -> None:
        for item in self.move_tree.get_children():
            self.move_tree.delete(item)

        candidate_ids = {move.action_id for move in candidate_moves}
        for move in legal_moves:
            tags = ("candidate",) if move.action_id in candidate_ids else ()
            self.move_tree.insert(
                "",
                "end",
                iid=str(move.action_id),
                values=(
                    move.action_id,
                    " -> ".join(str(square) for square in move.path),
                    ", ".join(str(square) for square in move.captures) or "-",
                ),
                tags=tags,
            )
        self.move_tree.tag_configure("candidate", background="#e8f4ea")

    def update_notation(self, notation: Sequence[str]) -> None:
        self.notation_list.delete(0, "end")
        for item in notation:
            self.notation_list.insert("end", item)
        if notation:
            self.notation_list.see("end")

    def log(self, message: str, level: str = "info") -> None:
        self.log_text.configure(state="normal")
        if level not in {"info", "action", "warn", "error", "success"}:
            level = "info"
        self.log_text.insert("end", message + "\n", level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def update_pause_button(self, paused: bool) -> None:
        if self.pause_button is not None:
            self.pause_button.configure(text="继续AI" if paused else "暂停AI")

    def show_mode_dialog(self, black_kind: str, white_kind: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("新建对局")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        black_var = tk.StringVar(value=PLAYER_KIND_LABELS.get(black_kind, "人类"))
        white_var = tk.StringVar(value=PLAYER_KIND_LABELS.get(white_kind, "人类"))

        frame = ttk.Frame(dialog, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="选择双方决策源").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 10),
        )

        ttk.Label(frame, text="黑方").grid(row=1, column=0, sticky="w", pady=4)
        black_combo = ttk.Combobox(
            frame,
            textvariable=black_var,
            values=PLAYER_KIND_OPTIONS,
            state="readonly",
            width=14,
        )
        black_combo.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="白方").grid(row=2, column=0, sticky="w", pady=4)
        white_combo = ttk.Combobox(
            frame,
            textvariable=white_var,
            values=PLAYER_KIND_OPTIONS,
            state="readonly",
            width=14,
        )
        white_combo.grid(row=2, column=1, sticky="ew", pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def start_game() -> None:
            black_selected = PLAYER_LABEL_TO_KIND[black_var.get()]
            white_selected = PLAYER_LABEL_TO_KIND[white_var.get()]
            checkpoint_paths: Dict[int, str] = {}
            if black_selected == "supervised":
                path = self.ask_checkpoint_path("请选择黑方监督模型 checkpoint", dialog)
                if path is None:
                    return
                checkpoint_paths[BLACK_SIDE] = path
            if white_selected == "supervised":
                path = self.ask_checkpoint_path("请选择白方监督模型 checkpoint", dialog)
                if path is None:
                    return
                checkpoint_paths[WHITE_SIDE] = path
            try:
                self.controller.start_new_game(black_selected, white_selected, checkpoint_paths)
            except Exception as exc:
                messagebox.showerror("创建对局失败", str(exc), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(buttons, text="开始对局", command=start_game).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", start_game)
        dialog.bind("<Return>", lambda _event: start_game())
        black_combo.focus_set()
        self.wait_window(dialog)

    def ask_checkpoint_path(self, title: str, parent: tk.Toplevel) -> Optional[str]:
        checkpoint_dir = PROJECT_ROOT / "trainer" / "checkpoints"
        initial_dir = checkpoint_dir if checkpoint_dir.exists() else PROJECT_ROOT
        path = filedialog.askopenfilename(
            parent=parent,
            title=title,
            initialdir=str(initial_dir),
            filetypes=(
                ("PyTorch checkpoint", "*.pt *.pth"),
                ("所有文件", "*"),
            ),
        )
        if not path:
            messagebox.showwarning("未选择模型", "请选择 checkpoint 文件后再开始对局。", parent=parent)
            return None
        return path

    def _on_square_click(self, square_id: int) -> None:
        self.controller.handle_square_click(square_id)

    def _on_new_game(self) -> None:
        self.controller.open_new_game_dialog()

    def _on_restart(self) -> None:
        self.controller.reset_current_game(clear_log=True)

    def _on_undo(self) -> None:
        self.controller.undo_last_move()

    def _on_rotate_board(self) -> None:
        self.board.toggle_rotation()
        self.controller.refresh()
        self.log("棋盘视角已旋转 180 度。", "info")

    def _on_toggle_pause(self) -> None:
        self.controller.toggle_pause()

    def _on_ai_step_once(self) -> None:
        self.controller.ai_step_once()

    def _on_move_double_click(self, _event) -> None:
        selected = self.move_tree.selection()
        if not selected:
            return
        self.controller.execute_move_by_action_id(int(selected[0]))


def square_to_coord(square_id: int) -> tuple[int, int]:
    zero_based = square_id - 1
    row = zero_based // 5
    offset = zero_based % 5
    col = offset * 2 + (1 if row % 2 == 0 else 0)
    return row, col


def coord_to_square(row: int, col: int) -> Optional[int]:
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        return None
    if (row + col) % 2 == 0:
        return None
    return row * 5 + (col // 2) + 1


def group_moves_by_start(moves: Sequence[Move]) -> Dict[int, List[Move]]:
    grouped: Dict[int, List[Move]] = {}
    for move in moves:
        grouped.setdefault(move.path[0], []).append(move)
    return grouped


def find_move_by_action_id(moves: Sequence[Move], action_id: int) -> Optional[Move]:
    for move in moves:
        if move.action_id == action_id:
            return move
    return None


def opponent_side(side: int) -> int:
    return WHITE_SIDE if side == BLACK_SIDE else BLACK_SIDE


def format_move(move: Optional[Move]) -> str:
    if move is None:
        return "-"
    separator = "x" if move.is_capture else "-"
    path = separator.join(str(square) for square in move.path)
    if move.captures:
        return f"{move.action_id}: {path} ({','.join(str(square) for square in move.captures)})"
    return f"{move.action_id}: {path}"


def make_notation(turn_index: int, move: Move) -> str:
    return f"{turn_index}. {format_move(move)}"


def localize_side(side_name: object) -> str:
    return SIDE_LABELS.get(str(side_name), str(side_name))


def localize_result(result_name: object) -> str:
    return RESULT_LABELS.get(str(result_name), str(result_name))


def main() -> None:
    app = DraughtsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
