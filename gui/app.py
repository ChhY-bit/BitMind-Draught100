from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Path setup — ensure interface/ and project root are on sys.path before
# any submodules try to import draughts_env or draughts.*.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INTERFACE_DIR = _PROJECT_ROOT / "interface"
if str(_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from draughts_env import DraughtsEnv, Move  # noqa: E402

from .fonts import (  # noqa: E402
    _cjk_renders_correctly,
    _discover_system_cjk_ttfs,
    _FALLBACK_CJK_FONTS,
    _get_tk_font_families,
    _install_fontconfig,
    _install_x11_font,
    _PREFERRED_CJK_FONTS,
    _unregister_fonts,
    _get_bundled_ttf_path,
)
from .game_controller import GameController  # noqa: E402
from .i18n import _tr_static  # noqa: E402
from .protocol import PlayerAnalysis  # noqa: E402
from .utils import (  # noqa: E402
    BLACK_SIDE,
    WHITE_SIDE,
    format_move,
    opponent_side,
    value_to_win_pct,
)
from .widgets import BoardView, EvalBar, HardwareMonitor, ValueCurve  # noqa: E402


class DraughtsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.lang: str = "zh"
        self._i18n_widgets: list = []  # widgets to reconfigure on lang switch
        self._side_moves: dict[int, list[Move]] = {}  # per-side stored move lists
        self._configure_fonts()
        self.title(self._tr("window_title"))
        self.minsize(1520, 780)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_vars = {
            "mode": tk.StringVar(value="-"),
            "side": tk.StringVar(value="-"),
            "moves": tk.StringVar(value="-"),
            "result": tk.StringVar(value="-"),
            "last": tk.StringVar(value="-"),
        }
        self.pause_button: Optional[tk.Button] = None
        self.rotate_button: Optional[ttk.Button] = None

        self._build_layout()
        self.controller = GameController(self)
        self.after(0, self.controller.open_new_game_dialog)

    def _tr(self, key: str, **fmt) -> str:
        """Translate *key* into the current UI language.

        Extra keyword arguments are passed to ``str.format()`` on the
        translated string.  Missing keys fall back to the key name itself
        so the UI never crashes on a missing translation.
        """
        text = _tr_static(key, self.lang)
        if fmt:
            try:
                return text.format(**fmt)
            except Exception:
                return text  # best-effort: show raw template on format error
        return text

    def _on_close(self) -> None:
        """Clean up resources before destroying the window."""
        self.controller.cancel_ai_job()
        self.destroy()
        _unregister_fonts()

    def _configure_fonts(self) -> None:
        """Select and apply a CJK-capable font family across all widgets.

        Strategy (platform-aware, zero-config):
        1. Scan Tk's current font list for a modern CJK family.
        2. On Linux: install the bundled TTF into **both** fontconfig
           (for Xft-capable Tk) **and** the X11 core path (for noXft Tk).
           Re-scan and re-test after each installation.
        3. Fall back to X11 core CJK bitmaps.
        4. Verify the selected font actually renders CJK glyphs.
        """
        families = _get_tk_font_families(self)

        cjk_family: Optional[str] = None

        # ── Tier 1: check for an already-available modern CJK font ──
        for name in _PREFERRED_CJK_FONTS:
            if name in families:
                cjk_family = name
                break

        # ── Tier 2 (Linux): install bundled TTF via fontconfig ──
        if cjk_family is None and sys.platform.startswith('linux'):
            bundled_ttf = _get_bundled_ttf_path()
            if bundled_ttf is not None:
                _install_fontconfig(bundled_ttf)
                self.update()
                families = _get_tk_font_families(self)
                for name in _PREFERRED_CJK_FONTS:
                    if name in families and _cjk_renders_correctly(name, self):
                        cjk_family = name
                        break

        # ── Tier 3 (Linux): install system CJK fonts via X11 font path ──
        if cjk_family is None and sys.platform.startswith('linux'):
            for sys_font_path in _discover_system_cjk_ttfs():
                _install_x11_font(sys_font_path)
                self.update()
                families = _get_tk_font_families(self)
                for name in _PREFERRED_CJK_FONTS:
                    if name in families and _cjk_renders_correctly(name, self):
                        cjk_family = name
                        break
                if cjk_family is not None:
                    break

        # ── Tier 4 (Linux): install bundled TTF via X11 font path ──
        if cjk_family is None and sys.platform.startswith('linux'):
            bundled_ttf = _get_bundled_ttf_path()
            if bundled_ttf is not None:
                _install_x11_font(bundled_ttf)
                self.update()
                families = _get_tk_font_families(self)
                for name in _PREFERRED_CJK_FONTS:
                    if name in families and _cjk_renders_correctly(name, self):
                        cjk_family = name
                        break

        # ── Tier 5: X11 core CJK bitmaps (limited encoding!) ──
        if cjk_family is None:
            for name in _FALLBACK_CJK_FONTS:
                if name in families and _cjk_renders_correctly(name, self):
                    cjk_family = name
                    break

        # ── Absolute fallback ──
        if cjk_family is None:
            cjk_family = 'song ti'  # last resort — may render poorly

        # Apply the chosen family to all Tk built-in named fonts
        for name in ('TkDefaultFont', 'TkTextFont', 'TkHeadingFont',
                     'TkFixedFont', 'TkMenuFont'):
            try:
                font.nametofont(name).configure(family=cjk_family, size=12)
            except Exception:
                pass

        # ── ttk themed widgets use their own style engine ──
        _ttk_style = ttk.Style()
        _ttk_style.configure("TLabel", font=(cjk_family, 12))
        _ttk_style.configure("TButton", font=(cjk_family, 12))
        _ttk_style.configure("TCheckbutton", font=(cjk_family, 12))
        _ttk_style.configure("TCombobox", font=(cjk_family, 12))
        _ttk_style.configure("TEntry", font=(cjk_family, 12))
        _ttk_style.configure("Treeview", font=(cjk_family, 12))
        _ttk_style.configure("Treeview.Heading", font=(cjk_family, 12, "bold"))
        _ttk_style.configure("TLabelframe", font=(cjk_family, 12))
        _ttk_style.configure("TLabelframe.Label", font=(cjk_family, 12, "bold"))
        # Active-side move-frame style — gold-tinted label + light background
        _ttk_style.configure("Active.TLabelframe", font=(cjk_family, 12))
        _ttk_style.configure(
            "Active.TLabelframe.Label",
            font=(cjk_family, 12, "bold"),
            foreground="#c9880c",
        )

        # Store so child widgets (ValueCurve, etc.) can construct font specs
        self._cjk_family: str = cjk_family

    def _build_layout(self) -> None:
        tr = self._tr  # local alias for brevity
        self.columnconfigure(0, weight=0)  # notation + log (left panel)
        self.columnconfigure(1, weight=0)  # board + analysis
        self.columnconfigure(2, weight=0)  # (unused — spanned by col 1)
        self.columnconfigure(3, weight=1)  # sidebar
        self.rowconfigure(0, weight=1)   # board
        self.rowconfigure(1, weight=1)   # search tree

        # ---- Left panel (col 0, spans both rows) ----
        left_panel = ttk.Frame(self)
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew",
                        padx=(12, 0), pady=(12, 12))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)  # notation (50%)
        left_panel.rowconfigure(1, weight=1)  # log (50%)

        # Notation (upper half of left panel)
        notation_frame = ttk.LabelFrame(left_panel, text=tr("notation_frame"), padding=(8, 8, 8, 8))
        notation_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        notation_frame.columnconfigure(0, weight=1)
        notation_frame.rowconfigure(0, weight=1)
        self.notation_list = tk.Listbox(notation_frame, width=56)
        self.notation_list.grid(row=0, column=0, sticky="nsew")
        notation_scrollbar = ttk.Scrollbar(
            notation_frame,
            orient="vertical",
            command=self.notation_list.yview,
        )
        notation_scrollbar.grid(row=0, column=1, sticky="ns")
        self.notation_list.configure(yscrollcommand=notation_scrollbar.set)
        self._i18n_widgets.append(("labelframe", notation_frame, "notation_frame"))

        # Log (lower half of left panel)
        log_frame = ttk.LabelFrame(left_panel, text=tr("section_log"), padding=(4, 4, 4, 4))
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, width=56, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.tag_configure("info", foreground="#1f4e79")
        self.log_text.tag_configure("action", foreground="#1b6e3c")
        self.log_text.tag_configure("warn", foreground="#9a5b00")
        self.log_text.tag_configure("error", foreground="#b00020")
        self.log_text.tag_configure("success", foreground="#275d38")
        self._i18n_widgets.append(("labelframe", log_frame, "section_log"))

        # ---- Board column: eval bar + board, then analysis below ----
        board_column = ttk.Frame(self)
        board_column.grid(row=0, column=1, columnspan=2, sticky="nw", padx=8, pady=12)
        board_column.columnconfigure(0, weight=1)
        board_column.rowconfigure(0, weight=0)   # eval bar + board
        board_column.rowconfigure(1, weight=0)   # engine analysis

        # Row 0: eval bar + board side by side
        board_row = ttk.Frame(board_column)
        board_row.grid(row=0, column=0, sticky="nw")
        self.eval_bar = EvalBar(board_row)
        self.eval_bar.pack(side="left", fill="y")
        self.board = BoardView(board_row, self._on_square_click)
        self.board.pack(side="left")

        # Row 1: engine analysis (below the board)
        self.search_stats_frame = ttk.LabelFrame(board_column, text=tr("engine_frame"), padding=(6, 6, 6, 6))
        self.search_stats_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.search_stats_frame.columnconfigure(0, weight=1)
        self.search_stats_frame.columnconfigure(1, weight=1)
        self.search_stats_frame.columnconfigure(2, weight=1)
        self.search_stats_frame.columnconfigure(3, weight=1)
        self.search_stats_vars = {
            "eval": tk.StringVar(value=tr("analysis_na")),
            "sims": tk.StringVar(value=tr("analysis_na")),
            "best": tk.StringVar(value=tr("analysis_na")),
            "delta": tk.StringVar(value=tr("analysis_na")),
        }
        # Row 0: column headers
        lbl_val = ttk.Label(self.search_stats_frame, text=tr("analysis_value"))
        lbl_val.grid(row=0, column=0, sticky="w")
        self._i18n_widgets.append(("label", lbl_val, "analysis_value"))
        lbl_srch = ttk.Label(self.search_stats_frame, text=tr("analysis_search"))
        lbl_srch.grid(row=0, column=1, sticky="w")
        self._i18n_widgets.append(("label", lbl_srch, "analysis_search"))
        lbl_best = ttk.Label(self.search_stats_frame, text=tr("analysis_best"))
        lbl_best.grid(row=0, column=2, sticky="w")
        self._i18n_widgets.append(("label", lbl_best, "analysis_best"))
        lbl_delta = ttk.Label(self.search_stats_frame, text=tr("analysis_delta"))
        lbl_delta.grid(row=0, column=3, sticky="w")
        self._i18n_widgets.append(("label", lbl_delta, "analysis_delta"))
        # Row 1: values
        ttk.Label(self.search_stats_frame, textvariable=self.search_stats_vars["eval"]).grid(
            row=1, column=0, sticky="w", padx=(0, 12))
        ttk.Label(self.search_stats_frame, textvariable=self.search_stats_vars["sims"]).grid(
            row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Label(self.search_stats_frame, textvariable=self.search_stats_vars["best"]).grid(
            row=1, column=2, sticky="w", padx=(0, 12))
        ttk.Label(self.search_stats_frame, textvariable=self.search_stats_vars["delta"]).grid(
            row=1, column=3, sticky="w")
        self._i18n_widgets.append(("labelframe", self.search_stats_frame, "engine_frame"))

        # ---- System monitor (below board, row 1) ----
        mon_frame = ttk.LabelFrame(self, text=tr("mon_frame"), padding=(4, 4, 4, 4))
        mon_frame.grid(row=1, column=1, columnspan=2, sticky="new",
                       padx=8, pady=(0, 12))
        mon_frame.columnconfigure(0, weight=1)
        mon_frame.rowconfigure(0, weight=1)
        self.hw_monitor = HardwareMonitor(mon_frame, draughts_app=self, height=120)
        self.hw_monitor.pack(fill="both", expand=True)
        self._i18n_widgets.append(("labelframe", mon_frame, "mon_frame"))

        # ---- Sidebar (right) ----
        sidebar = ttk.Frame(self, padding=(0, 12, 12, 12))
        sidebar.grid(row=0, column=3, rowspan=2, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=2)  # move tree
        sidebar.rowconfigure(4, weight=1)  # curve

        # Status
        status_frame = ttk.LabelFrame(sidebar, text=tr("status_frame"))
        status_frame.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        self._status_row(status_frame, 0, "status_mode", self.status_vars["mode"])
        self._status_row(status_frame, 1, "status_side", self.status_vars["side"])
        self._status_row(status_frame, 2, "status_moves", self.status_vars["moves"])
        self._status_row(status_frame, 3, "status_result", self.status_vars["result"])
        self._status_row(status_frame, 4, "status_last", self.status_vars["last"])
        self._i18n_widgets.append(("labelframe", status_frame, "status_frame"))

        # Toolbar
        toolbar = ttk.Frame(sidebar)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        btn_new = ttk.Button(toolbar, text=tr("btn_new_game"), command=self._on_new_game)
        btn_new.pack(side="left")
        btn_restart = ttk.Button(toolbar, text=tr("btn_restart"), command=self._on_restart)
        btn_restart.pack(side="left", padx=(8, 0))
        btn_undo = ttk.Button(toolbar, text=tr("btn_undo"), command=self._on_undo)
        btn_undo.pack(side="left", padx=(8, 0))
        self.rotate_button = ttk.Button(toolbar, text=tr("btn_rotate"), command=self._on_rotate_board)
        self.rotate_button.pack(side="left", padx=(8, 0))
        self._i18n_widgets.extend([
            ("button", btn_new, "btn_new_game"),
            ("button", btn_restart, "btn_restart"),
            ("button", btn_undo, "btn_undo"),
            ("button", self.rotate_button, "btn_rotate"),
        ])

        # AI controls
        ai_toolbar = ttk.Frame(sidebar)
        ai_toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.pause_button = tk.Button(
            ai_toolbar,
            text=tr("btn_pause"),
            font="TkDefaultFont",
            bg="#27ae60", fg="white",
            activebackground="#219a52", activeforeground="white",
            relief="flat", padx=10, pady=2,
            command=self._on_toggle_pause,
        )
        self.pause_button.pack(side="left")
        btn_ai_step = ttk.Button(ai_toolbar, text=tr("btn_ai_step"), command=self._on_ai_step_once)
        btn_ai_step.pack(side="left", padx=(8, 0))
        self._i18n_widgets.append(("button", btn_ai_step, "btn_ai_step"))

        # Move list — two independent columns, one per side.
        moves_frame = ttk.Frame(sidebar)
        moves_frame.grid(row=3, column=0, sticky="nsew")
        moves_frame.columnconfigure(0, weight=1)
        moves_frame.columnconfigure(1, weight=1)
        moves_frame.rowconfigure(0, weight=1)

        # --- Black column ---
        self._black_wrapper = tk.Frame(moves_frame, bg="#e0e4e8",
                                       highlightthickness=2,
                                       highlightbackground="#e0e4e8")
        self._black_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._black_wrapper.columnconfigure(0, weight=1)
        self._black_wrapper.rowconfigure(0, weight=1)
        self._black_move_frame = ttk.LabelFrame(self._black_wrapper, text=tr("side_black"), padding=(4, 4, 4, 4))
        self._black_move_frame.grid(row=0, column=0, sticky="nsew")
        self._black_move_frame.columnconfigure(0, weight=1)
        self._black_move_frame.rowconfigure(0, weight=1)
        self._black_move_tree = self._make_move_treeview(self._black_move_frame)
        self._black_move_tree.grid(row=0, column=0, sticky="nsew")

        # --- White column ---
        self._white_wrapper = tk.Frame(moves_frame, bg="#e0e4e8",
                                       highlightthickness=2,
                                       highlightbackground="#e0e4e8")
        self._white_wrapper.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._white_wrapper.columnconfigure(0, weight=1)
        self._white_wrapper.rowconfigure(0, weight=1)
        self._white_move_frame = ttk.LabelFrame(self._white_wrapper, text=tr("side_white"), padding=(4, 4, 4, 4))
        self._white_move_frame.grid(row=0, column=0, sticky="nsew")
        self._white_move_frame.columnconfigure(0, weight=1)
        self._white_move_frame.rowconfigure(0, weight=1)
        self._white_move_tree = self._make_move_treeview(self._white_move_frame)
        self._white_move_tree.grid(row=0, column=0, sticky="nsew")

        self._i18n_widgets.extend([
            ("labelframe", self._black_move_frame, "side_black"),
            ("labelframe", self._white_move_frame, "side_white"),
        ])

        # Value curve (in sidebar, below move list)
        curve_frame = ttk.LabelFrame(sidebar, text=tr("curve_frame"), padding=(4, 4, 4, 4))
        curve_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        curve_frame.columnconfigure(0, weight=1)
        self.curve = ValueCurve(curve_frame, on_point_click=self._on_curve_point_click, app=self)
        self.curve.pack(fill="x", expand=False)
        self._i18n_widgets.append(("labelframe", curve_frame, "curve_frame"))

        # ---- Language toggle (in sidebar toolbar) ----
        self._lang_button = ttk.Button(toolbar, text=tr("lang_toggle"), command=self._on_toggle_language)
        self._lang_button.pack(side="right", padx=(8, 0))

    def _status_row(
        self,
        parent: ttk.Frame,
        row: int,
        label_key: str,
        variable: tk.StringVar,
    ) -> None:
        lbl = ttk.Label(parent, text=self._tr(label_key))
        lbl.grid(row=row, column=0, sticky="w", padx=6, pady=3)
        self._i18n_widgets.append(("label", lbl, label_key))
        ttk.Label(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=6,
            pady=3,
        )

    def _make_move_treeview(self, parent: ttk.Frame) -> ttk.Treeview:
        """Create a standard move-list treeview with 6 columns."""
        tv = ttk.Treeview(
            parent,
            columns=("action", "prob", "qval", "visits", "path", "captures"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        tv.heading("action", text=self._tr("tree_action"))
        tv.heading("prob", text=self._tr("tree_prob"))
        tv.heading("qval", text=self._tr("tree_qval"))
        tv.heading("visits", text=self._tr("tree_visits"))
        tv.heading("path", text=self._tr("tree_path"))
        tv.heading("captures", text=self._tr("tree_captures"))
        tv.column("action", width=35, stretch=False, anchor="center")
        tv.column("prob", width=54, stretch=False, anchor="center")
        tv.column("qval", width=46, stretch=False, anchor="center")
        tv.column("visits", width=40, stretch=False, anchor="center")
        tv.column("path", width=70, stretch=True)
        tv.column("captures", width=55, stretch=True)
        tv.tag_configure("candidate", background="#e8f4ea")
        tv.tag_configure("top_pick", background="#fff3cd")
        tv.tag_configure("q_pos", foreground="#1a7a2e")
        tv.tag_configure("q_neg", foreground="#c0392b")
        return tv

    def _highlight_active_side(self, side: int) -> None:
        """Highlight the active side's entire move column with a gold border."""
        ACTIVE_BORDER = "#f2c94c"
        INACTIVE_BORDER = "#e0e4e8"
        for s, wrapper, frame in (
            (BLACK_SIDE, self._black_wrapper, self._black_move_frame),
            (WHITE_SIDE, self._white_wrapper, self._white_move_frame),
        ):
            try:
                is_active = s == side
                wrapper.configure(
                    highlightbackground=ACTIVE_BORDER if is_active else INACTIVE_BORDER,
                )
                frame.configure(
                    style="Active.TLabelframe" if is_active else "TLabelframe",
                )
            except Exception:
                pass

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

    def update_eval_bar(self, analysis: PlayerAnalysis, side_to_move: int) -> None:
        if analysis.value is not None:
            self.eval_bar.set_value(analysis.value, side_to_move)
        else:
            self.eval_bar.clear()

    def update_search_stats(self, analysis: PlayerAnalysis, side_to_move: int) -> None:
        side_short = self._tr("side_black_short" if side_to_move == BLACK_SIDE else "side_white_short")
        if analysis.value is not None:
            v = analysis.value
            win_pct = value_to_win_pct(v, side_to_move)
            self.search_stats_vars["eval"].set(
                self._tr("analysis_eval", v=v, side=side_short, pct=win_pct)
            )
        else:
            self.search_stats_vars["eval"].set(self._tr("analysis_na"))

        info = analysis.search_info or {}
        if "simulations" in info:
            c_puct = info.get("c_puct", "—")
            self.search_stats_vars["sims"].set(
                self._tr("analysis_sims", sims=info["simulations"], c_puct=c_puct)
            )
        elif "depth" in info:
            self.search_stats_vars["sims"].set(self._tr("analysis_depth", depth=info["depth"]))
        else:
            self.search_stats_vars["sims"].set(self._tr("analysis_na"))

        if analysis.policy_probs:
            best_aid = max(analysis.policy_probs, key=lambda k: analysis.policy_probs[k])  # type: ignore[arg-type]
            best_pct = analysis.policy_probs[best_aid] * 100
            self.search_stats_vars["best"].set(
                self._tr("analysis_best_fmt", aid=best_aid, pct=best_pct))
        else:
            self.search_stats_vars["best"].set(self._tr("analysis_na"))

        # Search delta: network value → MCTS-corrected value
        if analysis.search_value is not None and analysis.value is not None:
            self.search_stats_vars["delta"].set(
                self._tr("analysis_delta_fmt",
                         netv=analysis.value, searchv=analysis.search_value))
        else:
            self.search_stats_vars["delta"].set(self._tr("analysis_na"))

    def update_status(self, observation: Dict[str, object], last_move: Optional[Move]) -> None:
        self.status_vars["mode"].set(self.controller.mode_label())
        self.status_vars["side"].set(self._tr(f"side_{observation['side_to_move_name']}"))
        self.status_vars["moves"].set(str(observation["legal_move_count"]))
        self.status_vars["result"].set(self._tr(f"result_{observation['game_result_name']}"))
        self.status_vars["last"].set(format_move(last_move) if last_move else "-")

    def update_side_move_list(
        self,
        side: int,
        legal_moves: Sequence[Move],
        candidate_moves: Sequence[Move],
        analysis: Optional[PlayerAnalysis] = None,
    ) -> None:
        """Populate *side*'s move treeview with *legal_moves*.

        If *analysis* carries ``policy_probs`` they are displayed
        directly (the caller is responsible for only passing an
        analysis that belongs to this position).
        """
        tree = self._black_move_tree if side == BLACK_SIDE else self._white_move_tree
        for item in tree.get_children():
            tree.delete(item)

        # Re-apply tag styling in case the treeview was emptied
        tree.tag_configure("candidate", background="#e8f4ea")
        tree.tag_configure("top_pick", background="#fff3cd")

        probs: Optional[dict[int, float]] = None
        visits: Optional[dict[int, int]] = None
        qvals: Optional[dict[int, float]] = None
        if analysis is not None:
            if analysis.policy_probs is not None:
                probs = analysis.policy_probs
            if analysis.visit_counts is not None:
                visits = analysis.visit_counts
            if analysis.q_values is not None:
                qvals = analysis.q_values

        candidate_ids = {move.action_id for move in candidate_moves}

        top_prob_aid: Optional[int] = None
        if probs and legal_moves:
            top_prob_aid = max(probs, key=lambda k: probs[k])  # type: ignore[arg-type]

        for move in legal_moves:
            aid = move.action_id
            tags = []
            if aid in candidate_ids:
                tags.append("candidate")
            if probs and aid == top_prob_aid:
                tags.append("top_pick")

            prob_str = ""
            if probs and aid in probs:
                pct = probs[aid] * 100
                if pct >= 10.0:
                    prob_str = f"{pct:.1f}%"
                elif pct >= 1.0:
                    prob_str = f"{pct:.2f}%"
                else:
                    prob_str = f"{pct:.3f}%"

            # Q-value string
            q_str = ""
            if qvals and aid in qvals:
                q = qvals[aid]
                q_str = f"{q:+.2f}"
                tags.append("q_pos" if q >= 0 else "q_neg")

            # Visit count string
            visit_str = ""
            if visits and aid in visits:
                vc = visits[aid]
                if vc >= 1000:
                    visit_str = f"{vc/1000:.0f}k"
                else:
                    visit_str = str(vc)

            tree.insert(
                "",
                "end",
                iid=str(aid),
                values=(
                    aid,
                    prob_str,
                    q_str,
                    visit_str,
                    " -> ".join(str(square) for square in move.path),
                    ", ".join(str(square) for square in move.captures) or "-",
                ),
                tags=tuple(tags),
            )
        self._side_moves[side] = list(legal_moves)

    def update_notation(self, notation: Sequence[str]) -> None:
        self.notation_list.delete(0, "end")
        for item in notation:
            self.notation_list.insert("end", item)
        if notation:
            self.notation_list.see("end")

    # -- value curve ------------------------------------------------------

    def update_curve(self, history) -> None:
        """Replace all points on the value curve (called after undo/rewind)."""
        self.curve.update_all(history)

    def clear_curve(self) -> None:
        self.curve.clear()

    def _on_curve_point_click(self, ply: int) -> None:
        self.controller.rewind_to_ply(ply)

    # -- logging -----------------------------------------------------------

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
            self.pause_button.configure(
                text=self._tr("btn_resume") if paused else self._tr("btn_pause")
            )
            if paused:
                self.pause_button.configure(
                    bg="#e74c3c", activebackground="#c0392b",
                    fg="white", activeforeground="white",
                )
            else:
                self.pause_button.configure(
                    bg="#27ae60", activebackground="#219a52",
                    fg="white", activeforeground="white",
                )

    def show_mode_dialog(self, black_kind: str, white_kind: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title(self._tr("dialog_title"))
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        # Build player kind options dynamically from translations
        _kind_keys = ("human", "random", "alphabeta", "supervised", "mcts")
        _kind_options = tuple(self._tr(f"kind_{k}") for k in _kind_keys)
        _kind_label_to_key = {self._tr(f"kind_{k}"): k for k in _kind_keys}

        black_var = tk.StringVar(value=self._tr(f"kind_{black_kind}"))
        white_var = tk.StringVar(value=self._tr(f"kind_{white_kind}"))

        frame = ttk.Frame(dialog, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text=self._tr("dialog_prompt")).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 10),
        )

        ttk.Label(frame, text=self._tr("dialog_black")).grid(row=1, column=0, sticky="w", pady=4)
        black_combo = ttk.Combobox(
            frame,
            textvariable=black_var,
            values=_kind_options,
            state="readonly",
            width=14,
        )
        black_combo.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text=self._tr("dialog_white")).grid(row=2, column=0, sticky="w", pady=4)
        white_combo = ttk.Combobox(
            frame,
            textvariable=white_var,
            values=_kind_options,
            state="readonly",
            width=14,
        )
        white_combo.grid(row=2, column=1, sticky="ew", pady=4)

        # 推理加速复选框
        fp16_var = tk.BooleanVar(value=False)
        compile_var = tk.BooleanVar(value=False)
        chk_fp16 = ttk.Checkbutton(
            frame, text=self._tr("dialog_fp16"), variable=fp16_var,
        )
        chk_fp16.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        chk_compile = ttk.Checkbutton(
            frame, text=self._tr("dialog_compile"), variable=compile_var,
        )
        chk_compile.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _needs_checkpoint(kind: str) -> bool:
            return kind in ("supervised", "mcts")

        def _needs_mcts_config(kind: str) -> bool:
            return kind == "mcts"

        def start_game() -> None:
            black_selected = _kind_label_to_key[black_var.get()]
            white_selected = _kind_label_to_key[white_var.get()]
            checkpoint_paths: Dict[int, str] = {}
            mcts_config_paths: Dict[int, str] = {}

            for side, kind in ((BLACK_SIDE, black_selected), (WHITE_SIDE, white_selected)):
                side_tr = self._tr("side_black" if side == BLACK_SIDE else "side_white")
                kind_tr = self._tr(f"kind_{kind}")
                if _needs_checkpoint(kind):
                    path = self.ask_checkpoint_path(
                        self._tr("dialog_checkpoint", side=side_tr, kind=kind_tr),
                        dialog,
                    )
                    if path is None:
                        return
                    checkpoint_paths[side] = path

                if _needs_mcts_config(kind):
                    cfg_path = self.ask_mcts_config_path(
                        self._tr("dialog_mcts_config", side=side_tr),
                        dialog,
                    )
                    if cfg_path is not None:
                        mcts_config_paths[side] = cfg_path

            try:
                self.controller.start_new_game(
                    black_selected, white_selected,
                    checkpoint_paths, mcts_config_paths,
                    force_fp16=fp16_var.get(),
                    force_compile=compile_var.get(),
                )
            except Exception as exc:
                messagebox.showerror(self._tr("dialog_create_fail"), str(exc), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(buttons, text=self._tr("dialog_start"), command=start_game).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", start_game)
        dialog.bind("<Return>", lambda _event: start_game())
        black_combo.focus_set()
        self.wait_window(dialog)

    def show_illegal_move_dialog(self, square_id: int) -> bool:
        """Show a modal dialog when a human clicks an immovable own piece.

        Returns ``True`` if the user chose to forfeit, ``False`` to continue.
        """
        side = self.controller.current_side()
        side_name = self._tr("side_black" if side == BLACK_SIDE else "side_white")

        result: List[bool] = []  # mutable box to capture the choice

        dialog = tk.Toplevel(self)
        dialog.title(self._tr("dialog_illegal_title"))
        dialog.transient(self)
        dialog.resizable(False, False)

        # Use the same CJK font family discovered at startup
        _font = font.Font(family=self._cjk_family, size=12)

        frame = ttk.Frame(dialog, padding=(24, 20, 24, 20))
        frame.grid(row=0, column=0, sticky="nsew")

        # Message — use tk.Label (not ttk) with explicit CJK font to avoid
        # rendering issues on noXft Tk builds where the ttk style may not
        # propagate to Toplevel children reliably.
        tk.Label(
            frame,
            text=self._tr("dialog_illegal_prompt", side=side_name, sq=square_id),
            wraplength=380,
            justify="left",
            font=_font,
        ).grid(row=0, column=0, columnspan=2, pady=(0, 18))

        # Buttons — also use tk buttons with explicit font for CJK safety
        btn_forfeit = tk.Button(
            frame,
            text=self._tr("btn_forfeit"),
            font=_font,
            bg="#e74c3c", fg="white",
            activebackground="#c0392b", activeforeground="white",
            relief="flat", padx=14, pady=4,
            command=lambda: (result.append(True), dialog.destroy()),
        )
        btn_forfeit.grid(row=1, column=0, sticky="e", padx=(0, 6))

        btn_continue = tk.Button(
            frame,
            text=self._tr("btn_continue_game"),
            font=_font,
            bg="#27ae60", fg="white",
            activebackground="#219a52", activeforeground="white",
            relief="flat", padx=14, pady=4,
            command=lambda: (result.append(False), dialog.destroy()),
        )
        btn_continue.grid(row=1, column=1, sticky="w", padx=(6, 0))

        # Center the button row
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Keyboard shortcuts: Enter = continue, Escape = continue
        dialog.bind("<Return>", lambda _e: (result.append(False), dialog.destroy()))
        dialog.bind("<Escape>", lambda _e: (result.append(False), dialog.destroy()))
        # Closing the window via WM = continue
        dialog.protocol("WM_DELETE_WINDOW", lambda: (result.append(False), dialog.destroy()))

        # Ensure the window is visible before grab_set (avoids "window not
        # viewable" error on some window managers).  Also compute geometry
        # to centre the dialog on the parent.
        dialog.update_idletasks()
        dialog.wait_visibility()
        dialog.grab_set()

        # Centre on parent
        try:
            pw = self.winfo_width()
            ph = self.winfo_height()
            px = self.winfo_rootx()
            py = self.winfo_rooty()
            dw = dialog.winfo_reqwidth()
            dh = dialog.winfo_reqheight()
            x = px + (pw - dw) // 2
            y = py + (ph - dh) // 2
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        dialog.lift()
        btn_continue.focus_set()

        self.wait_window(dialog)

        forfeited = len(result) > 0 and result[0]
        if forfeited:
            self.controller.forfeit(side)
        return forfeited

    def ask_checkpoint_path(self, title: str, parent: tk.Toplevel) -> Optional[str]:
        checkpoint_dir = _PROJECT_ROOT / "trainer" / "checkpoints"
        initial_dir = checkpoint_dir if checkpoint_dir.exists() else _PROJECT_ROOT
        path = filedialog.askopenfilename(
            parent=parent,
            title=title,
            initialdir=str(initial_dir),
            filetypes=(
                (self._tr("dialog_ft_pytorch"), "*.pt *.pth"),
                (self._tr("dialog_ft_all"), "*"),
            ),
        )
        if not path:
            messagebox.showwarning(
                self._tr("dialog_no_checkpoint"),
                self._tr("dialog_need_checkpoint"),
                parent=parent,
            )
            return None
        return path

    def ask_mcts_config_path(self, title: str, parent: tk.Toplevel) -> Optional[str]:
        config_dir = _PROJECT_ROOT / "draughts" / "config"
        initial_dir = config_dir if config_dir.exists() else _PROJECT_ROOT
        path = filedialog.askopenfilename(
            parent=parent,
            title=title,
            initialdir=str(initial_dir),
            filetypes=(
                (self._tr("dialog_ft_yaml"), "*.yaml *.yml"),
                (self._tr("dialog_ft_all"), "*"),
            ),
        )
        return path if path else None

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
        self.log(self._tr("log_board_rotated"), "info")

    def _on_toggle_pause(self) -> None:
        self.controller.toggle_pause()

    def _on_ai_step_once(self) -> None:
        self.controller.ai_step_once()

    def _on_toggle_language(self) -> None:
        """Toggle between Chinese and English UI."""
        new_lang = "en" if self.lang == "zh" else "zh"
        self._switch_language(new_lang)

    def _switch_language(self, lang: str) -> None:
        """Reconfigure all tracked widgets for the new language."""
        if lang == self.lang:
            return
        self.lang = lang

        # Window title
        self.title(self._tr("window_title"))

        # All tracked widgets
        for wtype, widget, key in self._i18n_widgets:
            try:
                if wtype == "button":
                    if widget.winfo_exists():
                        widget.configure(text=self._tr(key))
                elif wtype == "labelframe":
                    if widget.winfo_exists():
                        widget.configure(text=self._tr(key))
                elif wtype == "label":
                    if widget.winfo_exists():
                        widget.configure(text=self._tr(key))
            except Exception:
                pass  # widget may have been destroyed

        # Treeview headings — both columns
        for tv in (self._black_move_tree, self._white_move_tree):
            try:
                tv.heading("action", text=self._tr("tree_action"))
                tv.heading("prob", text=self._tr("tree_prob"))
                tv.heading("qval", text=self._tr("tree_qval"))
                tv.heading("visits", text=self._tr("tree_visits"))
                tv.heading("path", text=self._tr("tree_path"))
                tv.heading("captures", text=self._tr("tree_captures"))
            except Exception:
                pass

        # Active side highlight
        self._highlight_active_side(self.controller.current_side())

        # Analysis panel placeholders
        self.search_stats_vars["eval"].set(self._tr("analysis_na"))
        self.search_stats_vars["sims"].set(self._tr("analysis_na"))
        self.search_stats_vars["best"].set(self._tr("analysis_na"))

        # Pause button and lang toggle
        self.update_pause_button(self.controller.ai_paused)
        self._lang_button.configure(text=self._tr("lang_toggle"))

        # Redraw curve (Y-axis labels, empty-state text follow language)
        self.curve._redraw()

        # Refresh dynamic labels
        self.controller.refresh()


def main() -> None:
    app = DraughtsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
