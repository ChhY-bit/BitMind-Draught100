from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_DIR = PROJECT_ROOT / "interface"
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from draughts_env import DraughtsEnv, Move  # noqa: E402
from draughts.agents import (  # noqa: E402
    AlphaBetaAgent,
    MCTSAgent,
    SupervisedAgent,
)
from draughts.env import EncodedObservationAdapter, StandardDraughtsEnv, encode_observation  # noqa: E402
from draughts.move_encoding import coord_to_square, encode_legal_moves, move_to_dict, square_to_coord  # noqa: E402


# ---------------------------------------------------------------------------
# CJK font registration — multi-backend, works across Wayland / Xorg /
# macOS / Windows without user intervention.
# ---------------------------------------------------------------------------

# Font families we try, ordered by preference.  The list covers:
#   • fontconfig-managed fonts (Tk built with Xft on Linux)
#   • the bundled TTF installed into fontconfig / X11 path at startup
#   • X11 core bitmaps (last resort — limited encoding, may render poorly)
#   • Windows / macOS system fonts
_PREFERRED_CJK_FONTS = [
    # fontconfig families (Xft-capable Tk on Linux)
    'noto sans cjk sc', 'noto sans cjk tc', 'noto sans cjk jp',
    'ar pl uming cn', 'ar pl ukai cn',
    'wenquanyi micro hei', 'wenquanyi zen hei',
    # bundled font — family name as registered via fontconfig *or* X11
    'droid sans fallback',
    # Windows
    'microsoft yahei', 'simhei',
    # macOS
    'pingfang sc', 'heiti sc',
]

# ---------------------------------------------------------------------------
# Dynamic system CJK font discovery
# ---------------------------------------------------------------------------


def _discover_system_cjk_ttfs() -> list[str]:
    """Find CJK TrueType/OpenType font files installed on the system.

    Uses ``fc-list`` (fontconfig) for cross-distro discovery, with
    hardcoded fallback paths for minimal environments.

    Returns file paths ordered by preference (sans-serif families first,
    then serif) so the smoothest screen fonts are tried earliest.
    """
    discovered: list[str] = []

    # ── Dynamic discovery via fontconfig (works on all Linux distros) ──
    try:
        result = subprocess.run(
            ['fc-list', ':lang=zh', 'file'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                # Format: /path/to/font.ttc: family,family:style=Regular
                filepath = line.rsplit(':', 1)[0].strip()
                if filepath and os.path.exists(filepath):
                    discovered.append(filepath)
    except Exception:
        pass

    # ── Fallback: walk common system font directories ──
    if not discovered:
        for search_dir in [
            '/usr/share/fonts/truetype/wqy',
            '/usr/share/fonts/opentype/noto',
            '/usr/share/fonts/truetype/noto',
            '/usr/share/fonts/truetype/arphic',
            '/usr/share/fonts/opentype',
            '/usr/share/fonts/truetype',
            '/usr/local/share/fonts',
        ]:
            if not os.path.isdir(search_dir):
                continue
            for dirpath, _dirs, fnames in os.walk(search_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.ttf', '.ttc', '.otf')):
                        discovered.append(os.path.join(dirpath, fname))

    # ── Rank: sans-serif before serif (smoother at small sizes) ──
    # Also prefer Regular weight and prefer WenQuanYi (battle-tested on
    # XWayland / noXft combos).  Limit to 5 candidates to avoid
    # pointlessly copying dozens of weight variants into the X11 dir.
    _SANS_FAMILIES = (
        'wqy-microhei', 'wenquanyi micro hei', 'wenquanyi zen hei',
        'notosans', 'noto sans',
        'sourcehansans', 'source han sans',
        'microsoft yahei', 'simhei',
    )

    def _rank(p: str) -> tuple[int, int, int, str]:
        basename = os.path.basename(p).lower()
        # 1 = serif, 0 = sans-serif (sans is preferred)
        is_serif = 0 if any(k in basename for k in _SANS_FAMILIES) else 1
        # Prefer WQY Micro Hei (proven to work on XWayland)
        is_wqy = 0 if 'wqy-microhei' in basename else 1
        # Prefer Regular / Medium weight; demote Black/Thin/Light
        is_regular = 0 if any(w in basename for w in ('regular', 'medium', 'micro')) else 1
        return (is_serif, is_wqy, is_regular, basename)

    ranked = sorted(set(discovered), key=_rank)
    # Only feed the top candidates into X11 — there's no point copying
    # 7 weight variants of Noto Sans when one will do.
    return ranked[:5]

# X11 core bitmap CJK fonts — only used when nothing else is available.
# These use legacy encodings (GB2312) which may show garbled glyphs for
# characters outside the encoding's repertoire.
_FALLBACK_CJK_FONTS = ['song ti', 'fangsong ti']

# Track resources for cleanup on exit.
_registered_font_dir: Optional[str] = None
_fontconfig_font_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Font installation helpers
# ---------------------------------------------------------------------------


def _get_tk_font_families(root: tk.Tk) -> set:
    try:
        return set(root.tk.call('font', 'families'))
    except Exception:
        return set()


def _install_fontconfig(bundled_ttf: str) -> Optional[str]:
    """Copy the bundled TTF into the user fontconfig directory and run
    ``fc-cache``.  Returns the font family name on success.

    This is the preferred path for Tk builds that link against Xft /
    fontconfig (modern Linux distro packages, most pip wheels).
    """
    global _fontconfig_font_dir

    if not sys.platform.startswith('linux'):
        return None

    # Use XDG data directory (standard for user fonts)
    font_dir = os.path.join(
        os.environ.get('XDG_DATA_HOME', os.path.join(str(Path.home()), '.local', 'share')),
        'fonts',
    )
    os.makedirs(font_dir, exist_ok=True)

    dest = os.path.join(font_dir, os.path.basename(bundled_ttf))
    if not os.path.exists(dest):
        try:
            shutil.copy2(bundled_ttf, dest)
        except OSError:
            return None

    # Refresh fontconfig cache (ignore failures — font may still work)
    try:
        subprocess.run(
            ['fc-cache', '-f', font_dir],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass

    _fontconfig_font_dir = font_dir

    # Parse the TTF to extract the font family name using fc-scan
    try:
        result = subprocess.run(
            ['fc-scan', '--format=%{family}', dest],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            # fc-scan may return comma-separated names; take the first
            family = result.stdout.strip().split(',')[0].strip()
            return family
    except Exception:
        pass

    return None


def _install_x11_font(source_path: str) -> Optional[str]:
    """Register a TTF/TTC file in the X11 core font path (``xset +fp``).

    This is the fallback for Tk builds **without** Xft (e.g. conda's
    ``tk-noxft`` package).  It requires ``mkfontscale`` / ``mkfontdir``
    and a running X server that accepts font-path changes.

    Args:
        source_path: Path to a ``.ttf`` or ``.ttc`` file on disk.

    Returns the X11 font family name on success, or ``None``.
    """
    global _registered_font_dir

    if not sys.platform.startswith('linux'):
        return None

    font_dir = os.path.join(str(Path.home()), '.bitmind_draughts', 'fonts')
    os.makedirs(font_dir, exist_ok=True)

    dest = os.path.join(font_dir, os.path.basename(source_path))
    if not os.path.exists(dest):
        try:
            shutil.copy2(source_path, dest)
        except OSError:
            return None

    # Generate X11 font metadata
    try:
        subprocess.run(
            ['mkfontscale', font_dir],
            check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            ['mkfontdir', font_dir],
            check=True, capture_output=True, timeout=10,
        )
    except Exception:
        return None

    # Add to X server font path
    try:
        subprocess.run(
            ['xset', '+fp', font_dir],
            check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            ['xset', 'fp', 'rehash'],
            check=True, capture_output=True, timeout=10,
        )
    except Exception:
        return None

    _registered_font_dir = font_dir

    # Parse fonts.dir to extract the X11 font family name
    fonts_dir_file = os.path.join(font_dir, 'fonts.dir')
    try:
        with open(fonts_dir_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0].isdigit():
                    continue
                idx = line.find(' -')
                if idx < 0:
                    continue
                xlfd = line[idx + 1:].strip()
                fields = xlfd.split('-')
                if len(fields) >= 3:
                    return fields[2]
    except Exception:
        pass

    return None


def _unregister_fonts() -> None:
    """Best-effort cleanup of font registrations on exit."""
    global _registered_font_dir
    if _registered_font_dir is not None:
        try:
            subprocess.run(
                ['xset', '-fp', _registered_font_dir],
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ['xset', 'fp', 'rehash'],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        _registered_font_dir = None


# ---------------------------------------------------------------------------
# CJK rendering test
# ---------------------------------------------------------------------------


def _cjk_renders_correctly(family: str, root: tk.Tk) -> bool:
    """Verify *family* actually renders CJK glyphs correctly.

    Some font backends (notably XWayland serving TrueType fonts through
    the X11 core protocol) produce corrupted glyph metrics — CJK
    characters get wildly inconsistent widths.  We detect this by
    measuring several CJK glyphs and checking they are all roughly the
    same width (as expected for ideographic characters).

    Also catches fonts that lack CJK entirely (zero-width glyphs).
    """
    try:
        f = tk.font.Font(root=root, family=family, size=14)
        # Sample characters: 中对局黑白 (common chars used in the UI)
        cjk_widths = [f.measure(ch) for ch in '中对局黑白']
        ascii_w = f.measure('A')

        # Every CJK character must have a reasonable width.
        if any(w < ascii_w * 0.6 for w in cjk_widths):
            return False

        # CJK characters must be mutually consistent (±30 % of median).
        median = sorted(cjk_widths)[len(cjk_widths) // 2]
        if median <= 0:
            return False
        if any(abs(w - median) / median > 0.35 for w in cjk_widths):
            return False  # inconsistent → corrupted glyph mapping

        return True
    except Exception:
        return False


BOARD_SIZE = 10
PLAYABLE_SQUARES = 50
CELL_SIZE = 68
BOARD_PIXELS = BOARD_SIZE * CELL_SIZE

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
WHITE_MAN = 3
WHITE_KING = 4

BLACK_SIDE = 0
WHITE_SIDE = 1

# ---------------------------------------------------------------------------
# Translation table — keys are semantic identifiers, values are
# {"zh": ..., "en": ...}.  All user-visible strings live here.
# ---------------------------------------------------------------------------

_TRANSLATIONS = {
    # ---- Window / frame titles ----
    "window_title":         {"zh": "BitMind 国际跳棋",       "en": "BitMind Draughts"},
    "notation_frame":       {"zh": "棋谱",                    "en": "Notation"},
    "engine_frame":         {"zh": "引擎分析",                "en": "Engine Analysis"},
    "status_frame":         {"zh": "对局状态",                "en": "Game Status"},

    # ---- Status row labels ----
    "status_mode":          {"zh": "模式",                    "en": "Mode"},
    "status_side":          {"zh": "行棋方",                  "en": "Side"},
    "status_moves":         {"zh": "合法动作",                "en": "Legal Moves"},
    "status_result":        {"zh": "局面结果",                "en": "Result"},
    "status_last":          {"zh": "最近动作",                "en": "Last Move"},

    # ---- Buttons ----
    "btn_new_game":         {"zh": "新建对局",                "en": "New Game"},
    "btn_restart":          {"zh": "重开本局",                "en": "Restart"},
    "btn_undo":             {"zh": "悔棋",                    "en": "Undo"},
    "btn_rotate":           {"zh": "旋转棋盘",                "en": "Rotate"},
    "btn_pause":            {"zh": "暂停AI",                  "en": "Pause AI"},
    "btn_resume":           {"zh": "继续AI",                  "en": "Resume AI"},
    "btn_ai_step":          {"zh": "AI走一步",                "en": "AI Step"},

    # ---- Language toggle ----
    "lang_toggle":          {"zh": "English",                 "en": "中文"},

    # ---- Treeview headings ----
    "tree_action":          {"zh": "编号",                    "en": "ID"},
    "tree_prob":            {"zh": "概率",                    "en": "Prob"},
    "tree_qval":            {"zh": "Q值",                     "en": "Q"},
    "tree_visits":          {"zh": "访问",                    "en": "N"},
    "tree_path":            {"zh": "路径",                    "en": "Path"},
    "tree_captures":        {"zh": "吃子",                    "en": "Captures"},

    # ---- Section labels ----
    "section_moves":        {"zh": "合法动作",                "en": "Legal Moves"},
    "section_log":          {"zh": "日志",                    "en": "Log"},

    # ---- Analysis panel ----
    "analysis_value":       {"zh": "估值",                    "en": "Value"},
    "analysis_search":      {"zh": "搜索量",                  "en": "Search"},
    "analysis_best":        {"zh": "首选",                    "en": "Best"},
    "analysis_eval":        {"zh": "{v:+.3f}  →  {side}方胜率 {pct:.0f}%",
                             "en": "{v:+.3f} → {side} win {pct:.0f}%"},
    "analysis_sims":        {"zh": "{sims} 次模拟  (c_puct={c_puct})",
                             "en": "{sims} sims (c_puct={c_puct})"},
    "analysis_depth":       {"zh": "depth={depth}",           "en": "depth={depth}"},
    "analysis_best_fmt":    {"zh": "#{aid} ({pct:.1f}%)",     "en": "#{aid} ({pct:.1f}%)"},
    "analysis_delta":       {"zh": "搜索修正",                "en": "Search Δ"},
    "analysis_delta_fmt":   {"zh": "网络{netv:+.2f}→搜索{searchv:+.2f}",
                             "en": "Net{netv:+.2f}→Search{searchv:+.2f}"},
    "analysis_na":          {"zh": "—",                       "en": "—"},

    # ---- Side labels ----
    "side_black":           {"zh": "黑方",                    "en": "Black"},
    "side_white":           {"zh": "白方",                    "en": "White"},
    "side_black_short":     {"zh": "黑",                      "en": "B"},
    "side_white_short":     {"zh": "白",                      "en": "W"},

    # ---- Result labels ----
    "result_ongoing":       {"zh": "进行中",                  "en": "Ongoing"},
    "result_black_win":     {"zh": "黑方胜",                  "en": "Black Wins"},
    "result_white_win":     {"zh": "白方胜",                  "en": "White Wins"},
    "result_draw":          {"zh": "和棋",                    "en": "Draw"},

    # ---- Player kind labels ----
    "kind_human":           {"zh": "人类",                    "en": "Human"},
    "kind_random":          {"zh": "随机AI",                  "en": "Random AI"},
    "kind_alphabeta":       {"zh": "AlphaBeta AI",            "en": "AlphaBeta AI"},
    "kind_supervised":      {"zh": "监督模型AI",              "en": "Supervised AI"},
    "kind_mcts":            {"zh": "MCTS AI",                 "en": "MCTS AI"},

    # ---- Dialog ----
    "dialog_title":         {"zh": "新建对局",                "en": "New Game"},
    "dialog_prompt":        {"zh": "选择双方决策源",          "en": "Select Players"},
    "dialog_black":         {"zh": "黑方",                    "en": "Black"},
    "dialog_white":         {"zh": "白方",                    "en": "White"},
    "dialog_start":         {"zh": "开始对局",                "en": "Start Game"},
    "dialog_create_fail":   {"zh": "创建对局失败",            "en": "Create Game Failed"},
    "dialog_no_checkpoint": {"zh": "未选择模型",              "en": "No Model Selected"},
    "dialog_need_checkpoint": {"zh": "请选择 checkpoint 文件后再开始对局。",
                               "en": "Please select a checkpoint file."},
    "dialog_checkpoint":    {"zh": "请选择{side} {kind} checkpoint",
                             "en": "Select {side} {kind} checkpoint"},
    "dialog_mcts_config":   {"zh": "请选择{side} MCTS 配置文件（取消则使用默认）",
                             "en": "Select {side} MCTS config (cancel for default)"},
    "dialog_ft_pytorch":    {"zh": "PyTorch checkpoint",      "en": "PyTorch checkpoint"},
    "dialog_ft_all":        {"zh": "所有文件",                "en": "All files"},
    "dialog_ft_yaml":       {"zh": "YAML config",             "en": "YAML config"},

    # ---- Player name templates ----
    "player_name":          {"zh": "{side}{kind}",             "en": "{side} {kind}"},
    "player_name_human":    {"zh": "{side}玩家",              "en": "{side} (Human)"},

    # ---- Mode label template ----
    "mode_label":           {"zh": "黑方{kind_black} vs 白方{kind_white}",
                             "en": "Black ({kind_black}) vs White ({kind_white})"},

    # ---- Log messages ----
    "log_new_game":         {"zh": "新对局开始：{mode}。",
                             "en": "New game: {mode}."},
    "log_ai_paused":        {"zh": "AI 自动走棋已暂停。",
                             "en": "AI auto-play paused."},
    "log_ai_resumed":       {"zh": "AI 自动走棋已继续。",
                             "en": "AI auto-play resumed."},
    "log_game_ended_ai":    {"zh": "对局已经结束，无法继续执行 AI。",
                             "en": "Game already ended."},
    "log_human_no_ai_step": {"zh": "当前行棋方是人类，不能执行 AI 走一步。",
                             "en": "Current player is human, cannot AI-step."},
    "log_game_over_btn":    {"zh": "对局已经结束。请点击\"重新开始\"。",
                             "en": "Game over. Click \"Restart\"."},
    "log_ai_controlled":    {"zh": "当前行棋方由 AI 控制。",
                             "en": "Current turn is AI-controlled."},
    "log_selection_cancel": {"zh": "已取消选择。",
                             "en": "Selection cancelled."},
    "log_ambiguous":        {"zh": "有 {count} 条路线都落在 {sq} 号位，请在右侧动作列表中双击选择完整路线。",
                             "en": "{count} routes land on #{sq}, double-click in move list."},
    "log_not_landing":      {"zh": "该位置不是当前棋子的合法落点。",
                             "en": "Not a valid landing square."},
    "log_ai_ctrl_step":     {"zh": "当前行棋方由 AI 控制，请使用\"AI走一步\"或自动走棋。",
                             "en": "AI-controlled turn. Use \"AI Step\" or auto-play."},
    "log_action_stale":     {"zh": "动作 {aid} 已不再合法。",
                             "en": "Action {aid} no longer legal."},
    "log_exec_fail":        {"zh": "执行失败：{error}",
                             "en": "Execution failed: {error}"},
    "log_move":             {"zh": "第 {turn} 手，{actor} 执行动作 {aid}：{move}",
                             "en": "Move {turn}: {actor} action {aid}: {move}"},
    "log_game_result":      {"zh": "对局结束：{result}。",
                             "en": "Game over: {result}."},
    "log_no_undo":          {"zh": "当前没有可悔的动作。",
                             "en": "No moves to undo."},
    "log_undo_fail":        {"zh": "悔棋失败：{error}",
                             "en": "Undo failed: {error}"},
    "log_undone":           {"zh": "已悔棋：{notation}",
                             "en": "Undid: {notation}"},
    "log_no_moves":         {"zh": "{name} 没有可执行动作。",
                             "en": "{name} has no legal moves."},
    "log_model_loaded":     {"zh": "{name} 已加载模型：{file}",
                             "en": "{name} loaded model: {file}"},
    "log_model_loaded_mcts":{"zh": "{name} 已加载模型：{file}（MCTS 配置：{cfg}）",
                             "en": "{name} loaded model: {file} (MCTS config: {cfg})"},
    "log_board_rotated":    {"zh": "棋盘视角已旋转 180 度。",
                             "en": "Board rotated 180°."},
    "log_no_legal_moves":   {"zh": "该棋子当前没有合法动作。",
                             "en": "No legal moves for this piece."},
    "log_square_selected":  {"zh": "已选择 {sq} 号位，共 {count} 条合法路线。",
                             "en": "Selected #{sq}, {count} legal routes."},
    "log_need_checkpoint":  {"zh": "{side}监督模型AI需要选择 checkpoint 文件",
                             "en": "{side} Supervised AI needs a checkpoint file"},
    "log_need_mcts_ckpt":   {"zh": "{side}MCTS AI需要选择 checkpoint 文件",
                             "en": "{side} MCTS AI needs a checkpoint file"},
    # -- value curve --
    "curve_frame":          {"zh": "估值曲线",                "en": "Value Curve"},
    "curve_empty":          {"zh": "(无 AI 参与，估值曲线不可用)",
                             "en": "(No AI in game — curve unavailable)"},
    "curve_black_pct":      {"zh": "黑方胜率%",               "en": "Black Win%"},
    "curve_black_wins":     {"zh": "黑胜",                    "en": "Black"},
    "curve_white_wins":     {"zh": "白胜",                    "en": "White"},
    "curve_ply":            {"zh": "步",                       "en": "Ply"},
    "log_rewind":           {"zh": "已回溯至第 {ply} 步",       "en": "Rewound to ply {ply}"},
    "log_rewind_paused":    {"zh": "已暂停对弈并回溯至第 {ply} 步",
                             "en": "Paused game and rewound to ply {ply}"},
    # -- hardware monitor --
    "mon_frame":            {"zh": "系统监控",                "en": "System Monitor"},
    "mon_cpu":              {"zh": "CPU 总利用率",             "en": "CPU Usage"},
    "mon_ram":              {"zh": "系统内存",                "en": "Sys RAM"},
    "mon_proc":             {"zh": "进程 CPU",               "en": "Proc CPU"},
    "mon_gpu":              {"zh": "GPU 利用率",              "en": "GPU Usage"},
    "mon_vram":             {"zh": "显存占用",                "en": "VRAM"},
    "mon_temp":             {"zh": "GPU 温度",               "en": "GPU Temp"},
    "mon_no_gpu":           {"zh": "无 GPU",                 "en": "No GPU"},
}


def _tr_static(key: str, lang: str) -> str:
    """Look up a translation key for the given language.

    Falls back to ``zh`` if the language or key is missing.
    """
    entry = _TRANSLATIONS.get(key)
    if entry is None:
        return key  # safe fallback — shows the key name instead of crashing
    return entry.get(lang, entry.get("zh", key))


@dataclass
class PlayerAnalysis:
    """Rich analysis data produced by an AI player during move selection."""

    value: Optional[float] = None
    search_value: Optional[float] = None  # MCTS root Q (may differ from network value)
    policy_probs: Optional[dict[int, float]] = None
    search_info: Optional[dict[str, object]] = None
    analysed_action_ids: Optional[set[int]] = None
    visit_counts: Optional[dict[int, int]] = None
    q_values: Optional[dict[int, float]] = None


@dataclass
class HumanPlayer:
    name: str
    kind: str = "human"

    def __post_init__(self) -> None:
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return True


@dataclass
class RandomPlayer:
    """GUI player adapter for random move selection.

    This is distinct from draughts.agents.RandomAgent — it uses the GUI's
    choose_move() contract (returns Move or None) rather than the standard
    Agent.select_action() interface (returns action_id).
    """

    name: str
    rng: random.Random
    kind: str = "random"

    def __post_init__(self) -> None:
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None) -> Optional[Move]:
        if not moves:
            return None
        self.last_analysis = PlayerAnalysis()
        return self.rng.choice(list(moves))


@dataclass
class SupervisedPlayer:
    name: str
    checkpoint_path: str
    kind: str = "supervised"

    def __post_init__(self) -> None:
        self.agent = SupervisedAgent(self.checkpoint_path, device="cpu")
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

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
        self._capture_analysis(moves)
        return find_move_by_action_id(moves, action_id)

    def _capture_analysis(self, moves: Sequence[Move]) -> None:
        logits = getattr(self.agent, "last_policy_logits", None)
        value = getattr(self.agent, "last_value", None)
        if logits is not None and value is not None and len(logits) == len(moves):
            probs = _softmax_list(logits)
            self.last_analysis = PlayerAnalysis(
                value=value,
                policy_probs={
                    moves[i].action_id: probs[i] for i in range(len(moves))
                },
                analysed_action_ids=set(m.action_id for m in moves),
            )
        else:
            self.last_analysis = PlayerAnalysis()

    def infer_value(self, gui_env: DraughtsEnv) -> Optional[float]:
        """Run a single forward pass on the current position.

        Used by the value curve to record a post-move evaluation even
        when it is a human's turn (mixed data-collection strategy).
        """
        import torch

        obs = gui_env.observation()
        moves = gui_env.legal_moves()
        if not moves:
            return None
        adapter = EncodedObservationAdapter(gui_env)
        side = int(obs["side_to_move"])
        with torch.no_grad():
            obs_t = torch.tensor(
                [adapter.observation()], dtype=torch.float32,
                device=self.agent.device_obj,
            )
            move_dicts = [move_to_dict(m) for m in moves]
            move_t = torch.tensor(
                [encode_legal_moves(move_dicts, side_to_move=side)],
                dtype=torch.float32, device=self.agent.device_obj,
            )
            mask_t = torch.ones(
                1, len(moves), dtype=torch.float32, device=self.agent.device_obj,
            )
            output = self.agent.model(obs_t, move_t, mask_t)
            return float(output["value"][0].item())


@dataclass
class AlphaBetaPlayer:
    name: str
    kind: str = "alphabeta"

    def __post_init__(self) -> None:
        self.agent = AlphaBetaAgent()
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None) -> Optional[Move]:
        if not moves:
            return None
        if env is None:
            raise ValueError("AlphaBeta AI需要当前棋局环境")
        wrapped = GuiEnvAdapter(env)
        obs = wrapped.raw_observation()
        action_id = self.agent.select_action(obs, moves, wrapped)
        self.last_analysis = PlayerAnalysis(
            search_info={"depth": self.agent.config.search_depth},
        )
        return find_move_by_action_id(moves, action_id)


@dataclass
class MCTSPlayer:
    name: str
    checkpoint_path: str
    config_path: Optional[str] = None
    kind: str = "mcts"

    def __post_init__(self) -> None:
        self.agent = MCTSAgent(
            self.checkpoint_path,
            config_path=self.config_path,
            device="cpu",
        )
        self._search_env = StandardDraughtsEnv()
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None) -> Optional[Move]:
        if not moves:
            return None
        if env is None:
            raise ValueError("MCTS AI需要当前棋局环境")
        obs = env.observation()
        # Mirror the GUI env state into a StandardDraughtsEnv for MCTS search.
        self._search_env.set_state(
            squares=list(obs["squares"]),
            side_to_move=int(obs["side_to_move"]),
        )
        action_id = self.agent.select_action(
            self._search_env.raw_observation(),
            self._search_env.legal_actions(),
            self._search_env,
        )
        self._capture_analysis(moves)
        return find_move_by_action_id(moves, action_id)

    def _capture_analysis(self, moves: Sequence[Move]) -> None:
        result = self.agent.last_result
        if result is None:
            self.last_analysis = PlayerAnalysis()
            return

        search_lookup: dict[int, float] = dict(
            zip(result.action_ids, result.policy_probs)
        )
        probs_dict: dict[int, float] = {}
        visits_dict: dict[int, int] = {}
        qvals_dict: dict[int, float] = {}
        for move in moves:
            prob = search_lookup.get(move.action_id)
            if prob is not None:
                probs_dict[move.action_id] = prob
            if result.visit_counts:
                visits_dict[move.action_id] = result.visit_counts.get(move.action_id, 0)
            if result.q_values:
                qvals_dict[move.action_id] = result.q_values.get(move.action_id, 0.0)

        self.last_analysis = PlayerAnalysis(
            value=result.network_value,
            search_value=result.root_value,
            policy_probs=probs_dict,
            search_info={
                "simulations": self.agent.mcts_config.num_simulations,
                "c_puct": self.agent.mcts_config.c_puct,
            },
            analysed_action_ids=set(m.action_id for m in moves),
            visit_counts=visits_dict if visits_dict else None,
            q_values=qvals_dict if qvals_dict else None,
        )

    def infer_value(self, gui_env: DraughtsEnv) -> Optional[float]:
        """Run a single forward pass on the current position via MCTS net."""
        from draughts.mcts import _infer_network

        obs = gui_env.observation()
        self._search_env.set_state(
            squares=list(obs["squares"]),
            side_to_move=int(obs["side_to_move"]),
        )
        legal_actions = self._search_env.legal_actions()
        if not legal_actions:
            return None
        _, value = _infer_network(
            self._search_env, self.agent.model,
            legal_actions, self.agent.device_obj,
        )
        return value


@dataclass
class GuiEnvAdapter:
    """Wraps a C-level DraughtsEnv so it exposes the interface expected by
    alphabeta / MCTS agents (clone, step, raw_observation,
    legal_actions)."""

    _env: DraughtsEnv

    def clone(self) -> "GuiEnvAdapter":
        return GuiEnvAdapter(self._env.clone())

    def step(self, action_id: int):
        obs, info = self._env.step(action_id)
        return obs, info["reward"], info["done"], info

    def raw_observation(self) -> Dict[str, object]:
        return self._env.observation()

    def legal_actions(self) -> List[Move]:
        return self._env.legal_moves()


class EvalBar(tk.Canvas):
    """Vertical evaluation bar showing win probability from black's view.

    The top portion (dark fill) represents black's win chance; the bottom
    portion (light fill) represents white's.  A centred label shows the
    current player's win percentage.  Value changes are animated with a
    smooth ease-in-out transition (~350 ms).
    """

    BAR_WIDTH = 38
    BAR_HEIGHT = BOARD_PIXELS
    _ANIM_STEPS = 20
    _ANIM_INTERVAL = 17  # ms (~60 fps)

    def __init__(self, master: tk.Widget, **kwargs) -> None:
        super().__init__(
            master,
            width=self.BAR_WIDTH,
            height=self.BAR_HEIGHT,
            background="#f0f2f5",
            highlightthickness=1,
            highlightbackground="#b0b8c1",
            **kwargs,
        )
        self._value: float = 0.0
        self._side_to_move: int = BLACK_SIDE
        # Animation state
        self._anim_job: Optional[str] = None
        self._anim_target: float = 0.0
        self._anim_start: float = 0.0
        self._anim_step_i: int = 0
        self._display_value: float = 0.0

    def set_value(self, value: float, side_to_move: int) -> None:
        """Update the evaluation with smooth transition.

        Args:
            value: Network/MCTS evaluation **from the current player's
                   perspective**, in [-1, 1].  +1 = current player winning.
            side_to_move: ``BLACK_SIDE`` (0) or ``WHITE_SIDE`` (1).
        """
        target = max(-1.0, min(1.0, float(value)))
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        # When the perspective flips (e.g. after a move), convert the
        # currently displayed value so the animation starts from the
        # same visual position instead of jumping.
        if self._side_to_move != side_to_move:
            self._anim_start = -self._display_value
            self._display_value = -self._display_value
        else:
            self._anim_start = self._display_value
        self._anim_target = target
        self._side_to_move = side_to_move  # target perspective
        self._anim_step_i = 0
        self._anim_tick()

    def clear(self) -> None:
        """Reset the bar to the neutral (even) position."""
        self.set_value(0.0, BLACK_SIDE)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _anim_tick(self) -> None:
        if self._anim_step_i >= self._ANIM_STEPS:
            self._display_value = self._anim_target
            self._value = self._display_value
            self._anim_job = None
            self._redraw()
            return

        # Smoothstep ease-in-out: 3t² - 2t³
        t = self._anim_step_i / self._ANIM_STEPS
        eased = 3.0 * t * t - 2.0 * t * t * t
        self._display_value = self._anim_start + (self._anim_target - self._anim_start) * eased
        self._value = self._display_value
        self._anim_step_i += 1
        self._redraw()
        self._anim_job = self.after(self._ANIM_INTERVAL, self._anim_tick)

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("all")

        w = self.BAR_WIDTH
        actual_h = self.winfo_height()
        h = actual_h if actual_h > 1 else self.BAR_HEIGHT

        # --- value from black's perspective ---
        if self._side_to_move == BLACK_SIDE:
            value_black = self._value
        else:
            value_black = -self._value
        black_share = (value_black + 1.0) / 2.0  # [0, 1]

        # Clamp so the smaller slice never vanishes completely
        black_share = max(0.03, min(0.97, black_share))
        # Y coordinate of the boundary: black occupies the *top*
        # (0 → split_y), white the *bottom* (split_y → h).
        split_y = int(h * black_share)

        # Black portion (top)
        self.create_rectangle(0, 0, w, split_y, fill="#1a2332", outline="")
        # White portion (bottom)
        self.create_rectangle(0, split_y, w, h, fill="#f5f0e8", outline="")

        # Centre divider (neutral reference)
        self.create_line(0, h // 2, w, h // 2, fill="#8899aa", dash=(3, 4))

        # ── Confrontation marker at the split ──
        # Two opposing triangles (► ◄) pointing at each other across the
        # boundary, evoking a battle / stand-off.
        cx = w // 2
        tri_h = 7   # half-height of each triangle
        tri_w = 9   # half-width
        # Upper triangle ▼ (pointing down toward the split)
        self.create_polygon(
            cx - tri_w, split_y - tri_h,
            cx + tri_w, split_y - tri_h,
            cx, split_y,
            fill="#f2c94c", outline="#c99a2c",
        )
        # Lower triangle ▲ (pointing up toward the split)
        self.create_polygon(
            cx - tri_w, split_y + tri_h,
            cx + tri_w, split_y + tri_h,
            cx, split_y,
            fill="#f2c94c", outline="#c99a2c",
        )


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
                            x0 + 22,
                            y0 + 22,
                            x1 - 22,
                            y1 - 22,
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
                        font=("TkDefaultFont", 9),
                    )

        for square_id, piece in enumerate(squares, start=1):
            if piece != EMPTY:
                self._draw_piece(square_id, piece)

    def _draw_piece(self, square_id: int, piece: int) -> None:
        row, col = square_to_coord(square_id)
        view_row, view_col = self._view_coord_from_board(row, col)
        x0 = view_col * CELL_SIZE
        y0 = view_row * CELL_SIZE
        pad = 7
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
            width=4,
        )
        if text:
            self.create_text(
                x0 + CELL_SIZE / 2,
                y0 + CELL_SIZE / 2,
                text=text,
                fill=text_fill,
                font=("TkDefaultFont", 20, "bold"),
            )

    def _view_coord_from_board(self, row: int, col: int) -> tuple[int, int]:
        if not self.rotated:
            return row, col
        return BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col

    def _board_coord_from_view(self, row: int, col: int) -> tuple[int, int]:
        return self._view_coord_from_board(row, col)


# ---------------------------------------------------------------------------
# Value curve widget
# ---------------------------------------------------------------------------


class ValueCurve(tk.Frame):
    """Scrollable line chart of position evaluation across plies.

    Plots black-win% (derived from the network value head) after each
    move, with a horizontal scrollbar for navigating long games.
    """

    POINT_SPACING = 30
    CURVE_HEIGHT = 160
    MARGIN_TOP = 12
    MARGIN_BOTTOM = 24
    MARGIN_LEFT = 42  # room for Y-axis % labels
    MARGIN_RIGHT = 24
    DOT_RADIUS = 3
    MAX_POINTS = 300

    def __init__(
        self, parent: tk.Widget, on_point_click=None, app=None,
    ) -> None:
        super().__init__(parent)
        self._on_point_click_cb = on_point_click
        self._app = app
        self._points: list[tuple[int, float, int, str]] = []  # (ply, value, side, notation)
        self._current_ply: int = -1
        self._dot_positions: list[tuple[int, float, float]] = []  # (ply, canvas_x, canvas_y)

        bar_frame = ttk.Frame(self)
        bar_frame.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(
            self, height=self._total_h,
            bg=self._BG, highlightthickness=0,
        )
        self.canvas.pack(side="top", fill="x", expand=False)

        self.scrollbar = ttk.Scrollbar(
            bar_frame, orient="horizontal", command=self.canvas.xview,
        )
        self.scrollbar.pack(side="left", fill="x", expand=True)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self._redraw()

    # -- layout helpers ---------------------------------------------------

    @property
    def _total_h(self) -> int:
        return self.CURVE_HEIGHT + self.MARGIN_TOP + self.MARGIN_BOTTOM

    @property
    def _plot_top(self) -> int:
        return self.MARGIN_TOP

    @property
    def _plot_bottom(self) -> int:
        return self.MARGIN_TOP + self.CURVE_HEIGHT

    @property
    def _plot_left(self) -> int:
        return self.MARGIN_LEFT

    def _plot_right(self) -> int:
        # Use the available canvas width as the minimum, so the plot fills
        # the panel even with few points.  When points grow beyond canvas
        # width the horizontal scrollbar activates.
        canvas_w = self.canvas.winfo_width()
        if canvas_w < 20:  # not yet drawn — fall back to hard-coded minimum
            canvas_w = self._MIN_PLOT_PX + self.MARGIN_LEFT + self.MARGIN_RIGHT
        available_w = max(canvas_w - self.MARGIN_LEFT - self.MARGIN_RIGHT, self._MIN_PLOT_PX)
        needed_w = len(self._points) * self.POINT_SPACING
        return self._plot_left + max(needed_w, available_w)

    # -- value → coordinate helpers ---------------------------------------

    @staticmethod
    def _value_to_black_pct(value: float, side: int) -> float:
        """Convert value (∈[-1,1], current-player perspective) to black win%."""
        if side == BLACK_SIDE:
            return (1.0 + value) / 2.0 * 100.0
        else:
            return (1.0 - value) / 2.0 * 100.0

    def _value_to_y(self, value: float, side: int) -> float:
        black_pct = self._value_to_black_pct(value, side)
        return self._plot_bottom - (black_pct / 100.0) * self.CURVE_HEIGHT

    def _point_x(self, idx: int) -> float:
        return self._plot_left + idx * self.POINT_SPACING + self.POINT_SPACING / 2

    # -- public API -------------------------------------------------------

    def add_point(self, ply: int, value: float, side: int, notation: str = "") -> None:
        self._points.append((ply, float(value), int(side), notation))
        if len(self._points) > self.MAX_POINTS:
            self._points = self._points[-self.MAX_POINTS:]
        self._redraw()

    def remove_last(self) -> None:
        if self._points:
            self._points.pop()
            self._redraw()

    def set_current_ply(self, ply: int) -> None:
        self._current_ply = ply
        self._redraw()

    def clear(self) -> None:
        self._points.clear()
        self._current_ply = -1
        self._dot_positions.clear()
        self._redraw()

    def update_all(self, history) -> None:
        """Bulk-replace all points (used after undo / rewind)."""
        self._points = list(history)
        if len(self._points) > self.MAX_POINTS:
            self._points = self._points[-self.MAX_POINTS:]
        self._redraw()

    # -- font helpers ---------------------------------------------------

    def _cjk_font(self, size: int = 7, bold: bool = False) -> tuple:
        """Build a font spec using the app's resolved CJK family.

        On noXft Tk, ``("TkDefaultFont", …)`` tuples are interpreted as
        *family* names rather than named-font references, which breaks
        CJK rendering.  We use the discovered family directly (the same
        way ttk styles do) to keep Chinese / Japanese / Korean glyphs
        correct.
        """
        family = getattr(self._app, "_cjk_family", None) if self._app else None
        if family is None:
            return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)
        return (family, size, "bold") if bold else (family, size)

    # -- colour palette (light theme, matching the app) -------------------
    _BG = "#f5f6f8"
    _PLOT_BG = "#ffffff"
    _PLOT_BORDER = "#c8cdd5"
    _GRID_MINOR = "#e8ebf0"
    _GRID_MAJOR = "#c0c6d0"
    _AXIS_LABEL = "#889099"
    _LINE = "#3b6fb6"
    _LINE_FILL = "#dce8f5"
    _DOT_BORDER = "#ffffff"
    _DOT_FILL_HIGH = "#2c5282"
    _DOT_FILL_MID = "#7a8ba0"
    _DOT_FILL_LOW = "#c09070"
    _MARKER = "#e6b422"
    _MIN_PLOT_PX = 240  # minimum plot-area width so empty-state text fits

    # -- drawing -----------------------------------------------------------

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self._dot_positions.clear()

        total_w = self._plot_right() + self.MARGIN_RIGHT
        self.canvas.configure(scrollregion=(0, 0, total_w, self._total_h))

        plot_l = self._plot_left
        plot_t = self._plot_top
        plot_b = self._plot_bottom
        plot_r = max(self._plot_right(), plot_l + self._MIN_PLOT_PX)

        # Background
        self.canvas.create_rectangle(0, 0, total_w, self._total_h,
                                     fill=self._BG, outline="")
        # Plot area
        self.canvas.create_rectangle(plot_l, plot_t, plot_r, plot_b,
                                     fill=self._PLOT_BG, outline=self._PLOT_BORDER)

        # ── Y-axis: left-side axis line + tick marks + labels ──
        self.canvas.create_line(plot_l, plot_t, plot_l, plot_b,
                                fill=self._PLOT_BORDER, width=1)
        for pct in (0, 25, 50, 75, 100):
            y = plot_b - (pct / 100.0) * self.CURVE_HEIGHT
            is_major = (pct == 50)
            dash = (6, 4) if is_major else (2, 4)
            color = self._GRID_MAJOR if is_major else self._GRID_MINOR
            self.canvas.create_line(plot_l, y, plot_r, y, fill=color, dash=dash)
            self.canvas.create_line(plot_l - 4, y, plot_l, y, fill=self._PLOT_BORDER)
            self.canvas.create_text(plot_l - 8, y, text=f"{pct}%",
                                    anchor="e", font=self._cjk_font(7),
                                    fill=self._AXIS_LABEL)

        # ── X-axis: bottom line ──
        self.canvas.create_line(plot_l, plot_b, plot_r, plot_b,
                                fill=self._PLOT_BORDER, width=1)

        # ── Right-side labels ("Black Wins" / "White Wins") ──
        tr = self._app._tr if self._app else _tr_static
        self.canvas.create_text(
            plot_r + 6, plot_t + 3,
            text=tr("curve_black_wins"),
            anchor="nw", font=self._cjk_font(7, bold=True),
            fill=self._DOT_FILL_HIGH,
        )
        self.canvas.create_text(
            plot_r + 6, plot_b - 3,
            text=tr("curve_white_wins"),
            anchor="sw", font=self._cjk_font(7, bold=True),
            fill=self._DOT_FILL_LOW,
        )

        # ── Empty state ──
        if not self._points:
            tr = self._app._tr if self._app else _tr_static
            msg = tr("curve_empty") if self._app else "(No AI — curve unavailable)"
            self.canvas.create_text(
                (plot_l + plot_r) / 2, (plot_t + plot_b) / 2,
                text=msg, font=self._cjk_font(10), fill=self._AXIS_LABEL,
            )
            return

        # ── X-axis labels (every 5 plies) ──
        for i, (ply, _v, _s, _n) in enumerate(self._points):
            if ply % 5 == 0 or ply == 1:
                x = self._point_x(i)
                self.canvas.create_line(x, plot_b, x, plot_b + 4,
                                        fill=self._PLOT_BORDER)
                self.canvas.create_text(x, plot_b + 13, text=str(ply),
                                        font=self._cjk_font(7),
                                        fill=self._AXIS_LABEL, anchor="n")

        # ── Build data arrays ──
        coords: list[float] = []
        dot_data: list[tuple[float, float, float, int]] = []  # (x, y, black_pct, ply)
        for i, (_ply, value, side, _notation) in enumerate(self._points):
            x = self._point_x(i)
            y = self._value_to_y(value, side)
            coords.extend([x, y])
            black_pct = self._value_to_black_pct(value, side)
            dot_data.append((x, y, black_pct, self._points[i][0]))

        # ── Area fill under the curve (semi-transparent overlay) ──
        if len(coords) >= 4:
            mid_y = self._plot_bottom - self.CURVE_HEIGHT / 2.0
            poly_pts: list[float] = []
            for j in range(0, len(coords), 2):
                px, py = coords[j], coords[j + 1]
                poly_pts.extend([px, py])
            poly_pts.extend([coords[-2], mid_y, coords[0], mid_y])
            self.canvas.create_polygon(
                *poly_pts, fill=self._LINE_FILL, outline="",
            )

        # ── Connecting line ──
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=self._LINE, width=2.0,
                                    capstyle="round", joinstyle="round")

        # ── Data dots ──
        for x, y, black_pct, ply in dot_data:
            if black_pct >= 58:
                color = self._DOT_FILL_HIGH
            elif black_pct >= 42:
                color = self._DOT_FILL_MID
            else:
                color = self._DOT_FILL_LOW
            r = self.DOT_RADIUS
            # White border
            self.canvas.create_oval(
                x - r - 1, y - r - 1, x + r + 1, y + r + 1,
                fill=self._DOT_BORDER, outline=self._DOT_BORDER,
                tags=("dot",),
            )
            # Coloured centre
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline=color, tags=("dot",),
            )
            self._dot_positions.append((ply, x, y))

        # ── Current-ply marker ──
        if 0 < self._current_ply <= len(self._points):
            idx = self._current_ply - 1
            if 0 <= idx < len(self._points):
                x = self._point_x(idx)
                self.canvas.create_line(x, plot_t, x, plot_b,
                                        fill=self._MARKER, width=2, dash=(4, 3))

        # Auto-scroll to latest
        self.canvas.xview_moveto(1.0)

    # -- click handling ---------------------------------------------------

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self._dot_positions or not self._on_point_click_cb:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        closest_ply = -1
        closest_dist = float("inf")
        for ply, dx, dy in self._dot_positions:
            dist = ((canvas_x - dx) ** 2 + (canvas_y - dy) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_ply = ply

        if closest_ply >= 0 and closest_dist <= 10:
            self._on_point_click_cb(closest_ply)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hardware monitor — real-time CPU / GPU resource usage
# ---------------------------------------------------------------------------

class HardwareMonitor(tk.Canvas):
    """Compact system resource monitor with animated bar charts.

    2 columns × 3 rows: CPU usage, sys RAM, proc CPU | GPU usage, VRAM, temp.
    Data sources: /proc/stat, /proc/meminfo, /proc/self/stat, nvidia-smi,
    torch.cuda — zero new dependencies.

    Data collection runs in a daemon thread so nvidia-smi never blocks the
    Tk event loop.  Canvas items are created once then updated incrementally.
    """

    _BAR_H = 12
    _ROW_H = 20
    _PAD_T = 6
    _PAD_B = 6
    _PAD_L = 4
    _PAD_R = 4
    _LABEL_W = 88
    _GAP_COL = 24

    def __init__(self, parent, draughts_app=None, height=120):
        super().__init__(parent, bg="#f5f5f5", highlightthickness=1,
                         highlightbackground="#c8cdd5", height=height)
        self._app = draughts_app
        self._metrics: list[dict] = [
            {"key": "cpu",  "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "ram",  "current": 0.0, "target": 0.0, "fmt": "{:.1f}G",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "proc", "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "gpu",  "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "vram", "current": 0.0, "target": 0.0, "fmt": "{:.1f}G",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "temp", "current": 0.0, "target": 0.0, "fmt": "{:.0f}°C", "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
        ]
        self._gpu_available = False
        self._anim_id: Optional[str] = None
        self._anim_start = 0.0
        self._running = True
        self._items_created = False
        # Thread-safe: collector thread writes a fresh dict, main thread reads it.
        self._pending_data: Optional[dict] = None
        self._collect_lock = threading.Lock()

        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Destroy>", lambda _e: self._stop())

        # Start background collector + main-thread pick-up loop
        self._collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collector_thread.start()
        self._apply_loop()

    def _stop(self) -> None:
        self._running = False
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    # -- background data collection (never touches Tk) --------------------

    def _collect_loop(self) -> None:
        """Run in daemon thread: sample /proc + nvidia-smi, post results."""
        prev_cpu = None
        prev_proc_cpu = None
        while self._running:
            try:
                data = {}
                # CPU
                prev_cpu = self._sample_cpu(data, prev_cpu)
                # RAM
                self._sample_ram(data)
                # Proc CPU
                prev_proc_cpu = self._sample_proc_cpu(data, prev_proc_cpu)
                # GPU
                self._sample_gpu(data)
                # Publish
                self._pending_data = data
            except Exception:
                pass
            time.sleep(1.0)

    @staticmethod
    def _sample_cpu(data: dict, prev) -> tuple:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            if parts[0] == "cpu":
                ticks = [int(x) for x in parts[1:]]
                idle, total = ticks[3], sum(ticks)
                if prev is not None:
                    p_idle, p_total = prev
                    d_idle = idle - p_idle
                    d_total = total - p_total
                    if d_total > 0:
                        data["cpu"] = (1.0 - d_idle / d_total) * 100.0
                return (idle, total)
        except Exception:
            pass
        return prev

    @staticmethod
    def _sample_ram(data: dict) -> None:
        try:
            total = available = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1])
                    if total and available:
                        break
            if total > 0:
                used_gb = (total - available) / 1024.0 / 1024.0
                total_gb = total / 1024.0 / 1024.0
                data["ram"] = (used_gb, total_gb)
        except Exception:
            pass

    @staticmethod
    def _sample_proc_cpu(data: dict, prev) -> tuple:
        try:
            with open("/proc/self/stat") as f:
                stat = f.read().split()
            ticks = sum(int(stat[i]) for i in [13, 14, 15, 16])
            now = time.monotonic()
            if prev is not None:
                p_ticks, p_time = prev
                d_ticks = ticks - p_ticks
                d_time = now - p_time
                nprocs = os.cpu_count() or 1
                if d_time > 0:
                    pct = (d_ticks / os.sysconf("SC_CLK_TCK")) / d_time / nprocs * 100.0
                    data["proc"] = min(pct, 100.0 * nprocs)
            return (ticks, now)
        except Exception:
            pass
        return prev

    @staticmethod
    def _sample_gpu(data: dict) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = [x.strip() for x in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    data["gpu"] = float(parts[0])
                    data["vram"] = (float(parts[1]) / 1024.0, float(parts[2]) / 1024.0)
                    data["temp"] = float(parts[3])
                    data["gpu_avail"] = True
                    return
        except Exception:
            pass
        # Fallback: torch.cuda
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / 1024.0**3
                data["vram"] = (alloc, None)
                data["gpu"] = 0.0
                data["temp"] = 0.0
                data["gpu_avail"] = True
                return
        except Exception:
            pass
        data["gpu_avail"] = False

    # -- main-thread apply loop -------------------------------------------

    def _apply_loop(self) -> None:
        """Pick up data from the collector thread and update targets."""
        if not self._running:
            return
        data = self._pending_data
        if data is not None:
            self._pending_data = None
            # Apply CPU
            if "cpu" in data:
                self._set_target("cpu", data["cpu"])
            # Apply RAM
            if "ram" in data:
                used_gb, total_gb = data["ram"]
                self._set_target("ram", used_gb)
                m = next(x for x in self._metrics if x["key"] == "ram")
                m["fmt"] = f"{{:.1f}}/{total_gb:.1f}G"
                m["frac"] = used_gb / total_gb if total_gb > 0 else 0.0
            # Apply proc CPU
            if "proc" in data:
                self._set_target("proc", data["proc"])
            # Apply GPU
            gpu_avail = data.get("gpu_avail", False)
            self._gpu_available = gpu_avail
            if gpu_avail:
                if "gpu" in data:
                    self._set_target("gpu", data["gpu"])
                if "vram" in data:
                    vram_gb, vram_total = data["vram"]
                    self._set_target("vram", vram_gb)
                    m = next(x for x in self._metrics if x["key"] == "vram")
                    if vram_total is not None:
                        m["fmt"] = f"{{:.1f}}/{vram_total:.1f}G"
                        m["frac"] = vram_gb / vram_total if vram_total > 0 else 0.0
                    else:
                        m["fmt"] = "{:.1f}G"
                        m["frac"] = 0.0
                if "temp" in data:
                    self._set_target("temp", data["temp"])
                    m = next(x for x in self._metrics if x["key"] == "temp")
                    m["frac"] = data["temp"] / 100.0
        self.after(1000, self._apply_loop)

    def _set_target(self, key: str, value: float) -> None:
        for m in self._metrics:
            if m["key"] == key:
                m["target"] = value
                break
        self._start_animation()

    # -- animation (main thread) ------------------------------------------

    def _start_animation(self) -> None:
        if self._anim_id is not None:
            return
        self._anim_start = time.monotonic()
        self._animate()

    def _animate(self) -> None:
        self._anim_id = None
        now = time.monotonic()
        t = min((now - self._anim_start) / 0.25, 1.0)
        eased = 1.0 - (1.0 - t) * (1.0 - t)  # ease-out quad
        dirty = False
        for m in self._metrics:
            cur = m["current"]
            tgt = m["target"]
            if abs(cur - tgt) < 0.05:
                m["current"] = tgt
            else:
                m["current"] = cur + (tgt - cur) * eased
                dirty = True
        self._redraw()
        if dirty and t < 1.0:
            self._anim_id = self.after(50, self._animate)

    # -- drawing (incremental after first paint) --------------------------

    def _font(self, size=9, bold=False):
        family = getattr(self._app, "_cjk_family", None) if self._app else None
        if family is None:
            return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)
        return (family, size, "bold") if bold else (family, size)

    def _redraw(self) -> None:
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 20 or h < 20:
            return
        tr = self._app._tr if self._app else _tr_static

        mid_x = w / 2 + self._GAP_COL / 2
        bar_max_w = mid_x - self._LABEL_W - self._GAP_COL - self._PAD_L - 4

        if not self._items_created:
            self._create_items(w, h, mid_x, bar_max_w, tr)
            self._items_created = True
        else:
            # Update static geometry-dependent items
            self.coords(self._divider, mid_x - self._GAP_COL / 2, self._PAD_T,
                        mid_x - self._GAP_COL / 2, h - self._PAD_B)
            self.coords(self._bg_rect, 0, 0, w, h)

        # Update per-metric items
        for i, m in enumerate(self._metrics):
            col = 0 if i < 3 else 1
            row = i if i < 3 else i - 3
            y_mid = self._PAD_T + row * self._ROW_H + self._ROW_H / 2
            y_bar = y_mid - self._BAR_H / 2

            label_x = self._PAD_L if col == 0 else mid_x + self._GAP_COL / 2
            bar_x0 = label_x + self._LABEL_W
            bar_x1 = bar_x0 + bar_max_w

            # Update bar backgrounds and positions
            if m["bar_bg"] is not None:
                self.coords(m["bar_bg"], bar_x0, y_bar, bar_x1, y_bar + self._BAR_H)
                self.itemconfigure(m["bar_bg"], fill="#e8e8e8", outline="#ddd")

            # Bar fill — percentage-based metrics derive frac from current;
            # RAM/VRAM/temp have pre-computed frac from _apply_loop.
            if m["key"] in ("ram", "vram", "temp"):
                frac = m["frac"]
            else:
                frac = m["current"] / 100.0
            frac = max(0.0, min(1.0, frac))
            fill_w = frac * bar_max_w
            if m["bar_fill"] is not None:
                if fill_w > 0.5:
                    self.coords(m["bar_fill"], bar_x0, y_bar + 1,
                                bar_x0 + fill_w, y_bar + self._BAR_H - 1)
                    self.itemconfigure(m["bar_fill"], fill=self._bar_color(frac),
                                       outline="")
                else:
                    self.coords(m["bar_fill"], 0, 0, 0, 0)  # hide

            # Value text
            if m["key"] in ("gpu", "vram", "temp") and not self._gpu_available:
                val_str = tr("mon_no_gpu")
            else:
                val_str = m["fmt"].format(m["current"])
            if m["text"] is not None:
                self.coords(m["text"], bar_x1 + 8, y_mid)
                self.itemconfigure(m["text"], text=val_str)

    def _create_items(self, w: int, h: int, mid_x: float, bar_max_w: float, tr) -> None:
        """Create all canvas items once; _redraw() updates them in-place."""
        self._bg_rect = self.create_rectangle(0, 0, w, h, fill="#f5f5f5", outline="",
                                               tags=("bg",))
        self._divider = self.create_line(
            mid_x - self._GAP_COL / 2, self._PAD_T,
            mid_x - self._GAP_COL / 2, h - self._PAD_B,
            fill="#ddd", width=1, tags=("divider",),
        )

        for i, m in enumerate(self._metrics):
            col = 0 if i < 3 else 1
            row = i if i < 3 else i - 3
            y_mid = self._PAD_T + row * self._ROW_H + self._ROW_H / 2
            y_bar = y_mid - self._BAR_H / 2

            label_x = self._PAD_L if col == 0 else mid_x + self._GAP_COL / 2
            bar_x0 = label_x + self._LABEL_W
            bar_x1 = bar_x0 + bar_max_w

            tag = f"m{i}"
            # Label
            key_tr = f"mon_{m['key']}"
            m["label"] = self.create_text(label_x, y_mid, text=tr(key_tr),
                                          font=self._font(8), fill="#444",
                                          anchor="w", tags=(tag,))
            # Bar background
            m["bar_bg"] = self.create_rectangle(bar_x0, y_bar, bar_x1, y_bar + self._BAR_H,
                                                fill="#e8e8e8", outline="#ddd",
                                                tags=(tag,))
            # Bar fill (initially hidden)
            m["bar_fill"] = self.create_rectangle(0, 0, 0, 0,
                                                  fill="#27ae60", outline="",
                                                  tags=(tag,))
            # Value text
            m["text"] = self.create_text(bar_x1 + 8, y_mid, text="",
                                         font=self._font(9, bold=True),
                                         fill="#333", anchor="w", tags=(tag,))

    @staticmethod
    def _bar_color(frac: float) -> str:
        """Green → yellow → orange → red gradient."""
        frac = max(0.0, min(1.0, frac))
        stops = [(0.0, 0x27, 0xae, 0x60),
                 (0.5, 0xf0, 0xc0, 0x40),
                 (0.75, 0xe6, 0x7e, 0x22),
                 (1.0, 0xc0, 0x39, 0x2b)]
        for i in range(len(stops) - 1):
            t0, r0, g0, b0 = stops[i]
            t1, r1, g1, b1 = stops[i + 1]
            if t0 <= frac <= t1:
                progress = (frac - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = int(r0 + (r1 - r0) * progress)
                g = int(g0 + (g1 - g0) * progress)
                b = int(b0 + (b1 - b0) * progress)
                return f"#{r:02x}{g:02x}{b:02x}"
        return "#c0392b"


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
        # Probabilities are *never* forwarded from a previous position:
        # action_ids can coincidentally match across different positions.
        # They are only shown during ``perform_ai_move``'s pre-move
        # display, where the analysis was just computed for this position.
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
            bundled_ttf = os.path.join(
                str(PROJECT_ROOT), 'assets', 'fonts', 'DroidSansFallbackFull.ttf',
            )
            if os.path.exists(bundled_ttf):
                _install_fontconfig(bundled_ttf)
                self.update()
                families = _get_tk_font_families(self)
                for name in _PREFERRED_CJK_FONTS:
                    if name in families and _cjk_renders_correctly(name, self):
                        cjk_family = name
                        break

        # ── Tier 3 (Linux): install system CJK fonts via X11 font path ──
        # System TrueType fonts like WenQuanYi Micro Hei (sans-serif)
        # often render correctly through X11 even when the bundled Droid
        # font has glyph-metric corruption on XWayland.
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
            bundled_ttf = os.path.join(
                str(PROJECT_ROOT), 'assets', 'fonts', 'DroidSansFallbackFull.ttf',
            )
            if os.path.exists(bundled_ttf):
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
        # Without this, ttk.Button, ttk.Treeview, ttk.Combobox, etc.
        # ignore TkDefaultFont and fall back to a non-CJK system font.
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
        # Using a sub-frame allows notation and log to split 50/50
        # regardless of how the board/search-tree rows divvy up height.
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
        # Each column retains its side's last legal moves and analysis,
        # so probabilities never leak across sides.
        moves_frame = ttk.Frame(sidebar)
        moves_frame.grid(row=3, column=0, sticky="nsew")
        moves_frame.columnconfigure(0, weight=1)
        moves_frame.columnconfigure(1, weight=1)
        moves_frame.rowconfigure(0, weight=1)

        # --- Black column ---
        # tk.Frame wrapper provides a universally-supported highlight border
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
            # value ∈ [-1, 1] from the current player's view.
            # Map linearly to [0 %, 100 %] win probability.
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

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

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

    def ask_checkpoint_path(self, title: str, parent: tk.Toplevel) -> Optional[str]:
        checkpoint_dir = PROJECT_ROOT / "trainer" / "checkpoints"
        initial_dir = checkpoint_dir if checkpoint_dir.exists() else PROJECT_ROOT
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
        config_dir = PROJECT_ROOT / "draughts" / "config"
        initial_dir = config_dir if config_dir.exists() else PROJECT_ROOT
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

        # Section labels


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


def _softmax_list(logits: List[float]) -> List[float]:
    """Numerically stable softmax over a plain Python list."""
    import math as _math

    if not logits:
        return []
    max_logit = max(logits)
    exps = [_math.exp(v - max_logit) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def value_to_win_pct(value: float, side_to_move: int) -> float:
    """Convert a value in [-1, 1] (current-player perspective) to win% for display.

    Returns a number in [0, 100].
    """
    v = max(-1.0, min(1.0, value))
    return (v + 1.0) / 2.0 * 100.0


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




def main() -> None:
    app = DraughtsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
