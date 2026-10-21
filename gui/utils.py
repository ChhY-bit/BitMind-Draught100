from __future__ import annotations

import math as _math
from typing import Dict, List, Optional, Sequence

# C-level Move type — imported lazily by agents to avoid circular deps.
# For type-checking convenience we re-export the name here.
from draughts_env import Move  # noqa: F401
from draughts.notation import format_move_notation

# ---------------------------------------------------------------------------
# Board geometry constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 10
PLAYABLE_SQUARES = 50
CELL_SIZE = 68
BOARD_PIXELS = BOARD_SIZE * CELL_SIZE

# ---------------------------------------------------------------------------
# Piece and side constants
# ---------------------------------------------------------------------------

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
WHITE_MAN = 3
WHITE_KING = 4

BLACK_SIDE = 0
WHITE_SIDE = 1


# ---------------------------------------------------------------------------
# Math / misc helpers
# ---------------------------------------------------------------------------


def _softmax_list(logits: List[float]) -> List[float]:
    """Numerically stable softmax over a plain Python list."""
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


# ---------------------------------------------------------------------------
# Move / board helpers
# ---------------------------------------------------------------------------


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
    return f"{(turn_index + 1) // 2}. {format_move_notation(move)}"
