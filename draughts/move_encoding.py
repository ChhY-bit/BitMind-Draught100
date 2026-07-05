from __future__ import annotations

from typing import Dict, List, Sequence

# 引入底层走法结构
from draughts_env import Move


BOARD_SIZE = 10
SQUARE_COUNT = 50
MAX_PATH_POINTS = 10
MAX_CAPTURES = 9
MOVE_FEATURE_DIM = 4 + 3 + MAX_PATH_POINTS * 2 + MAX_CAPTURES * 2


BLACK_SIDE = 0
WHITE_SIDE = 1


# 将走法字典映射为走法向量
def encode_move(move: Dict[str, object], side_to_move: int = BLACK_SIDE) -> List[float]:
    path = [int(square) for square in move["path"]] # 获取所有当前走法的全部路径
    captures = [int(square) for square in move.get("captures", [])] # 获取当前走法的全部吃子
    # 异常处理：
    if not path:
        raise ValueError("move path must not be empty")
    if len(path) > MAX_PATH_POINTS:
        raise ValueError(f"move path length {len(path)} exceeds {MAX_PATH_POINTS}")
    if len(captures) > MAX_CAPTURES:
        raise ValueError(f"capture count {len(captures)} exceeds {MAX_CAPTURES}")

    # 根据己方/对方灵活处理：
    path = [perspective_square_id(square, side_to_move) for square in path]
    captures = [perspective_square_id(square, side_to_move) for square in captures]

    # 获取起始位置和结束位置：
    start_row, start_col = normalized_square_coord(path[0])
    end_row, end_col = normalized_square_coord(path[-1])

    # 汇总为用于深度学习的走法向量（含归一化）：
    features = [
        start_row,
        start_col,
        end_row,
        end_col,
        len(path) / MAX_PATH_POINTS,
        len(captures) / MAX_CAPTURES,
        1.0 if bool(move.get("is_capture", captures)) else 0.0,
    ]

    features.extend(encode_square_sequence(path, MAX_PATH_POINTS))
    features.extend(encode_square_sequence(captures, MAX_CAPTURES))
    if len(features) != MOVE_FEATURE_DIM:
        raise RuntimeError(f"encoded move has unexpected length: {len(features)}")
    return features


# 生成所有走法的向量列表
def encode_legal_moves(
    moves: Sequence[Dict[str, object]],
    side_to_move: int = BLACK_SIDE,
) -> List[List[float]]:
    return [encode_move(move, side_to_move=side_to_move) for move in moves]


def selected_move_index(
    legal_moves: Sequence[Dict[str, object]],
    selected_move: Dict[str, object],
) -> int:
    selected_path = [int(square) for square in selected_move["path"]]
    selected_captures = [int(square) for square in selected_move.get("captures", [])]
    selected_action_id = selected_move.get("action_id")

    for index, move in enumerate(legal_moves):
        if selected_action_id is not None and move.get("action_id") != selected_action_id:
            continue
        if [int(square) for square in move["path"]] != selected_path:
            continue
        if [int(square) for square in move.get("captures", [])] != selected_captures:
            continue
        return index

    raise ValueError("selected_move was not found in legal_moves")


def encode_square_sequence(square_ids: Sequence[int], max_length: int) -> List[float]:
    encoded: List[float] = []
    for square_id in square_ids:
        row, col = normalized_square_coord(int(square_id))
        encoded.extend([row, col])
    missing = max_length - len(square_ids)
    encoded.extend([0.0] * missing * 2)
    return encoded


def normalized_square_coord(square_id: int) -> tuple[float, float]:
    row, col = square_to_coord(square_id)
    return row / (BOARD_SIZE - 1), col / (BOARD_SIZE - 1)


def square_to_coord(square_id: int) -> tuple[int, int]:
    if square_id < 1 or square_id > SQUARE_COUNT:
        raise ValueError(f"square_id must be in 1..50: {square_id}")
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


def perspective_square_id(square_id: int, side_to_move: int) -> int:
    if square_id < 1 or square_id > SQUARE_COUNT:
        raise ValueError(f"square_id must be in 1..50: {square_id}")
    if side_to_move == BLACK_SIDE:
        return square_id
    if side_to_move == WHITE_SIDE:
        return SQUARE_COUNT + 1 - square_id
    raise ValueError(f"invalid side_to_move: {side_to_move}")


# 将底层走法结构转换为Python字典的格式
def move_to_dict(move: Move) -> Dict[str, object]:
    """Convert a ctypes Move to a plain dict for encoding."""
    return {
        "action_id": move.action_id,
        "path": list(move.path),
        "captures": list(move.captures),
        "is_capture": bool(move.is_capture),
    }
