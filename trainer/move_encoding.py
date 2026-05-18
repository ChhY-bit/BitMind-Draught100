from __future__ import annotations

from typing import Dict, List, Sequence


BOARD_SIZE = 10
SQUARE_COUNT = 50
MAX_PATH_POINTS = 10
MAX_CAPTURES = 9
MOVE_FEATURE_DIM = 4 + 3 + MAX_PATH_POINTS * 2 + MAX_CAPTURES * 2


def encode_move(move: Dict[str, object]) -> List[float]:
    path = [int(square) for square in move["path"]]
    captures = [int(square) for square in move.get("captures", [])]
    if not path:
        raise ValueError("move path must not be empty")
    if len(path) > MAX_PATH_POINTS:
        raise ValueError(f"move path length {len(path)} exceeds {MAX_PATH_POINTS}")
    if len(captures) > MAX_CAPTURES:
        raise ValueError(f"capture count {len(captures)} exceeds {MAX_CAPTURES}")

    start_row, start_col = normalized_square_coord(path[0])
    end_row, end_col = normalized_square_coord(path[-1])
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


def encode_legal_moves(moves: Sequence[Dict[str, object]]) -> List[List[float]]:
    return [encode_move(move) for move in moves]


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
