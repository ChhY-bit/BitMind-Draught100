from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_DIR = PROJECT_ROOT / "interface"
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))

from draughts_env import (  # noqa: E402
    DRAUGHTS_GAME_ONGOING,
    DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES,
    DraughtsEnv,
    Move,
)


BOARD_SIZE = 10
SQUARE_COUNT = 50
OBSERVATION_CHANNELS = 6

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
WHITE_MAN = 3
WHITE_KING = 4

BLACK_SIDE = 0
WHITE_SIDE = 1


Observation = List[List[List[int]]]


class StandardDraughtsEnv:
    """Training-friendly wrapper around the C-backed Python draughts interface."""

    def __init__(
        self,
        library_path: Optional[str] = None,
        max_actions: int = DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")

        self._library_path = library_path
        self._env = DraughtsEnv(library_path)
        self.max_actions = max_actions
        self._last_observation_dict = self._env.observation()
        self._action_history: List[int] = []

    def reset(self) -> Observation:
        self._last_observation_dict = self._env.reset()
        self._action_history = []
        return self.observation()

    def observation(self) -> Observation:
        self._last_observation_dict = self._env.observation()
        return encode_observation(self._last_observation_dict)

    def raw_observation(self) -> Dict[str, object]:
        self._last_observation_dict = self._env.observation()
        return dict(self._last_observation_dict)

    def legal_actions(self) -> List[Move]:
        return self._env.legal_moves()

    def action_mask(self) -> List[int]:
        legal_count = self._env.legal_move_count()
        if legal_count > self.max_actions:
            raise RuntimeError(
                f"legal action count {legal_count} exceeds max_actions {self.max_actions}"
            )
        return [1 if index < legal_count else 0 for index in range(self.max_actions)]

    def step(self, action_id: int) -> Tuple[Observation, int, bool, Dict[str, object]]:
        observation_dict, step_info = self._env.step(action_id)
        self._last_observation_dict = observation_dict
        self._action_history.append(int(action_id))

        observation = encode_observation(observation_dict)
        reward = int(step_info["reward"])
        done = bool(step_info["done"])
        info = {
            "reward": reward,
            "done": done,
            "game_result": int(step_info["game_result"]),
            "game_result_name": observation_dict["game_result_name"],
            "side_to_move": int(observation_dict["side_to_move"]),
            "side_to_move_name": observation_dict["side_to_move_name"],
            "legal_move_count": int(step_info["legal_move_count"]),
            "legal_actions": self.legal_actions() if not done else [],
        }
        return observation, reward, done, info

    def is_terminal(self) -> bool:
        return int(self._env.observation()["game_result"]) != DRAUGHTS_GAME_ONGOING

    def render_text(self) -> str:
        return self._env.render_text()

    def clone(self) -> "StandardDraughtsEnv":
        cloned = self.__class__.__new__(self.__class__)
        cloned._library_path = self._library_path
        cloned._env = self._env.clone()
        cloned.max_actions = self.max_actions
        cloned._last_observation_dict = cloned._env.observation()
        cloned._action_history = list(self._action_history)
        return cloned

    def action_history(self) -> List[int]:
        return list(self._action_history)


def encode_observation(observation: Dict[str, object]) -> Observation:
    side_to_move = int(observation["side_to_move"])
    squares = observation["squares"]
    encoded = empty_observation()

    for square_id in range(1, SQUARE_COUNT + 1):
        row, col = square_to_coord(square_id)
        piece = int(squares[square_id - 1])
        encoded[4][row][col] = 1
        encoded[5][row][col] = side_to_move

        if piece == EMPTY:
            continue
        channel = piece_channel(piece, side_to_move)
        encoded[channel][row][col] = 1

    return encoded


def empty_observation() -> Observation:
    return [
        [[0 for _col in range(BOARD_SIZE)] for _row in range(BOARD_SIZE)]
        for _channel in range(OBSERVATION_CHANNELS)
    ]


def piece_channel(piece: int, side_to_move: int) -> int:
    if side_to_move == BLACK_SIDE:
        if piece == BLACK_MAN:
            return 0
        if piece == BLACK_KING:
            return 1
        if piece == WHITE_MAN:
            return 2
        if piece == WHITE_KING:
            return 3
    else:
        if piece == WHITE_MAN:
            return 0
        if piece == WHITE_KING:
            return 1
        if piece == BLACK_MAN:
            return 2
        if piece == BLACK_KING:
            return 3

    raise ValueError(f"invalid piece or side: piece={piece}, side={side_to_move}")


def square_to_coord(square_id: int) -> Tuple[int, int]:
    if square_id < 1 or square_id > SQUARE_COUNT:
        raise ValueError(f"square_id must be in 1..50: {square_id}")

    zero_based = square_id - 1
    row = zero_based // 5
    offset = zero_based % 5
    col = offset * 2 + (1 if row % 2 == 0 else 0)
    return row, col
