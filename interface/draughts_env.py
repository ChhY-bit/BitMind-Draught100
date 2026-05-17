from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DRAUGHTS_SQUARE_COUNT = 50
DRAUGHTS_MAX_MOVE_PATH = 32
DRAUGHTS_MAX_MOVE_CAPTURES = 20
DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES = 512

DRAUGHTS_OK = 0
DRAUGHTS_GAME_ONGOING = 0
DRAUGHTS_GAME_BLACK_WIN = 1
DRAUGHTS_GAME_WHITE_WIN = 2
DRAUGHTS_GAME_DRAW = 3

PIECE_NAMES = {
    0: "empty",
    1: "black_man",
    2: "black_king",
    3: "white_man",
    4: "white_king",
}

SIDE_NAMES = {
    0: "black",
    1: "white",
}

RESULT_NAMES = {
    DRAUGHTS_GAME_ONGOING: "ongoing",
    DRAUGHTS_GAME_BLACK_WIN: "black_win",
    DRAUGHTS_GAME_WHITE_WIN: "white_win",
    DRAUGHTS_GAME_DRAW: "draw",
}


class _DraughtsState(ctypes.Structure):
    _fields_ = [
        ("squares", ctypes.c_ubyte * DRAUGHTS_SQUARE_COUNT),
        ("side_to_move", ctypes.c_ubyte),
    ]


class _DraughtsMove(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_ubyte * DRAUGHTS_MAX_MOVE_PATH),
        ("path_len", ctypes.c_ubyte),
        ("captures", ctypes.c_ubyte * DRAUGHTS_MAX_MOVE_CAPTURES),
        ("capture_count", ctypes.c_ubyte),
    ]


class _DraughtsEnvHandle(ctypes.Structure):
    _fields_ = [
        ("state", _DraughtsState),
        ("legal_moves", _DraughtsMove * DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES),
        ("legal_move_count", ctypes.c_int),
        ("game_result", ctypes.c_int),
    ]


class _DraughtsObservation(ctypes.Structure):
    _fields_ = [
        ("squares", ctypes.c_ubyte * DRAUGHTS_SQUARE_COUNT),
        ("side_to_move", ctypes.c_ubyte),
        ("game_result", ctypes.c_int),
        ("legal_move_count", ctypes.c_int),
    ]


class _DraughtsStepResult(ctypes.Structure):
    _fields_ = [
        ("reward", ctypes.c_int),
        ("done", ctypes.c_int),
        ("game_result", ctypes.c_int),
        ("legal_move_count", ctypes.c_int),
    ]


@dataclass(frozen=True)
class Move:
    action_id: int
    path: Tuple[int, ...]
    captures: Tuple[int, ...]

    @property
    def is_capture(self) -> bool:
        return len(self.captures) > 0


def _default_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / "libdraughts.so"


def _load_library(library_path: Optional[str]) -> ctypes.CDLL:
    path = Path(library_path) if library_path is not None else _default_library_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find C shared library: {path}. Run `make interface` from the project root first."
        )

    lib = ctypes.CDLL(str(path))

    lib.draughts_env_reset.argtypes = [
        ctypes.POINTER(_DraughtsEnvHandle),
        ctypes.POINTER(_DraughtsObservation),
    ]
    lib.draughts_env_reset.restype = ctypes.c_int

    lib.draughts_env_get_observation.argtypes = [
        ctypes.POINTER(_DraughtsEnvHandle),
        ctypes.POINTER(_DraughtsObservation),
    ]
    lib.draughts_env_get_observation.restype = ctypes.c_int

    lib.draughts_env_get_legal_move_count.argtypes = [
        ctypes.POINTER(_DraughtsEnvHandle),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.draughts_env_get_legal_move_count.restype = ctypes.c_int

    lib.draughts_env_get_legal_move.argtypes = [
        ctypes.POINTER(_DraughtsEnvHandle),
        ctypes.c_int,
        ctypes.POINTER(_DraughtsMove),
    ]
    lib.draughts_env_get_legal_move.restype = ctypes.c_int

    lib.draughts_env_step.argtypes = [
        ctypes.POINTER(_DraughtsEnvHandle),
        ctypes.c_int,
        ctypes.POINTER(_DraughtsObservation),
        ctypes.POINTER(_DraughtsStepResult),
    ]
    lib.draughts_env_step.restype = ctypes.c_int

    return lib


def _check_result(result: int, function_name: str) -> None:
    if result != DRAUGHTS_OK:
        raise RuntimeError(f"{function_name} failed with C error code {result}")


def _move_from_c(action_id: int, move: _DraughtsMove) -> Move:
    path = tuple(int(move.path[i]) for i in range(int(move.path_len)))
    captures = tuple(int(move.captures[i]) for i in range(int(move.capture_count)))
    return Move(action_id=action_id, path=path, captures=captures)


def _observation_to_dict(observation: _DraughtsObservation) -> Dict[str, object]:
    side = int(observation.side_to_move)
    game_result = int(observation.game_result)
    return {
        "squares": [int(observation.squares[i]) for i in range(DRAUGHTS_SQUARE_COUNT)],
        "side_to_move": side,
        "side_to_move_name": SIDE_NAMES.get(side, "unknown"),
        "game_result": game_result,
        "game_result_name": RESULT_NAMES.get(game_result, "unknown"),
        "legal_move_count": int(observation.legal_move_count),
    }


class DraughtsEnv:
    """Small Python wrapper around the C draughts environment interface."""

    def __init__(self, library_path: Optional[str] = None) -> None:
        self._lib = _load_library(library_path)
        self._env = _DraughtsEnvHandle()
        self.reset()

    def reset(self) -> Dict[str, object]:
        observation = _DraughtsObservation()
        result = self._lib.draughts_env_reset(
            ctypes.byref(self._env),
            ctypes.byref(observation),
        )
        _check_result(result, "draughts_env_reset")
        return _observation_to_dict(observation)

    def observation(self) -> Dict[str, object]:
        observation = _DraughtsObservation()
        result = self._lib.draughts_env_get_observation(
            ctypes.byref(self._env),
            ctypes.byref(observation),
        )
        _check_result(result, "draughts_env_get_observation")
        return _observation_to_dict(observation)

    def legal_move_count(self) -> int:
        move_count = ctypes.c_int()
        result = self._lib.draughts_env_get_legal_move_count(
            ctypes.byref(self._env),
            ctypes.byref(move_count),
        )
        _check_result(result, "draughts_env_get_legal_move_count")
        return int(move_count.value)

    def legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        for action_id in range(self.legal_move_count()):
            c_move = _DraughtsMove()
            result = self._lib.draughts_env_get_legal_move(
                ctypes.byref(self._env),
                action_id,
                ctypes.byref(c_move),
            )
            _check_result(result, "draughts_env_get_legal_move")
            moves.append(_move_from_c(action_id, c_move))
        return moves

    def step(self, action_id: int) -> Tuple[Dict[str, object], Dict[str, int]]:
        observation = _DraughtsObservation()
        step_result = _DraughtsStepResult()
        result = self._lib.draughts_env_step(
            ctypes.byref(self._env),
            int(action_id),
            ctypes.byref(observation),
            ctypes.byref(step_result),
        )
        _check_result(result, "draughts_env_step")

        info = {
            "reward": int(step_result.reward),
            "done": int(step_result.done),
            "game_result": int(step_result.game_result),
            "legal_move_count": int(step_result.legal_move_count),
        }
        return _observation_to_dict(observation), info

    def render_text(self) -> str:
        observation = self.observation()
        squares = observation["squares"]
        rows = []
        for row in range(10):
            cells = []
            for col in range(10):
                if (row + col) % 2 == 0:
                    cells.append(" .")
                else:
                    square_id = row * 5 + (col // 2) + 1
                    piece = squares[square_id - 1]
                    cells.append(_piece_char(piece).rjust(2))
            rows.append(" ".join(cells))
        return "\n".join(rows)


def _piece_char(piece: int) -> str:
    if piece == 1:
        return "b"
    if piece == 2:
        return "B"
    if piece == 3:
        return "w"
    if piece == 4:
        return "W"
    return "-"
