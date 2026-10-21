from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

from draughts_env import Move


RESULT_CODES = {"*", "1-0", "0-1", "1/2-1/2"}


def result_code(game_result: int) -> str:
    """Map engine results to first-player/second-player scores (black starts)."""
    try:
        return {0: "*", 1: "1-0", 2: "0-1", 3: "1/2-1/2"}[game_result]
    except KeyError as exc:
        raise ValueError(f"invalid engine game result: {game_result}") from exc


def format_move_notation(move: Move) -> str:
    """Format the complete path, without local action IDs or capture notes."""
    separator = "x" if move.is_capture else "-"
    return separator.join(f"{square:02d}" for square in move.path)


def format_move_pairs(moves: Sequence[Move]) -> list[str]:
    """One numbered line per pair of plies; retain a final unpaired move."""
    return [
        f"{index // 2 + 1}. "
        + " ".join(format_move_notation(move) for move in moves[index:index + 2])
        for index in range(0, len(moves), 2)
    ]


@dataclass
class GameRecord:
    """A record of actual moves, independent of GUI language and board view."""

    played_on: date = field(default_factory=date.today)
    first_player: str = "先手方"
    second_player: str = "后手方"
    moves: list[Move] = field(default_factory=list)
    result: str = "*"

    def move_lines(self) -> list[str]:
        return format_move_pairs(self.moves)

    def snapshot(self) -> GameRecord:
        return replace(self, moves=list(self.moves))

    def to_text(self) -> str:
        if self.result not in RESULT_CODES:
            raise ValueError(f"invalid game result: {self.result}")
        for name in (self.first_player, self.second_player):
            if not name.strip() or any(character in name for character in "\r\n"):
                raise ValueError("player names must be non-empty single lines")
        headers = [
            f"时间：{self.played_on:%Y.%m.%d}",
            "先手：1",
            f"先手方：{self.first_player.strip()}",
            f"后手方：{self.second_player.strip()}",
            f"结果：{self.result}",
        ]
        return "\n" + "\n".join(headers) + "\n\n" + "\n".join(
            [*self.move_lines(), self.result]
        ) + "\n"

    def write(self, path: str | Path) -> None:
        """Use UTF-8 with BOM and Windows line endings like the examples."""
        content = self.to_text().replace("\n", "\r\n").encode("utf-8-sig")
        destination = Path(path)
        temporary_path = None
        try:
            with NamedTemporaryFile(dir=destination.parent, prefix=".notation-", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
