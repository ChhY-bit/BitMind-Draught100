from __future__ import annotations

import tkinter as tk
from typing import Optional, Sequence

from draughts.move_encoding import coord_to_square, square_to_coord

from ..utils import (
    BLACK_KING,
    BLACK_MAN,
    BOARD_PIXELS,
    BOARD_SIZE,
    CELL_SIZE,
    EMPTY,
    WHITE_KING,
    WHITE_MAN,
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
