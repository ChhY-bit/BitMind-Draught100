from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from draughts.move_encoding import coord_to_square, square_to_coord
from gui.utils import CELL_SIZE
from gui.widgets.board import BoardView


# Transcribed from figure 1 in the supplied replay-software manual.
REFERENCE_BOARD = (
    (None, 1, None, 2, None, 3, None, 4, None, 5),
    (6, None, 7, None, 8, None, 9, None, 10, None),
    (None, 11, None, 12, None, 13, None, 14, None, 15),
    (16, None, 17, None, 18, None, 19, None, 20, None),
    (None, 21, None, 22, None, 23, None, 24, None, 25),
    (26, None, 27, None, 28, None, 29, None, 30, None),
    (None, 31, None, 32, None, 33, None, 34, None, 35),
    (36, None, 37, None, 38, None, 39, None, 40, None),
    (None, 41, None, 42, None, 43, None, 44, None, 45),
    (46, None, 47, None, 48, None, 49, None, 50, None),
)


class BoardNumberingTests(unittest.TestCase):
    def make_board(self, rotated):
        board = SimpleNamespace(rotated=rotated, _on_square_click=Mock(),
                                create_text=Mock(), create_rectangle=Mock(), delete=Mock())
        board._view_coord_from_board = lambda row, col: BoardView._view_coord_from_board(board, row, col)
        board._board_coord_from_view = lambda row, col: BoardView._board_coord_from_view(board, row, col)
        return board

    def test_all_fifty_squares_match_manual_and_light_squares_have_no_id(self):
        for row, line in enumerate(REFERENCE_BOARD):
            for col, square in enumerate(line):
                self.assertEqual(coord_to_square(row, col), square)
                if square is not None:
                    self.assertEqual(square_to_coord(square), (row, col))

    def test_clicks_keep_absolute_ids_in_both_board_orientations(self):
        for rotated in (False, True):
            board = self.make_board(rotated)
            for row, line in enumerate(REFERENCE_BOARD):
                for col, square in enumerate(line):
                    board._on_square_click.reset_mock()
                    view_row, view_col = (9 - row, 9 - col) if rotated else (row, col)
                    event = SimpleNamespace(x=int((view_col + 0.5) * CELL_SIZE),
                                            y=int((view_row + 0.5) * CELL_SIZE))
                    BoardView._handle_click(board, event)
                    if square is None:
                        board._on_square_click.assert_not_called()
                    else:
                        board._on_square_click.assert_called_once_with(square)

    def test_drawn_numbers_match_clicked_squares_after_rotation(self):
        for rotated in (False, True):
            board = self.make_board(rotated)
            BoardView.draw(board, [0] * 50, None, [], [])
            labels = {call.kwargs["text"]: call.args for call in board.create_text.call_args_list}
            self.assertEqual(set(labels), {str(square) for square in range(1, 51)})
            for row, line in enumerate(REFERENCE_BOARD):
                for col, square in enumerate(line):
                    if square is not None:
                        view_row, view_col = (9 - row, 9 - col) if rotated else (row, col)
                        self.assertEqual(labels[str(square)], (view_col * CELL_SIZE + 8,
                                                              view_row * CELL_SIZE + 8))


if __name__ == "__main__":
    unittest.main()
