import unittest

from move_encoding import (
    WHITE_SIDE,
    MAX_CAPTURES,
    MAX_PATH_POINTS,
    MOVE_FEATURE_DIM,
    encode_move,
    selected_move_index,
    perspective_square_id,
    square_to_coord,
)


class MoveEncodingTests(unittest.TestCase):
    def test_square_to_coord_matches_board_numbering(self):
        self.assertEqual(square_to_coord(1), (0, 1))
        self.assertEqual(square_to_coord(6), (1, 0))
        self.assertEqual(square_to_coord(50), (9, 8))

    def test_encode_quiet_move(self):
        move = {"action_id": 0, "path": [16, 21], "captures": [], "is_capture": False}

        features = encode_move(move)

        self.assertEqual(len(features), MOVE_FEATURE_DIM)
        self.assertEqual(features[4], 2 / MAX_PATH_POINTS)
        self.assertEqual(features[5], 0)
        self.assertEqual(features[6], 0)

    def test_encode_capture_move(self):
        move = {
            "action_id": 3,
            "path": [32, 23, 14],
            "captures": [28, 19],
            "is_capture": True,
        }

        features = encode_move(move)

        self.assertEqual(len(features), MOVE_FEATURE_DIM)
        self.assertEqual(features[4], 3 / MAX_PATH_POINTS)
        self.assertEqual(features[5], 2 / MAX_CAPTURES)
        self.assertEqual(features[6], 1)

    def test_white_move_is_encoded_from_current_side_perspective(self):
        black_move = {"action_id": 0, "path": [16, 21], "captures": [], "is_capture": False}
        white_move = {"action_id": 0, "path": [35, 30], "captures": [], "is_capture": False}

        self.assertEqual(encode_move(black_move), encode_move(white_move, side_to_move=WHITE_SIDE))

    def test_perspective_square_id_rotates_white_side(self):
        self.assertEqual(perspective_square_id(1, WHITE_SIDE), 50)
        self.assertEqual(perspective_square_id(50, WHITE_SIDE), 1)

    def test_selected_move_index_uses_action_path_and_captures(self):
        legal_moves = [
            {"action_id": 0, "path": [16, 21], "captures": [], "is_capture": False},
            {"action_id": 1, "path": [32, 23], "captures": [28], "is_capture": True},
        ]

        self.assertEqual(selected_move_index(legal_moves, legal_moves[1]), 1)

    def test_encode_rejects_too_long_move(self):
        move = {
            "action_id": 9,
            "path": list(range(1, MAX_PATH_POINTS + 2)),
            "captures": [],
            "is_capture": False,
        }

        with self.assertRaises(ValueError):
            encode_move(move)


if __name__ == "__main__":
    unittest.main()
