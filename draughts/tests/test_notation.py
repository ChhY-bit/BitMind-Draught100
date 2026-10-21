from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from draughts.notation import GameRecord, format_move_notation, format_move_pairs, result_code
from draughts_env import DraughtsEnv, Move


FIXTURES = Path(__file__).parent / "fixtures"


class NotationTests(unittest.TestCase):
    def test_two_digit_squares_and_complete_capture_path(self):
        self.assertEqual(format_move_notation(Move(31, (3, 9), ())), "03-09")
        self.assertEqual(format_move_notation(Move(8, (29, 38, 27, 18), (33, 32, 22))),
                         "29x38x27x18")

    def test_move_pairs_keep_white_absolute_squares_and_unpaired_black_move(self):
        moves = [Move(0, (20, 24), ()), Move(7, (33, 29), ()), Move(0, (24, 33), (29,))]
        self.assertEqual(format_move_pairs(moves), ["1. 20-24 33-29", "2. 24x33"])

    def test_default_headers_and_unfinished_result(self):
        text = GameRecord(played_on=date(2026, 10, 18)).to_text()
        self.assertEqual(text, "\n时间：2026.10.18\n先手：1\n先手方：先手方\n后手方：后手方\n结果：*\n\n*\n")

    def test_result_mapping(self):
        self.assertEqual([result_code(result) for result in range(4)],
                         ["*", "1-0", "0-1", "1/2-1/2"])
        with self.assertRaises(ValueError):
            result_code(999)

    def test_both_examples_replay_and_export_every_path_unchanged(self):
        for index, plies in ((1, 108), (2, 122)):
            with self.subTest(example=index):
                source = (FIXTURES / f"notation_example_{index}.txt").read_text(encoding="utf-8")
                headers = dict(line.split("：", 1) for line in source.splitlines() if "：" in line)
                record = GameRecord(
                    played_on=date.fromisoformat(headers["时间"].replace(".", "-")),
                    first_player=headers["先手方"], second_player=headers["后手方"],
                )
                env = DraughtsEnv()
                for line in source.splitlines():
                    if not re.match(r"^\d+\. ", line):
                        continue
                    for token in line.split()[1:]:
                        path = tuple(int(square) for square in re.split("[-x]", token))
                        matches = [move for move in env.legal_moves() if move.path == path]
                        self.assertEqual(len(matches), 1, token)
                        move = matches[0]
                        self.assertEqual(format_move_notation(move), token)
                        record.moves.append(move)
                        env.step(move.action_id)
                self.assertEqual(len(record.moves), plies)
                self.assertEqual(env.observation()["game_result_name"], "white_win")
                record.result = result_code(int(env.observation()["game_result"]))
                # The supplied examples label a second-player win as 1-0.
                # The user confirmed exporting actual results as 0-1 instead.
                expected = source.replace("结果：1-0", "结果：0-1").rstrip()
                expected = expected[:-3] + "0-1"
                self.assertEqual(record.to_text().rstrip(), expected)

    def test_snapshot_does_not_follow_later_moves_or_results(self):
        record = GameRecord(moves=[Move(0, (20, 24), ())])
        snapshot = record.snapshot()
        record.moves.append(Move(0, (33, 29), ()))
        record.result = "1-0"
        self.assertEqual(snapshot.move_lines(), ["1. 20-24"])
        self.assertEqual(snapshot.result, "*")

    def test_write_uses_utf8_bom_crlf_and_cleans_temporary_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "棋谱.txt"
            record = GameRecord()
            record.write(path)
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf\r\n"))
            self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
            self.assertEqual(raw.decode("utf-8-sig").replace("\r\n", "\n"), record.to_text())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_replace_preserves_existing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "棋谱.txt"
            path.write_text("previous record", encoding="utf-8")
            with patch("draughts.notation.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    GameRecord().write(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous record")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_metadata_does_not_overwrite_existing_record(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "棋谱.txt"
            path.write_text("previous record", encoding="utf-8")
            for record in (GameRecord(first_player=""), GameRecord(second_player="a\nb"),
                           GameRecord(result="unknown")):
                with self.subTest(record=record):
                    with self.assertRaises(ValueError):
                        record.write(path)
                    self.assertEqual(path.read_text(encoding="utf-8"), "previous record")


if __name__ == "__main__":
    unittest.main()
