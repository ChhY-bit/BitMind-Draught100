import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from selfplay import (
    BLACK_SIDE,
    BLACK_WIN,
    WHITE_SIDE,
    WHITE_WIN,
    generate_dataset,
    value_target_for_side,
)


class SelfPlayTests(unittest.TestCase):
    def test_value_target_for_side(self):
        self.assertEqual(value_target_for_side(BLACK_WIN, BLACK_SIDE), 1)
        self.assertEqual(value_target_for_side(BLACK_WIN, WHITE_SIDE), -1)
        self.assertEqual(value_target_for_side(WHITE_WIN, BLACK_SIDE), -1)
        self.assertEqual(value_target_for_side(WHITE_WIN, WHITE_SIDE), 1)
        self.assertEqual(value_target_for_side(0, BLACK_SIDE, maxed=True), 0)

    def test_generate_dataset_writes_jsonl_samples(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "selfplay.jsonl"

            stats = generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=6,
                output_path=str(output),
                log_interval=1,
                seed=1,
            )

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.samples, len(lines))
        self.assertGreater(len(lines), 0)

        sample = json.loads(lines[0])
        self.assertEqual(sample["schema_version"], 1)
        self.assertEqual(sample["game_id"], 0)
        self.assertEqual(sample["ply"], 0)
        self.assertIn(sample["side_to_move"], {BLACK_SIDE, WHITE_SIDE})
        self.assertEqual(len(sample["raw_squares"]), 50)
        self.assertEqual(len(sample["observation"]), 6)
        self.assertEqual(len(sample["action_mask"]), 512)
        self.assertIn(sample["selected_action_id"], sample["legal_action_ids"])
        self.assertIn("final_result", sample)
        self.assertIn(sample["value_target"], {-1, 0, 1})
        self.assertIn("selected_move", sample)


if __name__ == "__main__":
    unittest.main()
