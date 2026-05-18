import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from inspect_dataset import inspect_paths
from selfplay import generate_dataset


class InspectDatasetTests(unittest.TestCase):
    def test_inspect_generated_dataset(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "selfplay.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=6,
                output_path=str(output),
                log_interval=1,
                seed=1,
            )

            report = inspect_paths([str(output)])

        self.assertEqual(report.files, 1)
        self.assertEqual(report.game_count, 1)
        self.assertGreater(report.samples, 0)
        self.assertEqual(report.samples, report.valid_samples)
        self.assertEqual(report.invalid_samples, 0)
        self.assertEqual(report.invalid_json_lines, 0)
        self.assertIn(0, report.value_targets)

    def test_inspect_detects_invalid_jsonl_sample(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "bad.jsonl"
            bad_sample = {
                "schema_version": 1,
                "game_id": 0,
                "ply": 0,
                "side_to_move": 0,
                "raw_squares": [0] * 50,
                "observation": [[[0] * 10 for _row in range(10)] for _channel in range(6)],
                "legal_action_ids": [0],
                "action_mask": [0] * 512,
                "selected_action_id": 2,
                "final_result": 0,
                "max_plies_reached": True,
                "value_target": 5,
            }
            output.write_text(json.dumps(bad_sample) + "\n{bad json\n", encoding="utf-8")

            report = inspect_paths([str(output)])

        self.assertEqual(report.files, 1)
        self.assertEqual(report.samples, 2)
        self.assertEqual(report.valid_samples, 0)
        self.assertEqual(report.invalid_samples, 1)
        self.assertEqual(report.invalid_json_lines, 1)
        self.assertGreater(report.validation_errors["invalid_value_target"], 0)
        self.assertGreater(report.validation_errors["selected_action_not_legal"], 0)
        self.assertGreater(report.validation_errors["legal_action_missing_in_mask"], 0)


if __name__ == "__main__":
    unittest.main()
