import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset import (
    DraughtsJsonlDataset,
    batched,
    collate_batch,
    iter_jsonl_samples,
    resolve_jsonl_files,
)
from move_encoding import MOVE_FEATURE_DIM
from selfplay import generate_dataset


class DatasetTests(unittest.TestCase):
    def test_dataset_reads_selfplay_jsonl(self):
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

            dataset = DraughtsJsonlDataset([str(output)])

        self.assertGreater(len(dataset), 0)
        item = dataset[0]
        self.assertEqual(len(item["observation"]), 6)
        self.assertEqual(len(item["action_mask"]), 512)
        self.assertIn(item["value_target"], {-1, 0, 1})
        self.assertIsInstance(item["policy_target"], int)
        self.assertGreater(len(item["move_features"]), 0)
        self.assertEqual(len(item["move_features"][0]), MOVE_FEATURE_DIM)
        self.assertIsInstance(item["policy_target_index"], int)
        self.assertIn("game_id", item["metadata"])

    def test_resolve_directory_and_iter_samples(self):
        with TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            output = directory / "nested" / "selfplay.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(output),
                log_interval=1,
                seed=1,
            )

            files = resolve_jsonl_files([str(directory)])
            samples = list(iter_jsonl_samples([str(directory)]))

        self.assertEqual(len(files), 1)
        self.assertEqual(len(samples), 4)

    def test_collate_and_batched(self):
        items = [
            {
                "observation": [[[0]]],
                "action_mask": [1, 0],
                "policy_target": 0,
                "move_features": [[0.1] * MOVE_FEATURE_DIM],
                "policy_target_index": 0,
                "value_target": 1,
                "metadata": {"ply": 0},
            },
            {
                "observation": [[[1]]],
                "action_mask": [0, 1],
                "policy_target": 1,
                "move_features": [[0.2] * MOVE_FEATURE_DIM, [0.3] * MOVE_FEATURE_DIM],
                "policy_target_index": 1,
                "value_target": -1,
                "metadata": {"ply": 1},
            },
        ]

        batch = collate_batch(items)
        batches = list(batched(items, 1))

        self.assertEqual(batch["policy_target"], [0, 1])
        self.assertEqual(len(batch["move_features"]), 2)
        self.assertEqual(len(batch["move_features"][0]), 2)
        self.assertEqual(batch["move_mask"], [[1, 0], [1, 1]])
        self.assertEqual(batch["policy_target_index"], [0, 1])
        self.assertEqual(batch["value_target"], [1, -1])
        self.assertEqual(len(batches), 2)

    def test_dataset_validation_rejects_bad_mask(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "bad.jsonl"
            bad_sample = {
                "observation": [[[0 for _col in range(10)] for _row in range(10)] for _c in range(6)],
                "action_mask": [0] * 512,
                "selected_action_id": 0,
                "value_target": 1,
                "game_id": 0,
                "ply": 0,
                "side_to_move": 0,
                "agent": "test",
                "final_result": 1,
                "max_plies_reached": False,
            }
            output.write_text(__import__("json").dumps(bad_sample), encoding="utf-8")

            with self.assertRaises(ValueError):
                DraughtsJsonlDataset([str(output)])


if __name__ == "__main__":
    unittest.main()
