import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from draughts.dataset import DraughtsJsonlDataset
from search_improve import improve_dataset
from selfplay import generate_dataset


class SearchImproveTests(unittest.TestCase):
    def test_improve_dataset_writes_trainable_jsonl(self):
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.jsonl"
            improved = Path(tmp_dir) / "improved.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(source),
                log_interval=1,
                seed=1,
            )

            stats = improve_dataset(
                input_paths=[str(source)],
                output_path=str(improved),
                max_samples=2,
                log_interval=1,
                seed=2,
                alphabeta_depth=1,
            )
            lines = improved.read_text(encoding="utf-8").splitlines()
            sample = json.loads(lines[0])
            dataset = DraughtsJsonlDataset([str(improved)])
            item = dataset[0]

        self.assertEqual(stats.samples_written, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(sample["agent"], "search_improved")
        self.assertEqual(sample["teacher_agent"], "alphabeta")
        self.assertEqual(sample["teacher_depth"], 1)
        self.assertIn(sample["selected_action_id"], sample["legal_action_ids"])
        self.assertEqual(len(sample["search_scores"]), len(sample["legal_moves"]))
        self.assertIn("policy_target_index", item)
        # Verify soft targets are present and valid.
        self.assertIn("policy_target_probs", sample)
        probs = sample["policy_target_probs"]
        self.assertEqual(len(probs), len(sample["legal_moves"]))
        self.assertAlmostEqual(sum(probs), 1.0, places=4)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
        # The selected move should have the highest probability.
        selected_index = sample["legal_action_ids"].index(sample["selected_action_id"])
        self.assertAlmostEqual(probs[selected_index], max(probs), places=6)

    def test_improve_dataset_produces_soft_target_distribution(self):
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.jsonl"
            improved = Path(tmp_dir) / "improved.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(source),
                log_interval=1,
                seed=1,
            )

            improve_dataset(
                input_paths=[str(source)],
                output_path=str(improved),
                max_samples=2,
                log_interval=1,
                seed=2,
                alphabeta_depth=1,
                temperature=50.0,
            )
            sample = json.loads(improved.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(sample["teacher_temperature"], 50.0)
        probs = sample["policy_target_probs"]
        # A single legal move should have prob 1.0.
        if len(sample["legal_moves"]) == 1:
            self.assertAlmostEqual(probs[0], 1.0, places=6)

    def test_improve_dataset_can_filter_by_score_margin(self):
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.jsonl"
            improved = Path(tmp_dir) / "improved.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(source),
                log_interval=1,
                seed=1,
            )

            stats = improve_dataset(
                input_paths=[str(source)],
                output_path=str(improved),
                max_samples=2,
                log_interval=1,
                seed=2,
                alphabeta_depth=1,
                min_score_margin=1_000_000_000,
            )
            lines = improved.read_text(encoding="utf-8").splitlines()

        self.assertEqual(stats.samples_written, 0)
        self.assertGreater(stats.filtered, 0)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
