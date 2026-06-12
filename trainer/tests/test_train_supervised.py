import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from selfplay import generate_dataset
from trainer.train_supervised import (
    compute_loss,
    load_data_source_config,
    load_supervised_config,
    main,
    parse_data_source_arg,
    soft_cross_entropy,
    split_indices,
)


class TrainSupervisedTests(unittest.TestCase):
    def test_load_supervised_config(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "supervised.yaml"
            config_path.write_text(
                "epochs: 2\nbatch_size: 4\nlearning_rate: 0.01\n"
                "value_loss_weight: 0.5\nvalidation_split: 0.25\n"
                "seed: 3\nshuffle: 1\nlog_interval: 2\n"
                "csv_log_interval_epochs: 2\nnum_workers: 0\n",
                encoding="utf-8",
            )

            config = load_supervised_config(config_path)

        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 4)
        self.assertAlmostEqual(config.learning_rate, 0.01)
        self.assertAlmostEqual(config.value_loss_weight, 0.5)
        self.assertEqual(config.csv_log_interval_epochs, 2)

    def test_split_indices_keeps_training_set(self):
        train_indices, val_indices = split_indices(3, 0.5, 1)

        self.assertGreater(len(train_indices), 0)
        self.assertGreater(len(val_indices), 0)
        self.assertEqual(sorted(train_indices + val_indices), [0, 1, 2])

    def test_parse_weighted_data_sources(self):
        source = parse_data_source_arg("data/ab.jsonl:1.0:0.5:ab")

        self.assertEqual(source.path, "data/ab.jsonl")
        self.assertEqual(source.policy_weight, 1.0)
        self.assertEqual(source.value_weight, 0.5)
        self.assertEqual(source.name, "ab")

    def test_load_data_source_config(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data_sources.yaml"
            path.write_text(
                "source_1_path: data/ab.jsonl\n"
                "source_1_policy_weight: 1.0\n"
                "source_1_value_weight: 1.0\n"
                "source_1_name: ab\n"
                "source_2_path: data/sl.jsonl\n"
                "source_2_policy_weight: 0.2\n"
                "source_2_value_weight: 1.0\n"
                "source_2_name: sl\n",
                encoding="utf-8",
            )

            sources = load_data_source_config(path)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].name, "ab")
        self.assertEqual(sources[1].policy_weight, 0.2)

    def test_compute_loss_uses_policy_and_value_weights(self):
        output = {
            "policy_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
            "value": torch.tensor([0.0, 0.0], requires_grad=True),
        }
        batch = {
            "policy_target_index": torch.tensor([0, 0]),
            "value_target": torch.tensor([1.0, -1.0]),
            "policy_weight": torch.tensor([1.0, 0.0]),
            "value_weight": torch.tensor([0.0, 1.0]),
        }

        loss, policy_loss, value_loss = compute_loss(output, batch, value_loss_weight=0.5)

        self.assertLess(policy_loss.item(), 0.2)
        self.assertAlmostEqual(value_loss.item(), 1.0, places=5)
        self.assertAlmostEqual(loss.item(), policy_loss.item() + 0.5, places=5)

    def test_compute_loss_with_soft_targets(self):
        output = {
            "policy_logits": torch.tensor([[2.0, 1.0], [0.5, 2.0]], requires_grad=True),
            "value": torch.tensor([0.0, 0.0], requires_grad=True),
        }
        batch = {
            "policy_target_index": torch.tensor([0, 1]),
            "policy_target_probs": torch.tensor([[0.9, 0.1], [0.3, 0.7]]),
            "move_mask": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            "value_target": torch.tensor([1.0, -1.0]),
            "policy_weight": torch.tensor([1.0, 1.0]),
            "value_weight": torch.tensor([1.0, 1.0]),
        }

        loss, policy_loss, value_loss = compute_loss(output, batch, value_loss_weight=0.25)

        # Soft CE loss should be lower when model matches the distribution.
        self.assertGreater(policy_loss.item(), 0.0)
        # Hard-label accuracy check: argmax should still match.
        predictions = output["policy_logits"].argmax(dim=1)
        self.assertEqual(predictions[0].item(), 0)
        self.assertEqual(predictions[1].item(), 1)

    def test_soft_cross_entropy_matches_hard_label_when_onehot(self):
        logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
        # One-hot targets.
        target_probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        move_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        soft_loss = soft_cross_entropy(logits, target_probs, move_mask)
        hard_loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor([0, 2]), reduction="none"
        )

        torch.testing.assert_close(soft_loss, hard_loss)

    def test_soft_cross_entropy_masks_padded_positions(self):
        # Real scenario: model forward() already fills padded positions
        # with MASKED_LOGIT (-1e9), and target_probs are 0 there.
        MASKED_LOGIT = -1.0e9
        logits = torch.tensor([[3.0, float(MASKED_LOGIT)]])
        target_probs = torch.tensor([[0.8, 0.0]])  # padded probs = 0
        move_mask = torch.tensor([[1.0, 0.0]])

        loss = soft_cross_entropy(logits, target_probs, move_mask)

        # After softmax([3.0, -1e9]), the padded position gets ~0 probability.
        # loss ≈ -(0.8 * log_softmax(3.0) + 0.0 * log_softmax(-1e9))
        # log_softmax(3.0) ≈ 0.0 (since exp(3) dominates)
        self.assertLess(loss.item(), 0.01)

    def test_soft_cross_entropy_ignores_padded_target_probs(self):
        # When target_probs for a padded position is accidentally non-zero
        # (shouldn't happen in production, but safety net kicks in).
        MASKED_LOGIT = -1.0e9
        logits = torch.tensor([[3.0, float(MASKED_LOGIT)]])
        # target_probs has 0.2 at padded position — would give NaN without masking.
        target_probs = torch.tensor([[0.8, 0.2]])
        move_mask = torch.tensor([[1.0, 0.0]])

        loss = soft_cross_entropy(logits, target_probs, move_mask)

        # safe_log_probs zeros out the padded position's -inf/-large log_prob,
        # so 0.2 * 0 = 0 for that entry. Loss ≈ 0.
        self.assertLess(loss.item(), 0.01)

    def test_compute_loss_falls_back_to_hard_label_without_soft_targets(self):
        output = {
            "policy_logits": torch.tensor([[2.0, 0.0]], requires_grad=True),
            "value": torch.tensor([0.0], requires_grad=True),
        }
        batch = {
            "policy_target_index": torch.tensor([0]),
            "value_target": torch.tensor([1.0]),
            "policy_weight": torch.tensor([1.0]),
            "value_weight": torch.tensor([1.0]),
        }

        loss, policy_loss, value_loss = compute_loss(output, batch, value_loss_weight=0.25)

        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(policy_loss.item(), 0.1269, places=3)

    def test_supervised_training_writes_checkpoints_and_time_log(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "selfplay.jsonl"
            config_path = tmp_path / "supervised.yaml"
            model_config_path = tmp_path / "model.yaml"
            checkpoint_dir = tmp_path / "checkpoints"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=6,
                output_path=str(data_path),
                log_interval=1,
                seed=1,
            )
            config_path.write_text(
                "epochs: 1\nbatch_size: 2\nlearning_rate: 0.001\n"
                "value_loss_weight: 0.25\nvalidation_split: 0.25\n"
                "seed: 1\nshuffle: 1\nlog_interval: 100\n"
                "csv_log_interval_epochs: 1\nnum_workers: 0\n",
                encoding="utf-8",
            )
            model_config_path.write_text(
                "board_channels: 6\nboard_size: 10\nconv_channels: 4\n"
                "conv_layers: 1\nconv_kernel_size: 3\nconv_padding: 1\n"
                "board_feature_dim: 8\nmove_feature_dim: 45\n"
                "move_embedding_dim: 6\nhidden_dim: 5\n",
                encoding="utf-8",
            )

            self.run_main(
                str(data_path),
                "--config",
                str(config_path),
                "--model-config",
                str(model_config_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                "cpu",
            )

            latest = checkpoint_dir / "candidate_move_net_latest.pt"
            best = checkpoint_dir / "candidate_move_net_best.pt"
            time_log = checkpoint_dir / "training_time.json"
            self.assertTrue(latest.exists())
            self.assertTrue(best.exists())
            self.assertTrue(time_log.exists())
            csv_logs = list(checkpoint_dir.glob("training_log_*.csv"))
            self.assertEqual(len(csv_logs), 1)

            checkpoint = torch.load(latest, map_location="cpu")
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("cumulative_train_seconds", checkpoint)
            self.assertGreater(checkpoint["cumulative_train_seconds"], 0.0)

            log = json.loads(time_log.read_text(encoding="utf-8"))
            self.assertEqual(len(log), 1)
            first_seconds = next(iter(log.values()))["cumulative_train_seconds"]
            first_csv_lines = csv_logs[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(first_csv_lines), 2)
            self.assertIn("cumulative_train_seconds", first_csv_lines[0])

            self.run_main(
                str(data_path),
                "--config",
                str(config_path),
                "--model-config",
                str(model_config_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                "cpu",
            )

            log = json.loads(time_log.read_text(encoding="utf-8"))
            second_seconds = next(iter(log.values()))["cumulative_train_seconds"]
            self.assertGreater(second_seconds, first_seconds)
            second_csv_lines = csv_logs[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(second_csv_lines), 3)

    def test_default_checkpoint_policy_saves_only_final_epoch_file(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path, config_path, model_config_path, checkpoint_dir = self.make_training_fixture(
                tmp_path,
                epochs=2,
            )

            self.run_main(
                str(data_path),
                "--config",
                str(config_path),
                "--model-config",
                str(model_config_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                "cpu",
            )

            self.assertTrue((checkpoint_dir / "candidate_move_net_latest.pt").exists())
            self.assertTrue((checkpoint_dir / "candidate_move_net_best.pt").exists())
            self.assertFalse((checkpoint_dir / "candidate_move_net_epoch_001.pt").exists())
            self.assertTrue((checkpoint_dir / "candidate_move_net_epoch_002.pt").exists())

    def test_save_every_epoch_writes_intermediate_epoch_files(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path, config_path, model_config_path, checkpoint_dir = self.make_training_fixture(
                tmp_path,
                epochs=2,
            )

            self.run_main(
                str(data_path),
                "--config",
                str(config_path),
                "--model-config",
                str(model_config_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                "cpu",
                "--save-every-epoch",
            )

            self.assertTrue((checkpoint_dir / "candidate_move_net_epoch_001.pt").exists())
            self.assertTrue((checkpoint_dir / "candidate_move_net_epoch_002.pt").exists())

    def make_training_fixture(self, tmp_path: Path, epochs: int):
        data_path = tmp_path / "selfplay.jsonl"
        config_path = tmp_path / "supervised.yaml"
        model_config_path = tmp_path / "model.yaml"
        checkpoint_dir = tmp_path / "checkpoints"
        generate_dataset(
            black_name="random",
            white_name="random",
            games=1,
            max_plies=6,
            output_path=str(data_path),
            log_interval=1,
            seed=1,
        )
        config_path.write_text(
            f"epochs: {epochs}\nbatch_size: 2\nlearning_rate: 0.001\n"
            "value_loss_weight: 0.25\nvalidation_split: 0.25\n"
            "seed: 1\nshuffle: 1\nlog_interval: 100\n"
            "csv_log_interval_epochs: 1\nnum_workers: 0\n",
            encoding="utf-8",
        )
        model_config_path.write_text(
            "board_channels: 6\nboard_size: 10\nconv_channels: 4\n"
            "conv_layers: 1\nconv_kernel_size: 3\nconv_padding: 1\n"
            "board_feature_dim: 8\nmove_feature_dim: 45\n"
            "move_embedding_dim: 6\nhidden_dim: 5\n",
            encoding="utf-8",
        )
        return data_path, config_path, model_config_path, checkpoint_dir

    def run_main(self, *args):
        old_argv = sys.argv
        sys.argv = ["train_supervised.py", *args]
        try:
            main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
