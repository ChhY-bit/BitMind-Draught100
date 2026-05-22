import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from model import CandidateMoveNet, ModelConfig
from rl_selfplay import generate_rl_dataset
from train_selfplay import RlTrainConfig, compute_rl_loss, load_rl_train_config, main


class TrainSelfPlayTests(unittest.TestCase):
    def test_load_rl_train_config(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "rl_train.yaml"
            config_path.write_text(
                "epochs: 3\nbatch_size: 4\nlearning_rate: 0.0002\n"
                "policy_loss_weight: 1.0\nvalue_loss_weight: 0.5\n"
                "entropy_weight: 0.01\nadvantage_clip: 1.5\n"
                "ppo_clip_epsilon: 0.15\n"
                "validation_split: 0.25\nseed: 4\nshuffle: 1\n"
                "log_interval: 2\ncsv_log_interval_epochs: 2\nnum_workers: 0\n",
                encoding="utf-8",
            )

            config = load_rl_train_config(config_path)

        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertAlmostEqual(config.learning_rate, 0.0002)
        self.assertAlmostEqual(config.entropy_weight, 0.01)
        self.assertAlmostEqual(config.advantage_clip, 1.5)
        self.assertAlmostEqual(config.ppo_clip_epsilon, 0.15)

    def test_policy_gradient_follows_advantage_sign(self):
        positive_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        positive_parts = compute_rl_loss(
            {"policy_logits": positive_logits, "value": torch.tensor([0.0])},
            self.loss_batch(value_target=1.0),
            RlTrainConfig(value_loss_weight=0.0, advantage_clip=0.0),
        )
        positive_parts.loss.backward()

        negative_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        negative_parts = compute_rl_loss(
            {"policy_logits": negative_logits, "value": torch.tensor([0.0])},
            self.loss_batch(value_target=-1.0),
            RlTrainConfig(value_loss_weight=0.0, advantage_clip=0.0),
        )
        negative_parts.loss.backward()

        self.assertLess(positive_logits.grad[0, 0].item(), 0.0)
        self.assertGreater(negative_logits.grad[0, 0].item(), 0.0)

    def test_ppo_clip_stops_positive_ratio_growth_after_upper_bound(self):
        logits = torch.tensor([[4.0, 0.0]], requires_grad=True)
        parts = compute_rl_loss(
            {"policy_logits": logits, "value": torch.tensor([0.0])},
            self.loss_batch(value_target=1.0, selected_policy_prob=0.5),
            RlTrainConfig(value_loss_weight=0.0, advantage_clip=0.0, ppo_clip_epsilon=0.2),
        )
        parts.loss.backward()

        self.assertGreater(parts.ratio.item(), 1.2)
        self.assertTrue(parts.clipped.item())
        self.assertAlmostEqual(logits.grad[0, 0].item(), 0.0, places=6)

    def test_selfplay_training_writes_rl_checkpoints_and_logs(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_checkpoint = create_checkpoint(tmp_path / "base.pt")
            data_path = tmp_path / "rl_selfplay.jsonl"
            config_path = tmp_path / "rl_train.yaml"
            checkpoint_dir = tmp_path / "checkpoints"
            generate_rl_dataset(
                checkpoint_path=str(base_checkpoint),
                games=1,
                max_plies=6,
                output_path=str(data_path),
                log_interval=1,
                seed=1,
                temperature=0.8,
                device="cpu",
            )
            config_path.write_text(
                "epochs: 1\nbatch_size: 2\nlearning_rate: 0.0001\n"
                "policy_loss_weight: 1.0\nvalue_loss_weight: 0.5\n"
                "entropy_weight: 0.0\nadvantage_clip: 2.0\n"
                "ppo_clip_epsilon: 0.2\n"
                "validation_split: 0.25\nseed: 1\nshuffle: 1\n"
                "log_interval: 100\ncsv_log_interval_epochs: 1\nnum_workers: 0\n",
                encoding="utf-8",
            )

            self.run_main(
                str(data_path),
                "--config",
                str(config_path),
                "--resume",
                str(base_checkpoint),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                "cpu",
            )

            latest = checkpoint_dir / "candidate_move_net_rl_latest.pt"
            best = checkpoint_dir / "candidate_move_net_rl_best.pt"
            final_epoch = checkpoint_dir / "candidate_move_net_rl_epoch_001.pt"
            time_log = checkpoint_dir / "training_time.json"
            csv_logs = list(checkpoint_dir.glob("rl_training_log_*.csv"))
            checkpoint = torch.load(latest, map_location="cpu")
            log = json.loads(time_log.read_text(encoding="utf-8"))

            self.assertTrue(latest.exists())
            self.assertTrue(best.exists())
            self.assertTrue(final_epoch.exists())
            self.assertTrue(time_log.exists())
            self.assertEqual(len(csv_logs), 1)
            self.assertIn("rl_train_config", checkpoint)
            self.assertEqual(checkpoint["base_checkpoint"], str(base_checkpoint))
            self.assertEqual(len(log), 1)

    def loss_batch(self, value_target: float, selected_policy_prob: float = 0.5):
        return {
            "policy_target_index": torch.tensor([0]),
            "value_target": torch.tensor([value_target]),
            "selected_policy_prob": torch.tensor([selected_policy_prob]),
            "behavior_temperature": torch.tensor([1.0]),
            "behavior_value": torch.tensor([0.0]),
        }

    def run_main(self, *args):
        old_argv = sys.argv
        sys.argv = ["train_selfplay.py", *args]
        try:
            main()
        finally:
            sys.argv = old_argv


def create_checkpoint(path: Path) -> Path:
    config = ModelConfig(
        conv_channels=4,
        conv_layers=1,
        board_feature_dim=8,
        move_embedding_dim=6,
        hidden_dim=5,
    )
    model = CandidateMoveNet(config=config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "board_channels": config.board_channels,
                "board_size": config.board_size,
                "conv_channels": config.conv_channels,
                "conv_layers": config.conv_layers,
                "conv_kernel_size": config.conv_kernel_size,
                "conv_padding": config.conv_padding,
                "board_feature_dim": config.board_feature_dim,
                "move_feature_dim": config.move_feature_dim,
                "move_embedding_dim": config.move_embedding_dim,
                "hidden_dim": config.hidden_dim,
            },
        },
        path,
    )
    return path


if __name__ == "__main__":
    unittest.main()
