import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from model import CandidateMoveNet, ModelConfig
from rl_selfplay import (
    generate_rl_dataset,
    load_rl_selfplay_config,
    parse_args,
    policy_probs_from_logits,
    sample_index,
)


class RlSelfPlayTests(unittest.TestCase):
    def test_policy_probs_from_logits_respects_temperature(self):
        cool = policy_probs_from_logits(torch.tensor([0.0, 2.0]), temperature=0.5)
        warm = policy_probs_from_logits(torch.tensor([0.0, 2.0]), temperature=2.0)

        self.assertAlmostEqual(sum(cool), 1.0)
        self.assertGreater(cool[1], warm[1])
        self.assertIn(sample_index(cool, __import__("random").Random(1)), {0, 1})

    def test_generate_rl_dataset_writes_sampling_fields(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = create_zero_checkpoint(tmp_path)
            output = tmp_path / "rl_selfplay.jsonl"

            stats = generate_rl_dataset(
                checkpoint_path=str(checkpoint),
                games=1,
                max_plies=4,
                output_path=str(output),
                log_interval=1,
                seed=1,
                temperature=0.8,
                device="cpu",
            )
            samples = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.samples, len(samples))
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        self.assertEqual(sample["sample_kind"], "rl_selfplay")
        self.assertEqual(sample["behavior_policy"], "temperature_softmax")
        self.assertEqual(sample["temperature"], 0.8)
        self.assertEqual(len(sample["policy_probs"]), len(sample["legal_moves"]))
        self.assertIn(sample["selected_action_id"], sample["legal_action_ids"])
        self.assertAlmostEqual(sum(sample["policy_probs"]), 1.0)
        self.assertIn("selected_policy_prob", sample)
        self.assertIn("predicted_value", sample)
        self.assertIn("final_result", sample)
        self.assertIn(sample["value_target"], {-1, 0, 1})

    def test_load_config_and_cli_override(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "rl_selfplay.yaml"
            config_path.write_text(
                "checkpoint: checkpoints/base.pt\ngames: 2\nmax_plies: 12\n"
                "output: data/rl.jsonl\nlog_interval: 2\nseed: 9\n"
                "temperature: 0.7\ndevice: cpu\nappend: 0\nauto_output_name: 1\n",
                encoding="utf-8",
            )

            config = load_rl_selfplay_config(config_path)
            args = self.parse_with_argv(
                "--config",
                str(config_path),
                "--games",
                "3",
                "--temperature",
                "1.1",
                "--append",
            )

        self.assertEqual(config.checkpoint, "checkpoints/base.pt")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(args.games, 3)
        self.assertEqual(args.temperature, 1.1)
        self.assertTrue(args.append)
        self.assertFalse(args.auto_output_name)

    def parse_with_argv(self, *args):
        old_argv = sys.argv
        sys.argv = ["rl_selfplay.py", *args]
        try:
            return parse_args()
        finally:
            sys.argv = old_argv


def create_zero_checkpoint(directory: Path) -> Path:
    config = ModelConfig(
        conv_channels=4,
        conv_layers=1,
        board_feature_dim=8,
        move_embedding_dim=6,
        hidden_dim=5,
    )
    model = CandidateMoveNet(config=config)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    path = directory / "rl_policy.pt"
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
