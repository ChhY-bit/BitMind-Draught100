import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from dataset import DraughtsJsonlDataset, collate_batch, to_torch_batch
from model import MASKED_LOGIT, CandidateMoveNet, ModelConfig, load_model_config
from move_encoding import MOVE_FEATURE_DIM
from selfplay import generate_dataset


class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_masking(self):
        model = CandidateMoveNet()
        observation = torch.zeros((2, 6, 10, 10), dtype=torch.float32)
        move_features = torch.zeros((2, 3, MOVE_FEATURE_DIM), dtype=torch.float32)
        move_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

        output = model(observation, move_features, move_mask)

        self.assertEqual(tuple(output["policy_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["value"].shape), (2,))
        self.assertLessEqual(output["policy_logits"][0, 2].item(), MASKED_LOGIT / 2)
        self.assertLessEqual(output["policy_logits"][1, 1].item(), MASKED_LOGIT / 2)
        self.assertTrue(torch.all(output["value"] <= 1.0))
        self.assertTrue(torch.all(output["value"] >= -1.0))

    def test_load_model_config_from_yaml(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model.yaml"
            config_path.write_text(
                "board_channels: 6\nboard_size: 10\nconv_channels: 16\n"
                "conv_layers: 2\nconv_kernel_size: 3\nconv_padding: 1\n"
                "board_feature_dim: 32\nmove_feature_dim: 45\n"
                "move_embedding_dim: 24\nhidden_dim: 20\n",
                encoding="utf-8",
            )

            config = load_model_config(str(config_path))

        self.assertEqual(config.conv_channels, 16)
        self.assertEqual(config.conv_layers, 2)
        self.assertEqual(config.board_feature_dim, 32)
        self.assertEqual(config.move_embedding_dim, 24)

    def test_forward_with_small_custom_config(self):
        model = CandidateMoveNet(
            config=ModelConfig(
                conv_channels=8,
                conv_layers=1,
                board_feature_dim=16,
                move_embedding_dim=12,
                hidden_dim=10,
            )
        )
        observation = torch.zeros((1, 6, 10, 10), dtype=torch.float32)
        move_features = torch.zeros((1, 2, MOVE_FEATURE_DIM), dtype=torch.float32)

        output = model(observation, move_features)

        self.assertEqual(tuple(output["policy_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["value"].shape), (1,))

    def test_forward_accepts_dataset_batch(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "selfplay.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(output_path),
                log_interval=1,
                seed=1,
            )
            dataset = DraughtsJsonlDataset([str(output_path)])
            batch = collate_batch([dataset[0], dataset[1]])
            torch_batch = to_torch_batch(batch)

        model = CandidateMoveNet()
        output = model(
            torch_batch["observation"],
            torch_batch["move_features"],
            torch_batch["move_mask"],
        )

        self.assertEqual(output["policy_logits"].shape[0], 2)
        self.assertEqual(output["policy_logits"].shape[1], torch_batch["move_features"].shape[1])
        self.assertEqual(output["value"].shape[0], 2)

    def test_invalid_shapes_raise(self):
        model = CandidateMoveNet()
        with self.assertRaises(ValueError):
            model(torch.zeros((6, 10, 10)), torch.zeros((1, 2, MOVE_FEATURE_DIM)))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 6, 10, 10)), torch.zeros((2, MOVE_FEATURE_DIM)))

    def test_invalid_conv_config_raises(self):
        with self.assertRaises(ValueError):
            CandidateMoveNet(config=ModelConfig(conv_kernel_size=4, conv_padding=2))
        with self.assertRaises(ValueError):
            CandidateMoveNet(config=ModelConfig(conv_kernel_size=3, conv_padding=0))


if __name__ == "__main__":
    unittest.main()
