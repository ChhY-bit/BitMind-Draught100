from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .move_encoding import MOVE_FEATURE_DIM


BOARD_CHANNELS = 6
BOARD_SIZE = 10
MASKED_LOGIT = -1.0e9
DEFAULT_MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "model.yaml"


@dataclass
class ModelConfig:
    board_channels: int = BOARD_CHANNELS
    board_size: int = BOARD_SIZE
    conv_channels: int = 64
    conv_layers: int = 3
    conv_kernel_size: int = 3
    conv_padding: int = 1
    board_feature_dim: int = 256
    move_feature_dim: int = MOVE_FEATURE_DIM
    move_embedding_dim: int = 128
    hidden_dim: int = 128


class CandidateMoveNet(nn.Module):
    """Policy/value network that scores legal candidate moves."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or load_model_config(config_path)
        validate_model_config(self.config)

        self.board_encoder = build_board_encoder(self.config)
        self.move_encoder = nn.Sequential(
            nn.Linear(self.config.move_feature_dim, self.config.move_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.move_embedding_dim, self.config.move_embedding_dim),
            nn.ReLU(),
        )
        # 定义Policy头的网络结构
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.board_feature_dim + self.config.move_embedding_dim,
                self.config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.board_feature_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        move_features: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if observation.ndim != 4:
            raise ValueError("observation must have shape [B, 6, 10, 10]")
        if move_features.ndim != 3:
            raise ValueError("move_features must have shape [B, M, F]")

        batch_size, move_count, _feature_dim = move_features.shape
        board_feature = self.board_encoder(observation)
        move_embedding = self.move_encoder(move_features)

        expanded_board = board_feature.unsqueeze(1).expand(-1, move_count, -1)
        score_input = torch.cat([expanded_board, move_embedding], dim=-1)
        policy_logits = self.score_head(score_input).squeeze(-1)
        if move_mask is not None:
            if move_mask.shape != (batch_size, move_count):
                raise ValueError("move_mask must have shape [B, M]")
            policy_logits = policy_logits.masked_fill(move_mask <= 0, MASKED_LOGIT)

        value = self.value_head(board_feature).squeeze(-1)
        return {
            "policy_logits": policy_logits,
            "value": value,
        }


def build_board_encoder(config: ModelConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = config.board_channels
    for _index in range(config.conv_layers):
        layers.append(
            nn.Conv2d(
                in_channels,
                config.conv_channels,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_padding,
            )
        )
        layers.append(nn.ReLU())
        in_channels = config.conv_channels
    layers.extend(
        [
            nn.Flatten(),
            nn.Linear(
                config.conv_channels * config.board_size * config.board_size,
                config.board_feature_dim,
            ),
            nn.ReLU(),
        ]
    )
    return nn.Sequential(*layers)


def load_model_config(config_path: Optional[str] = None) -> ModelConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_MODEL_CONFIG_PATH
    default_config = ModelConfig()
    values = {
        "board_channels": default_config.board_channels,
        "board_size": default_config.board_size,
        "conv_channels": default_config.conv_channels,
        "conv_layers": default_config.conv_layers,
        "conv_kernel_size": default_config.conv_kernel_size,
        "conv_padding": default_config.conv_padding,
        "board_feature_dim": default_config.board_feature_dim,
        "move_feature_dim": default_config.move_feature_dim,
        "move_embedding_dim": default_config.move_embedding_dim,
        "hidden_dim": default_config.hidden_dim,
    }
    values.update(load_integer_yaml(path, values, "model"))
    config = ModelConfig(**values)
    validate_model_config(config)
    return config


def load_integer_yaml(path: Path, defaults: dict[str, int], config_name: str) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"{config_name} config not found: {path}")

    values: dict[str, int] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in defaults:
            raise ValueError(f"unknown {config_name} config key on line {line_number}: {key}")
        try:
            values[key] = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{config_name} config value must be an integer on line {line_number}"
            ) from exc
    return values


def validate_model_config(config: ModelConfig) -> None:
    positive_fields = {
        "board_channels": config.board_channels,
        "board_size": config.board_size,
        "conv_channels": config.conv_channels,
        "conv_layers": config.conv_layers,
        "conv_kernel_size": config.conv_kernel_size,
        "board_feature_dim": config.board_feature_dim,
        "move_feature_dim": config.move_feature_dim,
        "move_embedding_dim": config.move_embedding_dim,
        "hidden_dim": config.hidden_dim,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.conv_padding < 0:
        raise ValueError("conv_padding must be non-negative")
    if config.conv_kernel_size % 2 == 0:
        raise ValueError("conv_kernel_size must be odd")
    if config.conv_padding != config.conv_kernel_size // 2:
        raise ValueError("conv_padding must preserve board size")
