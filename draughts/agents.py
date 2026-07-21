from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from draughts_env import Move
from .mcts import (
    DEFAULT_MCTS_CONFIG_PATH,
    MCTSConfig,
    MCTSResult,
    load_mcts_config,
    mcts_search,
)
from .model import CandidateMoveNet, ModelConfig
from .move_encoding import encode_legal_moves, move_to_dict, square_to_coord

DEFAULT_ALPHABETA_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "alphabeta.yaml"

BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3
ONGOING = 0
NEG_INF = -10**12
POS_INF = 10**12


class Agent:
    name = "agent"

    def select_action(
        self,
        observation: Dict[str, object],
        legal_actions: Sequence[Move],
        env=None,
    ) -> int:
        raise NotImplementedError


@dataclass
class RandomAgent(Agent):
    seed: Optional[int] = None
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select_action(
        self,
        observation: Dict[str, object],
        legal_actions: Sequence[Move],
        env=None,
    ) -> int:
        if not legal_actions:
            raise ValueError("RandomAgent received no legal actions")
        return self._rng.choice(list(legal_actions)).action_id


@dataclass
class AlphaBetaConfig:
    search_depth: int = 2
    man_value: int = 100
    king_value: int = 250
    win_value: int = 1_000_000
    draw_value: int = 0
    advancement_value: int = 4
    center_value: int = 3
    edge_penalty: int = 2
    back_row_guard_value: int = 5


@dataclass
class AlphaBetaAgent(Agent):
    seed: Optional[int] = None
    config_path: Optional[str] = None
    depth: Optional[int] = None
    name: str = "alphabeta"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.config = load_alphabeta_config(self.config_path)
        if self.depth is not None:
            self.config.search_depth = self.depth

    def select_action(
        self,
        observation: Dict[str, object],
        legal_actions: Sequence[Move],
        env=None,
    ) -> int:
        if not legal_actions:
            raise ValueError("AlphaBetaAgent received no legal actions")
        if env is None:
            raise ValueError("AlphaBetaAgent requires an environment with clone()")

        root_side = int(observation["side_to_move"])
        scored_moves = []
        ordered_actions = order_moves(legal_actions, observation)
        for move in ordered_actions:
            cloned = env.clone()
            cloned.step(move.action_id)
            score = self._search(
                cloned,
                depth=self.config.search_depth - 1,
                root_side=root_side,
                alpha=NEG_INF,
                beta=POS_INF,
            )
            scored_moves.append((score, move))

        best_score = max(score for score, _move in scored_moves)
        candidates = [move for score, move in scored_moves if score == best_score]
        return self._rng.choice(candidates).action_id

    def _search(
        self,
        env,
        depth: int,
        root_side: int,
        alpha: int,
        beta: int,
    ) -> int:
        observation = env.raw_observation()
        result = int(observation["game_result"])
        if depth <= 0 or result != ONGOING:
            return evaluate_enhanced_position(observation, root_side, self.config)

        legal_actions = env.legal_actions()
        if not legal_actions:
            return evaluate_enhanced_position(observation, root_side, self.config)

        side_to_move = int(observation["side_to_move"])
        ordered_actions = order_moves(legal_actions, observation)
        if side_to_move == root_side:
            value = NEG_INF
            for move in ordered_actions:
                cloned = env.clone()
                cloned.step(move.action_id)
                value = max(
                    value,
                    self._search(cloned, depth - 1, root_side, alpha, beta),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = POS_INF
        for move in ordered_actions:
            cloned = env.clone()
            cloned.step(move.action_id)
            value = min(
                value,
                self._search(cloned, depth - 1, root_side, alpha, beta),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


@dataclass
class SupervisedAgent(Agent):
    checkpoint_path: str
    device: Optional[str] = None
    name: str = "supervised"

    def __post_init__(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("SupervisedAgent requires a checkpoint path")
        self.device_obj = torch.device(
            self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device_obj)
        model_config = load_model_config_from_checkpoint(checkpoint)
        self.model = CandidateMoveNet(config=model_config).to(self.device_obj)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.last_policy_logits: Optional[List[float]] = None
        self.last_value: Optional[float] = None
        self.last_latent: Optional[torch.Tensor] = None

    def select_action(
        self,
        observation: Dict[str, object],
        legal_actions: Sequence[Move],
        env=None,
    ) -> int:
        if not legal_actions:
            raise ValueError("SupervisedAgent received no legal actions")
        if env is None:
            raise ValueError("SupervisedAgent requires an environment with observation()")

        move_dicts = [move_to_dict(move) for move in legal_actions]
        with torch.no_grad():
            observation_tensor = torch.tensor(
                [env.observation()],
                dtype=torch.float32,
                device=self.device_obj,
            )
            move_tensor = torch.tensor(
                [encode_legal_moves(move_dicts, side_to_move=int(observation["side_to_move"]))],
                dtype=torch.float32,
                device=self.device_obj,
            )
            move_mask = torch.ones((1, len(legal_actions)), dtype=torch.float32, device=self.device_obj)
            output = self.model(observation_tensor, move_tensor, move_mask, return_latent=True)
            self.last_policy_logits = output["policy_logits"][0].tolist()
            self.last_value = float(output["value"][0].item())
            self.last_latent = output.get("latent")  # [1, M, 384]
            selected_index = int(output["policy_logits"][0].argmax().item())
        return legal_actions[selected_index].action_id


@dataclass
class MCTSAgent(Agent):
    """Agent that uses MCTS guided by a CandidateMoveNet for move selection."""

    checkpoint_path: str
    config_path: Optional[str] = None
    device: Optional[str] = None
    name: str = "mcts"

    def __post_init__(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("MCTSAgent requires a checkpoint path")
        self.device_obj = torch.device(
            self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mcts_config = load_mcts_config(self.config_path)
        self.last_result: Optional[MCTSResult] = None

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device_obj)
        model_config = load_model_config_from_checkpoint(checkpoint)
        self.model = CandidateMoveNet(config=model_config).to(self.device_obj)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_action(
        self,
        observation: Dict[str, object],
        legal_actions: Sequence[Move],
        env=None,
    ) -> int:
        if not legal_actions:
            raise ValueError("MCTSAgent received no legal actions")
        if env is None:
            raise ValueError("MCTSAgent requires an environment for clone() and observation()")

        self.last_result = mcts_search(
            env,
            self.model,
            self.mcts_config,
            self.device_obj,
            add_noise=self.mcts_config.add_noise,
        )
        return self.last_result.selected_action(temperature=self.mcts_config.temperature)


def load_model_config_from_checkpoint(checkpoint: object) -> ModelConfig:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return ModelConfig(**checkpoint["model_config"])
    return ModelConfig()


def promotion_score(move: Move, observation: Dict[str, object]) -> int:
    squares = observation["squares"]
    side_to_move = int(observation["side_to_move"])
    start = move.path[0]
    final = move.path[-1]
    moving_piece = int(squares[start - 1])
    final_row = (final - 1) // 5

    if side_to_move == 0 and moving_piece == 1 and final_row == 9:
        return 1
    if side_to_move == 1 and moving_piece == 3 and final_row == 0:
        return 1
    return 0


def evaluate_enhanced_position(
    observation: Dict[str, object],
    side: int,
    config: AlphaBetaConfig,
) -> int:
    result = int(observation["game_result"])
    if result == BLACK_WIN:
        return config.win_value if side == 0 else -config.win_value
    if result == WHITE_WIN:
        return config.win_value if side == 1 else -config.win_value
    if result == DRAW:
        return config.draw_value

    black_score = 0
    white_score = 0
    for index, piece in enumerate(observation["squares"], start=1):
        row, col = square_to_coord(index)
        if piece == 1:
            black_score += evaluate_man_features(row, col, 0, config)
        elif piece == 2:
            black_score += evaluate_king_features(row, col, config)
        elif piece == 3:
            white_score += evaluate_man_features(row, col, 1, config)
        elif piece == 4:
            white_score += evaluate_king_features(row, col, config)

    return black_score - white_score if side == 0 else white_score - black_score


def evaluate_man_features(row: int, col: int, side: int, config: AlphaBetaConfig) -> int:
    score = config.man_value
    advancement = row if side == 0 else 9 - row
    score += advancement * config.advancement_value
    if is_center_square(row, col):
        score += config.center_value
    if is_edge_square(col):
        score -= config.edge_penalty
    if (side == 0 and row == 0) or (side == 1 and row == 9):
        score += config.back_row_guard_value
    return score


def evaluate_king_features(row: int, col: int, config: AlphaBetaConfig) -> int:
    score = config.king_value
    if is_center_square(row, col):
        score += config.center_value
    if is_edge_square(col):
        score -= config.edge_penalty
    return score


def is_center_square(row: int, col: int) -> bool:
    return 2 <= row <= 7 and 2 <= col <= 7


def is_edge_square(col: int) -> bool:
    return col == 0 or col == 9


def load_alphabeta_config(config_path: Optional[str] = None) -> AlphaBetaConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_ALPHABETA_CONFIG_PATH
    default_config = AlphaBetaConfig()
    values = load_integer_yaml(
        path,
        {
            "search_depth": default_config.search_depth,
            "man_value": default_config.man_value,
            "king_value": default_config.king_value,
            "win_value": default_config.win_value,
            "draw_value": default_config.draw_value,
            "advancement_value": default_config.advancement_value,
            "center_value": default_config.center_value,
            "edge_penalty": default_config.edge_penalty,
            "back_row_guard_value": default_config.back_row_guard_value,
        },
        "alphabeta",
    )
    config = AlphaBetaConfig(**values)
    if config.search_depth <= 0:
        raise ValueError("alphabeta search_depth must be positive")
    return config


def load_integer_yaml(path: Path, defaults: Dict[str, int], config_name: str) -> Dict[str, int]:
    values = {key: value for key, value in defaults.items()}

    if not path.exists():
        raise FileNotFoundError(f"{config_name} config not found: {path}")

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in values:
            raise ValueError(f"unknown {config_name} config key on line {line_number}: {key}")
        try:
            values[key] = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{config_name} config value must be an integer on line {line_number}"
            ) from exc

    return values


def order_moves(legal_actions: Sequence[Move], observation: Dict[str, object]) -> List[Move]:
    return sorted(
        legal_actions,
        key=lambda move: (len(move.captures), promotion_score(move, observation)),
        reverse=True,
    )


def make_agent(
    name: str,
    seed: Optional[int] = None,
    alphabeta_config_path: Optional[str] = None,
    alphabeta_depth: Optional[int] = None,
    supervised_checkpoint_path: Optional[str] = None,
    mcts_config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Agent:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized in {"alphabeta", "alpha-beta", "ab"}:
        return AlphaBetaAgent(seed=seed, config_path=alphabeta_config_path, depth=alphabeta_depth)
    if normalized in {"supervised", "supervised-agent", "sl"}:
        if supervised_checkpoint_path is None:
            raise ValueError("supervised agent requires a checkpoint path")
        return SupervisedAgent(checkpoint_path=supervised_checkpoint_path, device=device)
    if normalized in {"mcts", "mcts-agent"}:
        if supervised_checkpoint_path is None:
            raise ValueError("mcts agent requires a checkpoint path (--supervised-checkpoint)")
        return MCTSAgent(
            checkpoint_path=supervised_checkpoint_path,
            config_path=mcts_config_path,
            device=device,
        )
    raise ValueError(f"unknown agent: {name}")


def available_agents() -> List[str]:
    return [
        "random",
        "alphabeta",
        "supervised",
        "mcts",
    ]
