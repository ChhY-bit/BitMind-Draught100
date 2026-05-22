from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from agents import load_model_config_from_checkpoint
from draughts_env import Move
from env import StandardDraughtsEnv
from model import CandidateMoveNet
from move_encoding import encode_legal_moves
from selfplay import (
    SelfPlayStats,
    finalize_samples,
    find_selected_move,
    move_to_dict,
    next_game_id,
    resolve_output_path,
)


DEFAULT_RL_SELFPLAY_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "rl_selfplay.yaml"


@dataclass
class RlSelfPlayConfig:
    checkpoint: Optional[str] = None
    games: int = 10
    max_plies: int = 400
    output: str = "data/rl_selfplay.jsonl"
    log_interval: int = 1
    seed: int = 1
    temperature: float = 0.8
    device: Optional[str] = None
    append: int = 0
    auto_output_name: int = 1


@dataclass
class PolicyChoice:
    action_id: int
    policy_probs: List[float]
    selected_policy_prob: float
    predicted_value: float


class RlPolicySampler:
    """Training-only policy sampler used by RL self-play data generation."""

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if not checkpoint_path:
            raise ValueError("RL self-play requires a checkpoint path")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.device_obj = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device_obj)
        model_config = load_model_config_from_checkpoint(checkpoint)
        self.model = CandidateMoveNet(config=model_config).to(self.device_obj)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def choose(self, raw_observation: Dict[str, object], legal_actions: Sequence[Move], env) -> PolicyChoice:
        if not legal_actions:
            raise ValueError("RlPolicySampler received no legal actions")

        move_dicts = [move_to_dict(move) for move in legal_actions]
        side_to_move = int(raw_observation["side_to_move"])
        with torch.no_grad():
            observation = torch.tensor(
                [env.observation()],
                dtype=torch.float32,
                device=self.device_obj,
            )
            move_features = torch.tensor(
                [encode_legal_moves(move_dicts, side_to_move=side_to_move)],
                dtype=torch.float32,
                device=self.device_obj,
            )
            move_mask = torch.ones(
                (1, len(legal_actions)),
                dtype=torch.float32,
                device=self.device_obj,
            )
            output = self.model(observation, move_features, move_mask)

        probs = policy_probs_from_logits(output["policy_logits"][0], self.temperature)
        selected_index = sample_index(probs, self.rng)
        return PolicyChoice(
            action_id=legal_actions[selected_index].action_id,
            policy_probs=probs,
            selected_policy_prob=probs[selected_index],
            predicted_value=float(output["value"][0].detach().cpu().item()),
        )


def policy_probs_from_logits(logits: torch.Tensor, temperature: float) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits_cpu = logits.detach().cpu()
    if logits_cpu.numel() == 0:
        raise ValueError("policy logits must not be empty")

    scaled_logits = [float(value.item()) / temperature for value in logits_cpu]
    max_logit = max(scaled_logits)
    weights = [math.exp(value - max_logit) for value in scaled_logits]
    total = sum(weights)
    if total <= 0:
        raise RuntimeError("policy probability normalization failed")
    return [weight / total for weight in weights]


def sample_index(probs: Sequence[float], rng: random.Random) -> int:
    if not probs:
        raise ValueError("policy probabilities must not be empty")
    return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


def generate_rl_dataset(
    checkpoint_path: str,
    games: int,
    max_plies: int,
    output_path: str,
    log_interval: int,
    seed: Optional[int],
    temperature: float,
    device: Optional[str] = None,
    append: bool = False,
    auto_output_name: bool = False,
) -> SelfPlayStats:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if append and auto_output_name:
        raise ValueError("append and auto_output_name cannot both be enabled")

    sampler = RlPolicySampler(
        checkpoint_path=checkpoint_path,
        temperature=temperature,
        seed=seed,
        device=device,
    )
    output = resolve_output_path(Path(output_path), append, auto_output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_game_id = next_game_id(output) if append else 0
    file_mode = "a" if append else "w"
    stats = SelfPlayStats()
    started = time.perf_counter()

    print(
        "rl_selfplay:"
        f" checkpoint={checkpoint_path} games={games} max_plies={max_plies}"
        f" temperature={temperature} seed={seed} device={sampler.device_obj}"
        f" output={output} mode={'append' if append else 'write'}"
        f" start_game_id={start_game_id}"
    )

    with output.open(file_mode, encoding="utf-8") as handle:
        for game_offset in range(games):
            samples, game_result, plies, maxed = play_and_collect_rl_game(
                game_id=start_game_id + game_offset,
                sampler=sampler,
                max_plies=max_plies,
            )
            for sample in finalize_samples(samples, game_result, maxed):
                handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

            stats.record(game_result, plies, maxed, len(samples))
            completed_games = game_offset + 1
            if completed_games % log_interval == 0 or completed_games == games:
                elapsed = time.perf_counter() - started
                print(
                    f"rl_selfplay: {completed_games}/{games} games,"
                    f" samples={stats.samples} black_wins={stats.black_wins}"
                    f" white_wins={stats.white_wins} draws={stats.draws}"
                    f" max_plies={stats.max_plies_reached}"
                    f" avg_plies={stats.avg_plies:.2f} elapsed={elapsed:.3f}s"
                )

    return stats


def play_and_collect_rl_game(
    game_id: int,
    sampler: RlPolicySampler,
    max_plies: int,
) -> tuple[List[Dict[str, object]], int, int, bool]:
    env = StandardDraughtsEnv()
    env.reset()
    samples: List[Dict[str, object]] = []

    for ply in range(max_plies):
        raw = env.raw_observation()
        if raw["game_result_name"] != "ongoing":
            return samples, int(raw["game_result"]), ply, False

        legal_actions = env.legal_actions()
        if not legal_actions:
            return samples, int(raw["game_result"]), ply, False

        choice = sampler.choose(raw, legal_actions, env)
        selected_move = find_selected_move(legal_actions, choice.action_id)
        samples.append(
            build_rl_sample(
                game_id=game_id,
                ply=ply,
                raw_observation=raw,
                encoded_observation=env.observation(),
                action_mask=env.action_mask(),
                legal_actions=legal_actions,
                selected_move=selected_move,
                choice=choice,
                checkpoint_path=sampler.checkpoint_path,
                temperature=sampler.temperature,
            )
        )
        env.step(choice.action_id)

    raw = env.raw_observation()
    return samples, int(raw["game_result"]), max_plies, True


def build_rl_sample(
    game_id: int,
    ply: int,
    raw_observation: Dict[str, object],
    encoded_observation: List[List[List[int]]],
    action_mask: Sequence[int],
    legal_actions: Sequence[Move],
    selected_move: Move,
    choice: PolicyChoice,
    checkpoint_path: str,
    temperature: float,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "sample_kind": "rl_selfplay",
        "game_id": game_id,
        "ply": ply,
        "side_to_move": int(raw_observation["side_to_move"]),
        "side_to_move_name": raw_observation["side_to_move_name"],
        "agent": "rl_policy_sampler",
        "behavior_policy": "temperature_softmax",
        "temperature": float(temperature),
        "policy_checkpoint": checkpoint_path,
        "raw_squares": list(raw_observation["squares"]),
        "observation": encoded_observation,
        "legal_action_ids": [move.action_id for move in legal_actions],
        "legal_moves": [move_to_dict(move) for move in legal_actions],
        "action_mask": list(action_mask),
        "selected_action_id": selected_move.action_id,
        "selected_move": move_to_dict(selected_move),
        "policy_probs": list(choice.policy_probs),
        "selected_policy_prob": float(choice.selected_policy_prob),
        "predicted_value": float(choice.predicted_value),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RL self-play JSONL data from a checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_RL_SELFPLAY_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-output-name", action=argparse.BooleanOptionalAction, default=None)
    args = merge_rl_selfplay_args(parser.parse_args())

    if not args.checkpoint:
        parser.error("--checkpoint is required by CLI or config")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.append and args.auto_output_name:
        parser.error("--append and --auto-output-name cannot be used together")
    return args


def merge_rl_selfplay_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_rl_selfplay_config(cli_args.config)
    values = asdict(config)
    overrides = {
        "checkpoint": cli_args.checkpoint,
        "games": cli_args.games,
        "max_plies": cli_args.max_plies,
        "output": cli_args.output,
        "log_interval": cli_args.log_interval,
        "seed": cli_args.seed,
        "temperature": cli_args.temperature,
        "device": cli_args.device,
        "append": cli_args.append,
        "auto_output_name": cli_args.auto_output_name,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if cli_args.append is True and cli_args.auto_output_name is None:
        values["auto_output_name"] = 0
    if cli_args.auto_output_name is True and cli_args.append is None:
        values["append"] = 0

    values["config"] = cli_args.config
    values["append"] = bool(int(values["append"]))
    values["auto_output_name"] = bool(int(values["auto_output_name"]))
    return argparse.Namespace(**values)


def load_rl_selfplay_config(config_path: str | Path) -> RlSelfPlayConfig:
    path = Path(config_path)
    defaults = asdict(RlSelfPlayConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(path, defaults, "rl_selfplay"))
    config = RlSelfPlayConfig(**values)
    validate_rl_selfplay_config(config)
    return config


def load_scalar_yaml(path: Path, defaults: Dict[str, object], config_name: str) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{config_name} config not found: {path}")

    values: Dict[str, object] = {}
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
        if value.lower() in {"none", "null"}:
            values[key] = None
            continue
        default_value = defaults[key]
        try:
            if isinstance(default_value, int) and not isinstance(default_value, bool):
                values[key] = int(value)
            elif isinstance(default_value, float):
                values[key] = float(value)
            else:
                values[key] = value
        except ValueError as exc:
            raise ValueError(f"{config_name} config value has wrong type on line {line_number}") from exc
    return values


def validate_rl_selfplay_config(config: RlSelfPlayConfig) -> None:
    if config.games <= 0:
        raise ValueError("rl_selfplay games must be positive")
    if config.max_plies <= 0:
        raise ValueError("rl_selfplay max_plies must be positive")
    if config.log_interval <= 0:
        raise ValueError("rl_selfplay log_interval must be positive")
    if config.temperature <= 0:
        raise ValueError("rl_selfplay temperature must be positive")
    if config.append not in {0, 1}:
        raise ValueError("rl_selfplay append must be 0 or 1")
    if config.auto_output_name not in {0, 1}:
        raise ValueError("rl_selfplay auto_output_name must be 0 or 1")
    if config.append and config.auto_output_name:
        raise ValueError("rl_selfplay append and auto_output_name cannot both be enabled")


def main() -> None:
    args = parse_args()
    generate_rl_dataset(
        checkpoint_path=args.checkpoint,
        games=args.games,
        max_plies=args.max_plies,
        output_path=args.output,
        log_interval=args.log_interval,
        seed=args.seed,
        temperature=args.temperature,
        device=args.device,
        append=args.append,
        auto_output_name=args.auto_output_name,
    )


if __name__ == "__main__":
    main()
