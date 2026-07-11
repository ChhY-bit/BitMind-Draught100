from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from draughts.agents import Agent, available_agents, make_agent
from draughts_env import Move
from draughts.env import StandardDraughtsEnv


BLACK_SIDE = 0
WHITE_SIDE = 1
BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3
DEFAULT_SELFPLAY_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "selfplay.yaml"


@dataclass
class SelfPlayConfig:
    black: str = "alphabeta"
    white: str = "alphabeta"
    games: int = 10
    max_plies: int = 400
    output: str = "data/selfplay.jsonl"
    log_interval: int = 1
    seed: int = 1
    opening_random_plies: int = 6
    material_config: Optional[str] = None
    alphabeta_config: Optional[str] = None
    black_checkpoint: Optional[str] = None
    white_checkpoint: Optional[str] = None
    mcts_config: Optional[str] = None
    device: Optional[str] = None
    append: int = 0
    auto_output_name: int = 1


@dataclass
class SelfPlayStats:
    games: int = 0
    samples: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    max_plies_reached: int = 0
    total_plies: int = 0

    def record(self, game_result: int, plies: int, maxed: bool, samples: int) -> None:
        self.games += 1
        self.samples += samples
        self.total_plies += plies
        if maxed:
            self.max_plies_reached += 1
        elif game_result == BLACK_WIN:
            self.black_wins += 1
        elif game_result == WHITE_WIN:
            self.white_wins += 1
        elif game_result == DRAW:
            self.draws += 1

    @property
    def avg_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_plies / self.games


def generate_dataset(
    black_name: str,
    white_name: str,
    games: int,
    max_plies: int,
    output_path: str,
    log_interval: int,
    seed: Optional[int],
    material_config_path: Optional[str] = None,
    alphabeta_config_path: Optional[str] = None,
    black_checkpoint_path: Optional[str] = None,
    white_checkpoint_path: Optional[str] = None,
    mcts_config_path: Optional[str] = None,
    device: Optional[str] = None,
    append: bool = False,
    auto_output_name: bool = False,
    opening_random_plies: int = 0,
) -> SelfPlayStats:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if opening_random_plies < 0:
        raise ValueError("opening_random_plies must be non-negative")

    black_agent = make_agent(
        black_name,
        seed=seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
        supervised_checkpoint_path=black_checkpoint_path,
        mcts_config_path=mcts_config_path,
        device=device,
    )
    white_seed = None if seed is None else seed + 1
    white_agent = make_agent(
        white_name,
        seed=white_seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
        supervised_checkpoint_path=white_checkpoint_path,
        mcts_config_path=mcts_config_path,
        device=device,
    )

    if append and auto_output_name:
        raise ValueError("append and auto_output_name cannot both be enabled")

    output = resolve_output_path(Path(output_path), append, auto_output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_game_id = next_game_id(output) if append else 0
    file_mode = "a" if append else "w"
    stats = SelfPlayStats()
    start = time.perf_counter()
    opening_rng = random.Random(seed) if opening_random_plies > 0 else None

    print(
        f"selfplay: black={black_agent.name} white={white_agent.name} "
        f"games={games} max_plies={max_plies} seed={seed} output={output} "
        f"mode={'append' if append else 'write'} start_game_id={start_game_id} "
        f"material_config={material_config_path or 'default'} "
        f"alphabeta_config={alphabeta_config_path or 'default'} "
        f"black_checkpoint={black_checkpoint_path or 'none'} "
        f"white_checkpoint={white_checkpoint_path or 'none'} "
        f"opening_random_plies={opening_random_plies}"
    )

    with output.open(file_mode, encoding="utf-8") as handle:
        for game_offset in range(games):
            game_id = start_game_id + game_offset
            samples, game_result, plies, maxed = play_and_collect_game(
                game_id=game_id,
                black_agent=black_agent,
                white_agent=white_agent,
                max_plies=max_plies,
                opening_random_plies=opening_random_plies,
                opening_rng=opening_rng,
            )
            for sample in finalize_samples(samples, game_result, maxed):
                handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

            stats.record(game_result, plies, maxed, len(samples))

            completed_games = game_offset + 1
            if completed_games % log_interval == 0 or completed_games == games:
                elapsed = time.perf_counter() - start
                print(
                    f"selfplay: {completed_games}/{games} games, "
                    f"samples={stats.samples} black_wins={stats.black_wins} "
                    f"white_wins={stats.white_wins} draws={stats.draws} "
                    f"max_plies={stats.max_plies_reached} "
                    f"avg_plies={stats.avg_plies:.2f} elapsed={elapsed:.3f}s"
                )

    return stats


def resolve_output_path(path: Path, append: bool, auto_output_name: bool) -> Path:
    if append or not auto_output_name or not path.exists():
        return path

    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    for index in range(1, 10000):
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find available output filename for {path}")


def next_game_id(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0

    max_game_id = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON while scanning {path}:{line_number}") from exc
            if not isinstance(sample, dict) or "game_id" not in sample:
                raise ValueError(f"missing game_id while scanning {path}:{line_number}")
            max_game_id = max(max_game_id, int(sample["game_id"]))
    return max_game_id + 1


def play_and_collect_game(
    game_id: int,
    black_agent: Agent,
    white_agent: Agent,
    max_plies: int,
    opening_random_plies: int = 0,
    opening_rng: Optional[random.Random] = None,
) -> tuple[List[Dict[str, object]], int, int, bool]:
    if opening_random_plies < 0:
        raise ValueError("opening_random_plies must be non-negative")
    if opening_random_plies > 0 and opening_rng is None:
        raise ValueError("opening_rng is required when opening_random_plies is positive")

    env = StandardDraughtsEnv()
    env.reset()
    samples: List[Dict[str, object]] = []

    # Phase 1: random opening moves (not recorded as training samples).
    plies, opening_done = apply_random_opening(
        env, opening_random_plies, max_plies, opening_rng
    )
    if opening_done:
        raw = env.raw_observation()
        return samples, int(raw["game_result"]), plies, opening_done

    # Phase 2: agent-driven play (recorded as training samples).
    for ply in range(plies, max_plies):
        raw = env.raw_observation()
        if raw["game_result_name"] != "ongoing":
            return samples, int(raw["game_result"]), ply, False

        legal_actions = env.legal_actions()
        if not legal_actions:
            return samples, int(raw["game_result"]), ply, False

        side_to_move = int(raw["side_to_move"])
        agent = black_agent if side_to_move == BLACK_SIDE else white_agent
        action_id = agent.select_action(raw, legal_actions, env)
        selected_move = find_selected_move(legal_actions, action_id)

        # Extract soft policy target from MCTS search result when available.
        policy_probs = None
        if hasattr(agent, "last_result") and agent.last_result is not None:
            policy_probs = agent.last_result.policy_probs

        samples.append(
            build_sample(
                game_id=game_id,
                ply=ply,
                raw_observation=raw,
                encoded_observation=env.observation(),
                action_mask=env.action_mask(),
                legal_actions=legal_actions,
                selected_move=selected_move,
                agent_name=agent.name,
                policy_target_probs=policy_probs,
            )
        )
        env.step(action_id)

    raw = env.raw_observation()
    return samples, int(raw["game_result"]), max_plies, True


def apply_random_opening(
    env: StandardDraughtsEnv,
    opening_random_plies: int,
    max_plies: int,
    rng: Optional[random.Random],
) -> tuple[int, bool]:
    """Advance a few random moves before the evaluated agents take over.

    These random moves are NOT recorded as training samples — they only
    serve to diversify the starting position so that deterministic agents
    (e.g. AlphaBeta) produce varied games.
    """
    plies = 0
    while plies < opening_random_plies and plies < max_plies:
        raw = env.raw_observation()
        if raw["game_result_name"] != "ongoing":
            return plies, True
        legal_actions = env.legal_actions()
        if not legal_actions:
            return plies, True
        if rng is None:
            raise ValueError("opening rng is required for random opening moves")
        env.step(rng.choice(legal_actions).action_id)
        plies += 1
    return plies, plies >= max_plies


def build_sample(
    game_id: int,
    ply: int,
    raw_observation: Dict[str, object],
    encoded_observation: List[List[List[int]]],
    action_mask: Sequence[int],
    legal_actions: Sequence[Move],
    selected_move: Move,
    agent_name: str,
    policy_target_probs: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    side_to_move = int(raw_observation["side_to_move"])
    sample: Dict[str, object] = {
        "schema_version": 1,
        "game_id": game_id,
        "ply": ply,
        "side_to_move": side_to_move,
        "side_to_move_name": raw_observation["side_to_move_name"],
        "agent": agent_name,
        "raw_squares": list(raw_observation["squares"]),
        "observation": encoded_observation,
        "legal_action_ids": [move.action_id for move in legal_actions],
        "legal_moves": [move_to_dict(move) for move in legal_actions],
        "action_mask": list(action_mask),
        "selected_action_id": selected_move.action_id,
        "selected_move": move_to_dict(selected_move),
    }
    if policy_target_probs is not None:
        sample["policy_target_probs"] = [float(p) for p in policy_target_probs]
    return sample

# 对局结束后，为所有记录补充对局结果信息
def finalize_samples(
    samples: Sequence[Dict[str, object]],
    final_result: int,
    maxed: bool,
) -> List[Dict[str, object]]:
    finalized = []
    for sample in samples:
        item = dict(sample)
        side_to_move = int(item["side_to_move"])
        item["final_result"] = final_result
        item["max_plies_reached"] = bool(maxed)
        item["value_target"] = value_target_for_side(final_result, side_to_move, maxed)
        finalized.append(item)
    return finalized

# 价值头的输出标签
def value_target_for_side(final_result: int, side: int, maxed: bool = False) -> int:
    if maxed or final_result == DRAW:
        return 0
    if final_result == BLACK_WIN:
        return 1 if side == BLACK_SIDE else -1
    if final_result == WHITE_WIN:
        return 1 if side == WHITE_SIDE else -1
    return 0


def find_selected_move(legal_actions: Sequence[Move], action_id: int) -> Move:
    for move in legal_actions:
        if move.action_id == action_id:
            return move
    raise ValueError(f"agent selected illegal action_id: {action_id}")


def move_to_dict(move: Move) -> Dict[str, object]:
    return {
        "action_id": move.action_id,
        "path": list(move.path),
        "captures": list(move.captures),
        "is_capture": bool(move.is_capture),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate draughts self-play JSONL data.")
    parser.add_argument("--config", default=str(DEFAULT_SELFPLAY_CONFIG_PATH))
    parser.add_argument("--black", default=None, choices=available_agents())
    parser.add_argument("--white", default=None, choices=available_agents())
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL dataset path.")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--opening-random-plies",
        type=int,
        default=None,
        help="Play N random moves before agents take over. "
        "Essential for deterministic agents like AlphaBeta to produce varied games.",
    )
    parser.add_argument(
        "--material-config",
        default=None,
        help="YAML file for MaterialAgent parameters.",
    )
    parser.add_argument(
        "--alphabeta-config",
        default=None,
        help="YAML file for AlphaBetaAgent parameters.",
    )
    parser.add_argument(
        "--black-checkpoint",
        default=None,
        help="Checkpoint path for black SupervisedAgent.",
    )
    parser.add_argument(
        "--white-checkpoint",
        default=None,
        help="Checkpoint path for white SupervisedAgent.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SupervisedAgent/MCTSAgent: cpu, cuda, or cuda:0. Defaults to auto.",
    )
    parser.add_argument(
        "--mcts-config",
        default=None,
        help="YAML file for MCTSAgent parameters.",
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append samples to --output and continue game_id from the existing file.",
    )
    parser.add_argument(
        "--auto-output-name",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If --output exists, write to output_001.jsonl, output_002.jsonl, etc.",
    )
    cli_args = parser.parse_args()
    args = merge_selfplay_args(cli_args)

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.black == "supervised" and not args.black_checkpoint:
        parser.error("--black supervised requires --black-checkpoint")
    if args.white == "supervised" and not args.white_checkpoint:
        parser.error("--white supervised requires --white-checkpoint")
    if args.black == "mcts" and not args.black_checkpoint:
        parser.error("--black mcts requires --black-checkpoint")
    if args.white == "mcts" and not args.white_checkpoint:
        parser.error("--white mcts requires --white-checkpoint")
    if args.append and args.auto_output_name:
        parser.error("--append and --auto-output-name cannot be used together")
    return args


def merge_selfplay_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_selfplay_config(cli_args.config)
    values = asdict(config)
    cli_overrides = {
        "black": cli_args.black,
        "white": cli_args.white,
        "games": cli_args.games,
        "max_plies": cli_args.max_plies,
        "output": cli_args.output,
        "log_interval": cli_args.log_interval,
        "seed": cli_args.seed,
        "opening_random_plies": cli_args.opening_random_plies,
        "material_config": cli_args.material_config,
        "alphabeta_config": cli_args.alphabeta_config,
        "black_checkpoint": cli_args.black_checkpoint,
        "white_checkpoint": cli_args.white_checkpoint,
        "device": cli_args.device,
        "append": cli_args.append,
        "auto_output_name": cli_args.auto_output_name,
    }
    for key, value in cli_overrides.items():
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


def load_selfplay_config(config_path: str | Path) -> SelfPlayConfig:
    path = Path(config_path)
    defaults = asdict(SelfPlayConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(path, defaults, "selfplay"))
    config = SelfPlayConfig(**values)
    validate_selfplay_config(config)
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
        default_value = defaults[key]
        if value.lower() in {"none", "null"}:
            values[key] = None
            continue
        try:
            if isinstance(default_value, int) and not isinstance(default_value, bool):
                values[key] = int(value)
            else:
                values[key] = value
        except ValueError as exc:
            raise ValueError(
                f"{config_name} config value has wrong type on line {line_number}"
            ) from exc
    return values


def validate_selfplay_config(config: SelfPlayConfig) -> None:
    if config.black not in available_agents():
        raise ValueError(f"unknown black agent in selfplay config: {config.black}")
    if config.white not in available_agents():
        raise ValueError(f"unknown white agent in selfplay config: {config.white}")
    if config.games <= 0:
        raise ValueError("selfplay games must be positive")
    if config.max_plies <= 0:
        raise ValueError("selfplay max_plies must be positive")
    if config.log_interval <= 0:
        raise ValueError("selfplay log_interval must be positive")
    if config.append not in {0, 1}:
        raise ValueError("selfplay append must be 0 or 1")
    if config.auto_output_name not in {0, 1}:
        raise ValueError("selfplay auto_output_name must be 0 or 1")
    if config.opening_random_plies < 0:
        raise ValueError("selfplay opening_random_plies must be non-negative")
    if config.append and config.auto_output_name:
        raise ValueError("selfplay append and auto_output_name cannot both be enabled")


def main() -> None:
    args = parse_args()
    generate_dataset(
        black_name=args.black,
        white_name=args.white,
        games=args.games,
        max_plies=args.max_plies,
        output_path=args.output,
        log_interval=args.log_interval,
        seed=args.seed,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        black_checkpoint_path=args.black_checkpoint,
        white_checkpoint_path=args.white_checkpoint,
        mcts_config_path=args.mcts_config,
        device=args.device,
        append=args.append,
        auto_output_name=args.auto_output_name,
        opening_random_plies=args.opening_random_plies,
    )


if __name__ == "__main__":
    main()
