from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from draughts.agents import Agent, available_agents, make_agent
from draughts.env import StandardDraughtsEnv
from selfplay import load_scalar_yaml


BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3
DEFAULT_ARENA_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "arena.yaml"


@dataclass
class ArenaConfig:
    black: str = "random"
    white: str = "random"
    games: int = 100
    max_plies: int = 300
    log_interval: int = 20
    seed: int = 1
    opening_random_plies: int = 0
    material_config: Optional[str] = None
    alphabeta_config: Optional[str] = None
    black_checkpoint: Optional[str] = None
    white_checkpoint: Optional[str] = None
    mcts_config: Optional[str] = None
    device: Optional[str] = None


@dataclass
class ArenaStats:
    games: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    max_plies_reached: int = 0
    total_plies: int = 0

    def record(self, game_result: int, plies: int, maxed: bool) -> None:
        self.games += 1
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


def play_game(
    black_agent: Agent,
    white_agent: Agent,
    max_plies: int,
    opening_random_plies: int = 0,
    opening_rng: Optional[random.Random] = None,
) -> tuple[int, int, bool]:
    env = StandardDraughtsEnv()
    env.reset()

    if opening_random_plies < 0:
        raise ValueError("opening_random_plies must be non-negative")
    if opening_random_plies > 0 and opening_rng is None:
        raise ValueError("opening_rng is required when opening_random_plies is positive")

    plies, opening_finished = apply_random_opening(
        env,
        opening_random_plies,
        max_plies,
        opening_rng,
    )
    if opening_finished:
        raw = env.raw_observation()
        return int(raw["game_result"]), plies, plies >= max_plies

    for ply in range(plies, max_plies):
        raw = env.raw_observation()
        if raw["game_result_name"] != "ongoing":
            return int(raw["game_result"]), ply, False

        legal_actions = env.legal_actions()
        if not legal_actions:
            return int(raw["game_result"]), ply, False

        agent = black_agent if int(raw["side_to_move"]) == 0 else white_agent
        action_id = agent.select_action(raw, legal_actions, env)
        env.step(action_id)

    raw = env.raw_observation()
    return int(raw["game_result"]), max_plies, True


def apply_random_opening(
    env: StandardDraughtsEnv,
    opening_random_plies: int,
    max_plies: int,
    rng: Optional[random.Random],
) -> tuple[int, bool]:
    """Advance one legal random opening before evaluated agents start moving."""
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


def run_arena(
    black_name: str,
    white_name: str,
    games: int,
    max_plies: int,
    log_interval: int,
    seed: Optional[int],
    material_config_path: Optional[str] = None,
    alphabeta_config_path: Optional[str] = None,
    black_checkpoint_path: Optional[str] = None,
    white_checkpoint_path: Optional[str] = None,
    mcts_config_path: Optional[str] = None,
    device: Optional[str] = None,
    opening_random_plies: int = 0,
) -> ArenaStats:
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
    stats = ArenaStats()
    start = time.perf_counter()
    opening_rng = random.Random(None if seed is None else seed + 2)

    print(
        f"arena: black={black_agent.name} white={white_agent.name} "
        f"games={games} max_plies={max_plies} seed={seed} "
        f"opening_random_plies={opening_random_plies} "
        f"material_config={material_config_path or 'default'} "
        f"alphabeta_config={alphabeta_config_path or 'default'} "
        f"black_checkpoint={black_checkpoint_path or 'none'} "
        f"white_checkpoint={white_checkpoint_path or 'none'}"
    )

    for game_index in range(1, games + 1):
        result, plies, maxed = play_game(
            black_agent,
            white_agent,
            max_plies,
            opening_random_plies=opening_random_plies,
            opening_rng=opening_rng,
        )
        stats.record(result, plies, maxed)

        if game_index % log_interval == 0 or game_index == games:
            elapsed = time.perf_counter() - start
            print(
                f"arena: {stats.games}/{games} games, "
                f"black_wins={stats.black_wins} white_wins={stats.white_wins} "
                f"draws={stats.draws} max_plies={stats.max_plies_reached} "
                f"avg_plies={stats.avg_plies:.2f} elapsed={elapsed:.3f}s"
            )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline draughts agents against each other.")
    parser.add_argument("--config", default=str(DEFAULT_ARENA_CONFIG_PATH))
    parser.add_argument("--black", default=None, choices=available_agents())
    parser.add_argument("--white", default=None, choices=available_agents())
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--opening-random-plies",
        type=int,
        default=None,
        help="Random legal half-moves to play before evaluated agents start.",
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
    args = merge_arena_args(parser.parse_args())

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.opening_random_plies < 0:
        parser.error("--opening-random-plies must be non-negative")
    if args.black == "supervised" and not args.black_checkpoint:
        parser.error("--black supervised requires --black-checkpoint")
    if args.white == "supervised" and not args.white_checkpoint:
        parser.error("--white supervised requires --white-checkpoint")
    if args.black == "mcts" and not args.black_checkpoint:
        parser.error("--black mcts requires --black-checkpoint")
    if args.white == "mcts" and not args.white_checkpoint:
        parser.error("--white mcts requires --white-checkpoint")
    return args


def merge_arena_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_arena_config(cli_args.config)
    values = asdict(config)
    for key in (
        "black",
        "white",
        "games",
        "max_plies",
        "log_interval",
        "seed",
        "opening_random_plies",
        "material_config",
        "alphabeta_config",
        "black_checkpoint",
        "white_checkpoint",
        "device",
    ):
        value = getattr(cli_args, key)
        if value is not None:
            values[key] = value
    values["config"] = cli_args.config
    return argparse.Namespace(**values)


def load_arena_config(config_path: str | Path) -> ArenaConfig:
    path = Path(config_path)
    defaults = asdict(ArenaConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(path, defaults, "arena"))
    config = ArenaConfig(**values)
    validate_arena_config(config)
    return config


def validate_arena_config(config: ArenaConfig) -> None:
    if config.black not in available_agents():
        raise ValueError(f"unknown black agent in arena config: {config.black}")
    if config.white not in available_agents():
        raise ValueError(f"unknown white agent in arena config: {config.white}")
    if config.games <= 0:
        raise ValueError("arena games must be positive")
    if config.max_plies <= 0:
        raise ValueError("arena max_plies must be positive")
    if config.log_interval <= 0:
        raise ValueError("arena log_interval must be positive")
    if config.opening_random_plies < 0:
        raise ValueError("arena opening_random_plies must be non-negative")


def main() -> None:
    args = parse_args()
    run_arena(
        black_name=args.black,
        white_name=args.white,
        games=args.games,
        max_plies=args.max_plies,
        log_interval=args.log_interval,
        seed=args.seed,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        black_checkpoint_path=args.black_checkpoint,
        white_checkpoint_path=args.white_checkpoint,
        mcts_config_path=args.mcts_config,
        device=args.device,
        opening_random_plies=args.opening_random_plies,
    )


if __name__ == "__main__":
    main()
