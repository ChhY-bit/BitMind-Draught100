from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

from agents import Agent, available_agents, make_agent
from env import StandardDraughtsEnv


BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3


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
) -> tuple[int, int, bool]:
    env = StandardDraughtsEnv()
    env.reset()

    for ply in range(max_plies):
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
    device: Optional[str] = None,
) -> ArenaStats:
    black_agent = make_agent(
        black_name,
        seed=seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
        supervised_checkpoint_path=black_checkpoint_path,
        device=device,
    )
    white_seed = None if seed is None else seed + 1
    white_agent = make_agent(
        white_name,
        seed=white_seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
        supervised_checkpoint_path=white_checkpoint_path,
        device=device,
    )
    stats = ArenaStats()
    start = time.perf_counter()

    print(
        f"arena: black={black_agent.name} white={white_agent.name} "
        f"games={games} max_plies={max_plies} seed={seed} "
        f"material_config={material_config_path or 'default'} "
        f"alphabeta_config={alphabeta_config_path or 'default'} "
        f"black_checkpoint={black_checkpoint_path or 'none'} "
        f"white_checkpoint={white_checkpoint_path or 'none'}"
    )

    for game_index in range(1, games + 1):
        result, plies, maxed = play_game(black_agent, white_agent, max_plies)
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
    parser.add_argument("--black", default="random", choices=available_agents())
    parser.add_argument("--white", default="greedy", choices=available_agents())
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
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
        help="Device for SupervisedAgent: cpu, cuda, or cuda:0. Defaults to auto.",
    )
    args = parser.parse_args()

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
    return args


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
        device=args.device,
    )


if __name__ == "__main__":
    main()
