from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agents import Agent, available_agents, make_agent
from draughts_env import Move
from env import StandardDraughtsEnv


BLACK_SIDE = 0
WHITE_SIDE = 1
BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3


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
) -> SelfPlayStats:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")

    black_agent = make_agent(
        black_name,
        seed=seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
    )
    white_seed = None if seed is None else seed + 1
    white_agent = make_agent(
        white_name,
        seed=white_seed,
        material_config_path=material_config_path,
        alphabeta_config_path=alphabeta_config_path,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = SelfPlayStats()
    start = time.perf_counter()

    print(
        f"selfplay: black={black_agent.name} white={white_agent.name} "
        f"games={games} max_plies={max_plies} seed={seed} output={output} "
        f"material_config={material_config_path or 'default'} "
        f"alphabeta_config={alphabeta_config_path or 'default'}"
    )

    with output.open("w", encoding="utf-8") as handle:
        for game_id in range(games):
            samples, game_result, plies, maxed = play_and_collect_game(
                game_id=game_id,
                black_agent=black_agent,
                white_agent=white_agent,
                max_plies=max_plies,
            )
            for sample in finalize_samples(samples, game_result, maxed):
                handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

            stats.record(game_result, plies, maxed, len(samples))

            completed_games = game_id + 1
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


def play_and_collect_game(
    game_id: int,
    black_agent: Agent,
    white_agent: Agent,
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

        side_to_move = int(raw["side_to_move"])
        agent = black_agent if side_to_move == BLACK_SIDE else white_agent
        action_id = agent.select_action(raw, legal_actions, env)
        selected_move = find_selected_move(legal_actions, action_id)

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
            )
        )
        env.step(action_id)

    raw = env.raw_observation()
    return samples, int(raw["game_result"]), max_plies, True


def build_sample(
    game_id: int,
    ply: int,
    raw_observation: Dict[str, object],
    encoded_observation: List[List[List[int]]],
    action_mask: Sequence[int],
    legal_actions: Sequence[Move],
    selected_move: Move,
    agent_name: str,
) -> Dict[str, object]:
    side_to_move = int(raw_observation["side_to_move"])
    return {
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
    parser.add_argument("--black", default="alphabeta", choices=available_agents())
    parser.add_argument("--white", default="alphabeta", choices=available_agents())
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--output", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--log-interval", type=int, default=1)
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
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    return args


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
    )


if __name__ == "__main__":
    main()
