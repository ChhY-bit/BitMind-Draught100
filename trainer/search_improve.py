from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agents import AlphaBetaAgent, order_moves
from dataset import iter_jsonl_samples
from draughts_env import Move
from env import StandardDraughtsEnv
from selfplay import move_to_dict


@dataclass
class SearchImproveStats:
    samples_read: int = 0
    samples_sampled: int = 0
    samples_written: int = 0
    skipped: int = 0
    filtered: int = 0


def improve_dataset(
    input_paths: Sequence[str],
    output_path: str,
    max_samples: Optional[int] = None,
    log_interval: int = 100,
    seed: int = 1,
    alphabeta_config_path: Optional[str] = None,
    alphabeta_depth: Optional[int] = None,
    min_score_margin: int = 0,
    position_sample_rate: float = 1.0,
) -> SearchImproveStats:
    if not input_paths:
        raise ValueError("input_paths must not be empty")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when set")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if min_score_margin < 0:
        raise ValueError("min_score_margin must be non-negative")
    if not 0 < position_sample_rate <= 1:
        raise ValueError("position_sample_rate must be in (0, 1]")

    rng = random.Random(seed)
    teacher = AlphaBetaAgent(seed=seed, config_path=alphabeta_config_path)
    if alphabeta_depth is not None:
        if alphabeta_depth <= 0:
            raise ValueError("alphabeta_depth must be positive when set")
        teacher.config.search_depth = alphabeta_depth

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = StandardDraughtsEnv()
    stats = SearchImproveStats()
    start = time.perf_counter()

    print(
        f"search_improve: inputs={','.join(input_paths)} output={output} "
        f"max_samples={max_samples or 'all'} seed={seed} "
        f"alphabeta_config={alphabeta_config_path or 'default'} "
        f"alphabeta_depth={teacher.config.search_depth} "
        f"min_score_margin={min_score_margin} "
        f"position_sample_rate={position_sample_rate:g}",
        flush=True,
    )

    with output.open("w", encoding="utf-8") as handle:
        for sample in iter_jsonl_samples(input_paths):
            if max_samples is not None and stats.samples_written >= max_samples:
                break

            stats.samples_read += 1
            if position_sample_rate < 1.0 and rng.random() > position_sample_rate:
                continue
            stats.samples_sampled += 1
            try:
                improved = improve_sample(
                    sample,
                    env,
                    teacher,
                    rng,
                    min_score_margin=min_score_margin,
                )
            except LowConfidencePosition as exc:
                stats.filtered += 1
                if stats.filtered <= 5 or stats.filtered % log_interval == 0:
                    print(
                        f"search_improve: filter sample #{stats.samples_read}: {exc}",
                        flush=True,
                    )
                continue
            except ValueError as exc:
                stats.skipped += 1
                print(
                    f"search_improve: skip sample #{stats.samples_read}: {exc}",
                    flush=True,
                )
                continue

            handle.write(json.dumps(improved, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            stats.samples_written += 1

            if stats.samples_written % log_interval == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"search_improve: written={stats.samples_written} "
                    f"read={stats.samples_read} sampled={stats.samples_sampled} "
                    f"skipped={stats.skipped} filtered={stats.filtered} "
                    f"elapsed={elapsed:.3f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    print(
        f"search_improve: done written={stats.samples_written} "
        f"read={stats.samples_read} sampled={stats.samples_sampled} "
        f"skipped={stats.skipped} filtered={stats.filtered} elapsed={elapsed:.3f}s",
        flush=True,
    )
    return stats


def improve_sample(
    sample: Dict[str, object],
    env: StandardDraughtsEnv,
    teacher: AlphaBetaAgent,
    rng: random.Random,
    min_score_margin: int = 0,
) -> Dict[str, object]:
    if "raw_squares" not in sample:
        raise ValueError("missing raw_squares")
    if "side_to_move" not in sample:
        raise ValueError("missing side_to_move")

    raw_squares = list(sample["raw_squares"])
    side_to_move = int(sample["side_to_move"])
    env.set_state(raw_squares, side_to_move)
    raw_observation = env.raw_observation()
    legal_actions = env.legal_actions()
    if not legal_actions:
        raise ValueError("position has no legal actions")

    scored_moves = score_legal_actions(env, raw_observation, legal_actions, teacher)
    best_score = max(score for score, _move in scored_moves)
    best_candidates = [move for score, move in scored_moves if score == best_score]
    selected_move = rng.choice(best_candidates)
    margin = score_margin(scored_moves)
    if margin < min_score_margin:
        raise LowConfidencePosition(
            f"search_score_margin {margin} below minimum {min_score_margin}"
        )

    improved = dict(sample)
    improved["schema_version"] = sample.get("schema_version", 1)
    improved["side_to_move"] = side_to_move
    improved["side_to_move_name"] = raw_observation["side_to_move_name"]
    improved["agent"] = "search_improved"
    improved["raw_squares"] = list(raw_observation["squares"])
    improved["observation"] = env.observation()
    improved["legal_action_ids"] = [move.action_id for move in legal_actions]
    improved["legal_moves"] = [move_to_dict(move) for move in legal_actions]
    improved["action_mask"] = env.action_mask()
    improved["selected_action_id"] = selected_move.action_id
    improved["selected_move"] = move_to_dict(selected_move)
    improved["teacher_agent"] = teacher.name
    improved["teacher_depth"] = teacher.config.search_depth
    improved["teacher_original_agent"] = sample.get("agent", "unknown")
    improved["teacher_original_selected_action_id"] = sample.get("selected_action_id")
    improved["search_best_score"] = best_score
    improved["search_scores"] = [
        {
            "action_id": move.action_id,
            "score": score,
            "path": list(move.path),
            "captures": list(move.captures),
        }
        for score, move in sorted(scored_moves, key=lambda item: item[1].action_id)
    ]
    improved["search_score_margin"] = margin
    return improved


def score_legal_actions(
    env: StandardDraughtsEnv,
    raw_observation: Dict[str, object],
    legal_actions: Sequence[Move],
    teacher: AlphaBetaAgent,
) -> List[Tuple[int, Move]]:
    root_side = int(raw_observation["side_to_move"])
    scored_moves: List[Tuple[int, Move]] = []
    for move in order_moves(legal_actions, raw_observation):
        cloned = env.clone()
        cloned.step(move.action_id)
        score = teacher._search(
            cloned,
            depth=teacher.config.search_depth - 1,
            root_side=root_side,
            alpha=-10**12,
            beta=10**12,
        )
        scored_moves.append((score, move))
    return scored_moves


def score_margin(scored_moves: Sequence[Tuple[int, Move]]) -> int:
    scores = sorted((score for score, _move in scored_moves), reverse=True)
    if len(scores) < 2:
        return 0
    return scores[0] - scores[1]


class LowConfidencePosition(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate search-improved JSONL policy targets from existing samples."
    )
    parser.add_argument("inputs", nargs="+", help="Input JSONL file(s) or directories.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--min-score-margin",
        type=int,
        default=0,
        help="Keep only positions where the best AlphaBeta move beats runner-up by at least this score.",
    )
    parser.add_argument(
        "--position-sample-rate",
        type=float,
        default=1.0,
        help="Randomly score this fraction of input samples before margin filtering.",
    )
    parser.add_argument(
        "--alphabeta-config",
        default=None,
        help="YAML file for AlphaBetaAgent parameters. Defaults to trainer/config/alphabeta.yaml.",
    )
    parser.add_argument(
        "--alphabeta-depth",
        type=int,
        default=None,
        help="Override AlphaBeta search depth for the teacher.",
    )
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.alphabeta_depth is not None and args.alphabeta_depth <= 0:
        parser.error("--alphabeta-depth must be positive")
    if args.min_score_margin < 0:
        parser.error("--min-score-margin must be non-negative")
    if not 0 < args.position_sample_rate <= 1:
        parser.error("--position-sample-rate must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    improve_dataset(
        input_paths=args.inputs,
        output_path=args.output,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        seed=args.seed,
        alphabeta_config_path=args.alphabeta_config,
        alphabeta_depth=args.alphabeta_depth,
        min_score_margin=args.min_score_margin,
        position_sample_rate=args.position_sample_rate,
    )


if __name__ == "__main__":
    main()
