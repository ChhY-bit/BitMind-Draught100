from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from agents import available_agents
from arena import ArenaStats, run_arena


DEFAULT_COMPARE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "compare.yaml"
SUPPORTED_EXTERNAL_AGENTS = {"random", "greedy", "material", "alphabeta"}
SUPPORTED_PRESETS = {"custom", "quick", "standard", "full", "alphabeta"}
PRESET_OVERRIDES: dict[str, dict[str, object]] = {
    "quick": {
        "games": 50,
        "seeds": "2028",
        "opening_random_plies": "4",
        "external_agents": "",
    },
    "standard": {
        "games": 100,
        "seeds": "2028,2029",
        "opening_random_plies": "2,8",
        "external_agents": "material",
    },
    "full": {
        "games": 100,
        "seeds": "2028,2029",
        "opening_random_plies": "2,8",
        "external_agents": "material,alphabeta",
    },
    "alphabeta": {
        "games": 50,
        "seeds": "2028",
        "opening_random_plies": "8",
        "external_agents": "alphabeta",
    },
}


@dataclass
class CompareConfig:
    preset: str = "custom"
    games: int = 100
    max_plies: int = 300
    log_interval: int = 100
    seeds: str = "2028,2029"
    opening_random_plies: str = "2,8"
    external_agents: str = "material"
    promote_threshold: float = 0.53
    reject_threshold: float = 0.50
    bucket_floor: float = 0.45
    report: Optional[str] = None
    material_config: Optional[str] = None
    alphabeta_config: Optional[str] = None
    device: Optional[str] = None


@dataclass
class Score:
    candidate_wins: int = 0
    baseline_wins: int = 0
    draws: int = 0
    max_plies: int = 0
    games: int = 0
    total_plies: int = 0

    @property
    def decisive_games(self) -> int:
        return self.candidate_wins + self.baseline_wins

    @property
    def candidate_win_rate(self) -> float:
        if self.decisive_games == 0:
            return 0.0
        return self.candidate_wins / self.decisive_games

    @property
    def avg_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_plies / self.games

    def add(self, other: "Score") -> None:
        self.candidate_wins += other.candidate_wins
        self.baseline_wins += other.baseline_wins
        self.draws += other.draws
        self.max_plies += other.max_plies
        self.games += other.games
        self.total_plies += other.total_plies


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    report = compare_checkpoints(args)
    print_report(report, time.perf_counter() - started)
    if args.report:
        write_report(Path(args.report), report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two supervised checkpoints across openings.")
    parser.add_argument("--config", default=str(DEFAULT_COMPARE_CONFIG_PATH))
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path.")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path.")
    parser.add_argument(
        "--preset",
        choices=sorted(SUPPORTED_PRESETS - {"custom"}),
        default=None,
        help="Evaluation preset: quick, standard, full, or alphabeta.",
    )
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 2028,2029.")
    parser.add_argument(
        "--opening-random-plies",
        default=None,
        help="Comma-separated opening depths, e.g. 2,8.",
    )
    parser.add_argument(
        "--external-agents",
        default=None,
        help="Comma-separated external agents: random,greedy,material,alphabeta.",
    )
    parser.add_argument("--promote-threshold", type=float, default=None)
    parser.add_argument("--reject-threshold", type=float, default=None)
    parser.add_argument("--bucket-floor", type=float, default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--material-config", default=None)
    parser.add_argument("--alphabeta-config", default=None)
    parser.add_argument("--device", default=None)
    args = merge_compare_args(parser.parse_args())
    validate_compare_args(args, parser)
    return args


def merge_compare_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_compare_config(cli_args.config)
    values = asdict(config)
    preset = cli_args.preset or str(values["preset"])
    if preset != "custom":
        values.update(PRESET_OVERRIDES[preset])
    values["preset"] = preset
    for key in (
        "preset",
        "games",
        "max_plies",
        "log_interval",
        "seeds",
        "opening_random_plies",
        "external_agents",
        "promote_threshold",
        "reject_threshold",
        "bucket_floor",
        "report",
        "material_config",
        "alphabeta_config",
        "device",
    ):
        value = getattr(cli_args, key)
        if value is not None:
            values[key] = value
    values["config"] = cli_args.config
    values["candidate"] = cli_args.candidate
    values["baseline"] = cli_args.baseline
    return argparse.Namespace(**values)


def load_compare_config(config_path: str | Path) -> CompareConfig:
    defaults = asdict(CompareConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(Path(config_path), defaults, "compare"))
    config = CompareConfig(**values)
    validate_compare_config(config)
    return config


def load_scalar_yaml(path: Path, defaults: dict[str, object], config_name: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{config_name} config not found: {path}")
    values: dict[str, object] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid {config_name} config line {line_number}: {raw_line}")
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
            raise ValueError(
                f"{config_name} config value has wrong type on line {line_number}"
            ) from exc
    return values


def validate_compare_config(config: CompareConfig) -> None:
    if config.preset not in SUPPORTED_PRESETS:
        raise ValueError(f"compare preset must be one of: {','.join(sorted(SUPPORTED_PRESETS))}")
    if config.games <= 0:
        raise ValueError("compare games must be positive")
    if config.max_plies <= 0:
        raise ValueError("compare max_plies must be positive")
    if config.log_interval <= 0:
        raise ValueError("compare log_interval must be positive")
    parse_int_list(config.seeds, "seeds")
    openings = parse_int_list(config.opening_random_plies, "opening_random_plies")
    if any(value < 0 for value in openings):
        raise ValueError("compare opening_random_plies must be non-negative")
    parse_external_agents(config.external_agents)
    if not 0 <= config.bucket_floor <= 1:
        raise ValueError("compare bucket_floor must be in [0, 1]")
    if not 0 <= config.reject_threshold <= 1:
        raise ValueError("compare reject_threshold must be in [0, 1]")
    if not 0 <= config.promote_threshold <= 1:
        raise ValueError("compare promote_threshold must be in [0, 1]")
    if config.reject_threshold > config.promote_threshold:
        raise ValueError("compare reject_threshold must be <= promote_threshold")


def validate_compare_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.candidate == args.baseline:
        parser.error("--candidate and --baseline must be different paths")
    if not Path(args.candidate).exists():
        parser.error(f"--candidate not found: {args.candidate}")
    if not Path(args.baseline).exists():
        parser.error(f"--baseline not found: {args.baseline}")
    try:
        seeds = parse_int_list(args.seeds, "seeds")
        openings = parse_int_list(args.opening_random_plies, "opening_random_plies")
        externals = parse_external_agents(args.external_agents)
    except ValueError as exc:
        parser.error(str(exc))
    if not seeds:
        parser.error("--seeds must not be empty")
    if not openings:
        parser.error("--opening-random-plies must not be empty")
    if any(value < 0 for value in openings):
        parser.error("--opening-random-plies values must be non-negative")
    for agent in externals:
        if agent not in available_agents():
            parser.error(f"unknown external agent: {agent}")
        if agent == "supervised":
            parser.error("external supervised checkpoints are not supported in this tool")


def parse_int_list(raw: str, name: str) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return []
    try:
        return [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc


def parse_external_agents(raw: Optional[str]) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    agents = [part.strip() for part in str(raw).split(",") if part.strip()]
    unknown = [agent for agent in agents if agent not in SUPPORTED_EXTERNAL_AGENTS]
    if unknown:
        raise ValueError(f"unsupported external agents: {','.join(unknown)}")
    return agents


def compare_checkpoints(args: argparse.Namespace) -> dict[str, object]:
    seeds = parse_int_list(args.seeds, "seeds")
    openings = parse_int_list(args.opening_random_plies, "opening_random_plies")
    external_agents = parse_external_agents(args.external_agents)
    settings = {
        "preset": args.preset,
        "games": args.games,
        "max_plies": args.max_plies,
        "log_interval": args.log_interval,
        "seeds": seeds,
        "opening_random_plies": openings,
        "external_agents": external_agents,
        "promote_threshold": args.promote_threshold,
        "reject_threshold": args.reject_threshold,
        "bucket_floor": args.bucket_floor,
        "material_config": args.material_config,
        "alphabeta_config": args.alphabeta_config,
        "device": args.device,
    }

    main_buckets: list[dict[str, object]] = []
    main_overall = Score()
    for opening in openings:
        for seed in seeds:
            bucket = run_checkpoint_pair(args, seed, opening)
            main_buckets.append(bucket)
            main_overall.add(score_from_dict(bucket["score"]))

    warnings = build_main_warnings(main_buckets, args.bucket_floor)
    recommendation = recommend(main_overall, main_buckets, args)

    external: dict[str, object] = {}
    for agent in external_agents:
        external[agent] = run_external_pair(args, agent, seeds, openings)
        warning = external_warning(agent, external[agent])
        if warning:
            warnings.append(warning)

    return {
        "candidate": args.candidate,
        "baseline": args.baseline,
        "settings": settings,
        "main": {
            "buckets": main_buckets,
            "overall": score_to_dict(main_overall),
            "recommendation": recommendation,
        },
        "external": external,
        "warnings": warnings,
    }


def run_checkpoint_pair(args: argparse.Namespace, seed: int, opening: int) -> dict[str, object]:
    candidate_black = run_arena(
        black_name="supervised",
        white_name="supervised",
        games=args.games,
        max_plies=args.max_plies,
        log_interval=args.log_interval,
        seed=seed,
        black_checkpoint_path=args.candidate,
        white_checkpoint_path=args.baseline,
        device=args.device,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        opening_random_plies=opening,
    )
    baseline_black = run_arena(
        black_name="supervised",
        white_name="supervised",
        games=args.games,
        max_plies=args.max_plies,
        log_interval=args.log_interval,
        seed=seed,
        black_checkpoint_path=args.baseline,
        white_checkpoint_path=args.candidate,
        device=args.device,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        opening_random_plies=opening,
    )
    score = paired_score(candidate_black, baseline_black)
    return {
        "seed": seed,
        "opening_random_plies": opening,
        "candidate_black": arena_stats_to_dict(candidate_black),
        "baseline_black": arena_stats_to_dict(baseline_black),
        "score": score_to_dict(score),
    }


def run_external_pair(
    args: argparse.Namespace,
    agent: str,
    seeds: Sequence[int],
    openings: Sequence[int],
) -> dict[str, object]:
    candidate_buckets: list[dict[str, object]] = []
    baseline_buckets: list[dict[str, object]] = []
    candidate_overall = Score()
    baseline_overall = Score()
    for opening in openings:
        for seed in seeds:
            candidate_bucket = run_model_vs_external(args, args.candidate, agent, seed, opening)
            baseline_bucket = run_model_vs_external(args, args.baseline, agent, seed, opening)
            candidate_buckets.append(candidate_bucket)
            baseline_buckets.append(baseline_bucket)
            candidate_overall.add(score_from_dict(candidate_bucket["score"]))
            baseline_overall.add(score_from_dict(baseline_bucket["score"]))
    return {
        "candidate": {
            "buckets": candidate_buckets,
            "overall": score_to_dict(candidate_overall),
        },
        "baseline": {
            "buckets": baseline_buckets,
            "overall": score_to_dict(baseline_overall),
        },
        "delta": {
            "wins": candidate_overall.candidate_wins - baseline_overall.candidate_wins,
            "losses": candidate_overall.baseline_wins - baseline_overall.baseline_wins,
            "win_rate": candidate_overall.candidate_win_rate - baseline_overall.candidate_win_rate,
        },
    }


def run_model_vs_external(
    args: argparse.Namespace,
    checkpoint: str,
    agent: str,
    seed: int,
    opening: int,
) -> dict[str, object]:
    model_black = run_arena(
        black_name="supervised",
        white_name=agent,
        games=args.games,
        max_plies=args.max_plies,
        log_interval=args.log_interval,
        seed=seed,
        black_checkpoint_path=checkpoint,
        device=args.device,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        opening_random_plies=opening,
    )
    external_black = run_arena(
        black_name=agent,
        white_name="supervised",
        games=args.games,
        max_plies=args.max_plies,
        log_interval=args.log_interval,
        seed=seed,
        white_checkpoint_path=checkpoint,
        device=args.device,
        material_config_path=args.material_config,
        alphabeta_config_path=args.alphabeta_config,
        opening_random_plies=opening,
    )
    score = paired_score(model_black, external_black)
    return {
        "seed": seed,
        "opening_random_plies": opening,
        "model_black": arena_stats_to_dict(model_black),
        "external_black": arena_stats_to_dict(external_black),
        "score": score_to_dict(score),
    }


def paired_score(model_black: ArenaStats, opponent_black: ArenaStats) -> Score:
    return Score(
        candidate_wins=model_black.black_wins + opponent_black.white_wins,
        baseline_wins=model_black.white_wins + opponent_black.black_wins,
        draws=model_black.draws + opponent_black.draws,
        max_plies=model_black.max_plies_reached + opponent_black.max_plies_reached,
        games=model_black.games + opponent_black.games,
        total_plies=model_black.total_plies + opponent_black.total_plies,
    )


def recommend(overall: Score, buckets: Sequence[dict[str, object]], args: argparse.Namespace) -> str:
    if overall.decisive_games <= 0:
        return "inconclusive"
    if overall.candidate_win_rate < args.reject_threshold:
        return "reject"
    if overall.candidate_win_rate < args.promote_threshold:
        return "inconclusive"
    for bucket in buckets:
        score = score_from_dict(bucket["score"])
        if score.decisive_games > 0 and score.candidate_win_rate < args.bucket_floor:
            return "inconclusive"
    return "promote"


def build_main_warnings(buckets: Sequence[dict[str, object]], bucket_floor: float) -> list[str]:
    warnings: list[str] = []
    for bucket in buckets:
        score = score_from_dict(bucket["score"])
        if score.decisive_games > 0 and score.candidate_win_rate < bucket_floor:
            warnings.append(
                "candidate below bucket floor "
                f"for seed={bucket['seed']} opening={bucket['opening_random_plies']}: "
                f"{score.candidate_win_rate:.3f}"
            )
    return warnings


def external_warning(agent: str, result: object) -> Optional[str]:
    data = result if isinstance(result, dict) else {}
    candidate = score_from_dict(data["candidate"]["overall"])
    baseline = score_from_dict(data["baseline"]["overall"])
    if candidate.candidate_wins + 2 < baseline.candidate_wins:
        return (
            f"candidate won fewer games than baseline against {agent}: "
            f"{candidate.candidate_wins} vs {baseline.candidate_wins}"
        )
    return None


def arena_stats_to_dict(stats: ArenaStats) -> dict[str, object]:
    return {
        "games": stats.games,
        "black_wins": stats.black_wins,
        "white_wins": stats.white_wins,
        "draws": stats.draws,
        "max_plies": stats.max_plies_reached,
        "total_plies": stats.total_plies,
        "avg_plies": stats.avg_plies,
    }


def score_to_dict(score: Score) -> dict[str, object]:
    return {
        "candidate_wins": score.candidate_wins,
        "baseline_wins": score.baseline_wins,
        "draws": score.draws,
        "max_plies": score.max_plies,
        "games": score.games,
        "total_plies": score.total_plies,
        "avg_plies": score.avg_plies,
        "decisive_games": score.decisive_games,
        "candidate_win_rate": score.candidate_win_rate,
    }


def score_from_dict(data: object) -> Score:
    value = data if isinstance(data, dict) else {}
    return Score(
        candidate_wins=int(value.get("candidate_wins", 0)),
        baseline_wins=int(value.get("baseline_wins", 0)),
        draws=int(value.get("draws", 0)),
        max_plies=int(value.get("max_plies", 0)),
        games=int(value.get("games", 0)),
        total_plies=int(value.get("total_plies", 0)),
    )


def print_report(report: dict[str, object], elapsed: float) -> None:
    print("\ncompare_checkpoints summary")
    print(f"candidate: {report['candidate']}")
    print(f"baseline:  {report['baseline']}")
    print(f"preset:    {report['settings']['preset']}")
    print("\nmain candidate vs baseline")
    for bucket in report["main"]["buckets"]:
        score = score_from_dict(bucket["score"])
        print(
            f"opening={bucket['opening_random_plies']} seed={bucket['seed']}: "
            f"candidate={score.candidate_wins} baseline={score.baseline_wins} "
            f"maxed={score.max_plies} wr={score.candidate_win_rate:.3f}"
        )
    overall = score_from_dict(report["main"]["overall"])
    print(
        "overall: "
        f"candidate={overall.candidate_wins} baseline={overall.baseline_wins} "
        f"maxed={overall.max_plies} decisive_wr={overall.candidate_win_rate:.3f}"
    )
    print(f"recommendation: {report['main']['recommendation']}")
    if report["external"]:
        print("\nexternal baselines")
        for agent, result in report["external"].items():
            candidate = score_from_dict(result["candidate"]["overall"])
            baseline = score_from_dict(result["baseline"]["overall"])
            delta = result["delta"]
            print(
                f"{agent}: candidate_wins={candidate.candidate_wins} "
                f"baseline_wins={baseline.candidate_wins} "
                f"delta_wins={delta['wins']} "
                f"candidate_wr={candidate.candidate_win_rate:.3f} "
                f"baseline_wr={baseline.candidate_win_rate:.3f}"
            )
    if report["warnings"]:
        print("\nwarnings")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print(f"\nelapsed={elapsed:.3f}s")


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote report: {path}")


if __name__ == "__main__":
    main()
