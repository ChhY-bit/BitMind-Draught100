from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


EXPECTED_SCHEMA_VERSION = 1
OBSERVATION_SHAPE = (6, 10, 10)
ACTION_MASK_LENGTH = 512
BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3


REQUIRED_FIELDS = {
    "schema_version",
    "game_id",
    "ply",
    "side_to_move",
    "raw_squares",
    "observation",
    "legal_action_ids",
    "action_mask",
    "selected_action_id",
    "final_result",
    "max_plies_reached",
    "value_target",
}


@dataclass
class GameSummary:
    final_result: int = 0
    max_plies_reached: bool = False
    sample_count: int = 0
    max_ply: int = -1


@dataclass
class DatasetReport:
    files: int = 0
    samples: int = 0
    valid_samples: int = 0
    invalid_json_lines: int = 0
    invalid_samples: int = 0
    missing_fields: Counter = field(default_factory=Counter)
    schema_versions: Counter = field(default_factory=Counter)
    value_targets: Counter = field(default_factory=Counter)
    final_results: Counter = field(default_factory=Counter)
    side_to_move: Counter = field(default_factory=Counter)
    agents: Counter = field(default_factory=Counter)
    games: Dict[Tuple[str, int], GameSummary] = field(default_factory=dict)
    validation_errors: Counter = field(default_factory=Counter)

    @property
    def game_count(self) -> int:
        return len(self.games)

    @property
    def black_wins(self) -> int:
        return count_games_by_result(self.games.values(), BLACK_WIN, False)

    @property
    def white_wins(self) -> int:
        return count_games_by_result(self.games.values(), WHITE_WIN, False)

    @property
    def draws(self) -> int:
        return count_games_by_result(self.games.values(), DRAW, False)

    @property
    def max_plies_reached(self) -> int:
        return sum(1 for game in self.games.values() if game.max_plies_reached)

    @property
    def avg_plies(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.sample_count for game in self.games.values()) / len(self.games)

    @property
    def ok(self) -> bool:
        return self.invalid_json_lines == 0 and self.invalid_samples == 0


def inspect_paths(paths: Sequence[str]) -> DatasetReport:
    files = resolve_input_files(paths)
    report = DatasetReport(files=len(files))
    for path in files:
        inspect_file(path, report)
    return report


def resolve_input_files(paths: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.jsonl")))
        else:
            resolved.append(path)
    return resolved


def inspect_file(path: Path, report: DatasetReport) -> None:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        report.invalid_samples += 1
        report.validation_errors["file_open_error"] += 1
        return

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            report.samples += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                report.invalid_json_lines += 1
                continue

            errors = validate_sample(sample)
            if errors:
                report.invalid_samples += 1
                report.validation_errors.update(errors)
                for field in REQUIRED_FIELDS - set(sample):
                    report.missing_fields[field] += 1
                continue

            update_report(report, path, sample)


def validate_sample(sample: object) -> List[str]:
    try:
        return validate_sample_inner(sample)
    except (TypeError, ValueError):
        return ["invalid_field_type"]


def validate_sample_inner(sample: object) -> List[str]:
    if not isinstance(sample, dict):
        return ["sample_not_object"]

    errors: List[str] = []
    missing = REQUIRED_FIELDS - set(sample)
    if missing:
        errors.append("missing_required_field")
        return errors

    if int(sample["schema_version"]) != EXPECTED_SCHEMA_VERSION:
        errors.append("unexpected_schema_version")
    if not isinstance(sample["raw_squares"], list) or len(sample["raw_squares"]) != 50:
        errors.append("invalid_raw_squares_length")
    if not has_observation_shape(sample["observation"], OBSERVATION_SHAPE):
        errors.append("invalid_observation_shape")
    if not isinstance(sample["action_mask"], list) or len(sample["action_mask"]) != ACTION_MASK_LENGTH:
        errors.append("invalid_action_mask_length")
    if int(sample["value_target"]) not in {-1, 0, 1}:
        errors.append("invalid_value_target")
    if int(sample["side_to_move"]) not in {0, 1}:
        errors.append("invalid_side_to_move")

    legal_action_ids = sample["legal_action_ids"]
    selected_action_id = int(sample["selected_action_id"])
    action_mask = sample["action_mask"]
    if not isinstance(legal_action_ids, list):
        errors.append("invalid_legal_action_ids")
        return errors
    if selected_action_id not in legal_action_ids:
        errors.append("selected_action_not_legal")
    if not legal_action_ids:
        errors.append("empty_legal_actions")
    if isinstance(action_mask, list) and len(action_mask) == ACTION_MASK_LENGTH:
        for action_id in legal_action_ids:
            if not isinstance(action_id, int) or action_id < 0 or action_id >= ACTION_MASK_LENGTH:
                errors.append("legal_action_id_out_of_range")
                break
            if int(action_mask[action_id]) != 1:
                errors.append("legal_action_missing_in_mask")
                break
        if 0 <= selected_action_id < ACTION_MASK_LENGTH and int(action_mask[selected_action_id]) != 1:
            errors.append("selected_action_missing_in_mask")

    return errors


def has_observation_shape(value: object, shape: Tuple[int, int, int]) -> bool:
    channels, rows, cols = shape
    if not isinstance(value, list) or len(value) != channels:
        return False
    for channel in value:
        if not isinstance(channel, list) or len(channel) != rows:
            return False
        for row in channel:
            if not isinstance(row, list) or len(row) != cols:
                return False
    return True


def update_report(report: DatasetReport, path: Path, sample: Dict[str, object]) -> None:
    report.valid_samples += 1
    report.schema_versions[int(sample["schema_version"])] += 1
    report.value_targets[int(sample["value_target"])] += 1
    report.final_results[int(sample["final_result"])] += 1
    report.side_to_move[int(sample["side_to_move"])] += 1
    report.agents[str(sample.get("agent", "unknown"))] += 1

    game_key = (str(path), int(sample["game_id"]))
    game = report.games.setdefault(game_key, GameSummary())
    game.final_result = int(sample["final_result"])
    game.max_plies_reached = bool(sample["max_plies_reached"])
    game.sample_count += 1
    game.max_ply = max(game.max_ply, int(sample["ply"]))


def count_games_by_result(
    games: Iterable[GameSummary],
    result: int,
    maxed: bool,
) -> int:
    return sum(
        1
        for game in games
        if game.final_result == result and game.max_plies_reached == maxed
    )


def format_report(report: DatasetReport) -> str:
    lines = [
        "dataset inspection",
        f"files: {report.files}",
        f"samples: {report.samples}",
        f"valid_samples: {report.valid_samples}",
        f"invalid_json_lines: {report.invalid_json_lines}",
        f"invalid_samples: {report.invalid_samples}",
        f"games: {report.game_count}",
        f"black_wins: {report.black_wins}",
        f"white_wins: {report.white_wins}",
        f"draws: {report.draws}",
        f"max_plies_reached: {report.max_plies_reached}",
        f"avg_plies: {report.avg_plies:.2f}",
        "value_target: " + format_counter(report.value_targets, [-1, 0, 1]),
        "side_to_move: " + format_counter(report.side_to_move, [0, 1]),
        "final_result: " + format_counter(report.final_results, [0, 1, 2, 3]),
        "agents: " + format_counter(report.agents, sorted(report.agents)),
        "validation_errors: "
        + (format_counter(report.validation_errors, sorted(report.validation_errors)) or "none"),
        f"status: {'ok' if report.ok else 'invalid'}",
    ]
    return "\n".join(lines)


def format_counter(counter: Counter, keys: Sequence[object]) -> str:
    if not counter and not keys:
        return ""
    return " ".join(f"{key}={counter.get(key, 0)}" for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect draughts JSONL training data.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories. Directories are searched recursively for *.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_paths(args.paths)
    print(format_report(report))


if __name__ == "__main__":
    main()
