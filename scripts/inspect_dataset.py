from __future__ import annotations

import argparse
import json
import os
import sys
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


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def inspect_paths(
    paths: Sequence[str],
    *,
    max_samples: int | None = None,
    quick: bool = False,
    show_progress: bool = True,
) -> DatasetReport:
    """Inspect all JSONL files under *paths* (files or directories).

    Args:
        paths: JSONL files or directories (recursively searched).
        max_samples: Stop after this many total samples (useful for large datasets).
        quick: Skip expensive deep checks (observation shape, action-mask
               consistency) — ~2× faster on large files.
        show_progress: Print progress to stderr every 5 %.
    """
    files = resolve_input_files(paths)
    report = DatasetReport(files=len(files))

    # Pre-scan total bytes for progress bar.
    total_bytes = 0
    for f in files:
        try:
            total_bytes += f.stat().st_size
        except OSError:
            pass

    bytes_read = 0
    last_pct = -1

    for path in files:
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            report.invalid_samples += 1
            report.validation_errors["file_open_error"] += 1
            if total_bytes > 0:
                try:
                    bytes_read += path.stat().st_size
                except OSError:
                    pass
            continue

        with handle:
            for raw_line in handle:
                if max_samples is not None and report.samples >= max_samples:
                    break

                # Track bytes for progress (encode overhead is negligible vs JSON parse).
                bytes_read += len(raw_line.encode("utf-8"))

                line = raw_line.strip()
                if not line:
                    continue
                report.samples += 1

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    report.invalid_json_lines += 1
                    continue

                errors = validate_sample(sample, quick=quick)
                if errors:
                    report.invalid_samples += 1
                    report.validation_errors.update(errors)
                    for field in REQUIRED_FIELDS - set(sample):
                        report.missing_fields[field] += 1
                    continue

                update_report(report, path, sample)

                # ---- progress ----
                if show_progress and total_bytes > 0:
                    pct = int(bytes_read * 100 / total_bytes)
                    if pct - last_pct >= 5:
                        mb_read = bytes_read // (1024 * 1024)
                        mb_total = total_bytes // (1024 * 1024)
                        print(
                            f"\r  inspecting: {min(pct, 99)}% ({mb_read}/{mb_total} MB, "
                            f"{report.samples} samples)  ",
                            end="", file=sys.stderr, flush=True,
                        )
                        last_pct = pct

        if max_samples is not None and report.samples >= max_samples:
            break

    if show_progress and total_bytes > 0:
        print(
            f"\r  inspecting: 100% — {report.samples} samples  ",
            file=sys.stderr, flush=True,
        )

    return report


# ---------------------------------------------------------------------------
# 文件发现
# ---------------------------------------------------------------------------


def resolve_input_files(paths: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.jsonl")))
        else:
            resolved.append(path)
    return resolved


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def validate_sample(sample: object, *, quick: bool = False) -> List[str]:
    """Validate a single JSONL sample dict.

    Args:
        sample: The parsed JSON object.
        quick: If True, skip expensive deep checks (observation shape,
               action-mask consistency).  ~2× faster on large datasets.
    """
    try:
        return _validate(sample, quick=quick)
    except (TypeError, ValueError):
        return ["invalid_field_type"]


def _validate(sample: object, *, quick: bool) -> List[str]:
    if not isinstance(sample, dict):
        return ["sample_not_object"]

    errors: List[str] = []
    missing = REQUIRED_FIELDS - set(sample)
    if missing:
        errors.append("missing_required_field")
        return errors

    # --- cheap checks always run ---
    if int(sample["schema_version"]) != EXPECTED_SCHEMA_VERSION:
        errors.append("unexpected_schema_version")
    if not isinstance(sample["raw_squares"], list) or len(sample["raw_squares"]) != 50:
        errors.append("invalid_raw_squares_length")
    if not isinstance(sample["action_mask"], list) or len(sample["action_mask"]) != ACTION_MASK_LENGTH:
        errors.append("invalid_action_mask_length")
    if int(sample["value_target"]) not in {-1, 0, 1}:
        errors.append("invalid_value_target")
    if int(sample["side_to_move"]) not in {0, 1}:
        errors.append("invalid_side_to_move")

    legal_action_ids = sample["legal_action_ids"]
    selected_action_id = int(sample["selected_action_id"])
    if not isinstance(legal_action_ids, list):
        errors.append("invalid_legal_action_ids")
        return errors
    if selected_action_id not in legal_action_ids:
        errors.append("selected_action_not_legal")
    if not legal_action_ids:
        errors.append("empty_legal_actions")

    # --- expensive checks: skipped in quick mode ---
    if not quick:
        if not _check_observation_shape(sample["observation"], OBSERVATION_SHAPE):
            errors.append("invalid_observation_shape")

        action_mask = sample["action_mask"]
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


def _check_observation_shape(value: object, shape: Tuple[int, int, int]) -> bool:
    """Check that *value* is a nested list of the given (C, H, W) shape.

    Micro-optimised: flatten the 3-deep check into a tight loop.
    """
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


# ---------------------------------------------------------------------------
# 统计汇总
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect draughts JSONL training data.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories. Directories are searched recursively for *.jsonl.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip expensive deep checks (observation shape, action-mask consistency). "
        "~2× faster on large datasets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Stop after inspecting N samples (useful for quick spot-checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_paths(
        args.paths,
        max_samples=args.max_samples,
        quick=args.quick,
        show_progress=os.isatty(sys.stderr.fileno()),
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
