from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from .move_encoding import MOVE_FEATURE_DIM, encode_legal_moves, selected_move_index
from .env import encode_observation


OBSERVATION_SHAPE = (6, 10, 10)
ACTION_MASK_LENGTH = 512


@dataclass(frozen=True)
class DataSource:
    path: str
    policy_weight: float = 1.0
    value_weight: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class _SampleIndex:
    """Lightweight reference to a single JSONL sample on disk.

    Stores just enough to seek back and read the line on demand.
    """
    file_path: str
    byte_offset: int
    source_index: int  # index into DraughtsJsonlDataset.sources


def _build_sample_index(
    sources: Sequence[DataSource],
    validate: bool = False,
    cache_in_memory: bool = True,
    show_progress: bool = True,
) -> tuple[list[_SampleIndex], list[dict[str, object]] | None]:
    """Scan JSONL files and record byte offsets for random access.

    When *validate* is True every sample is fully parsed and checked
    (useful for first-time data inspection).  When *cache_in_memory* is
    True the raw parsed JSON dicts are kept so that ``__getitem__`` can
    skip the disk read entirely.

    Returns ``(index, cache)`` where *cache* is a list of pre-parsed
    sample dicts when *cache_in_memory* is True, or ``None`` otherwise.
    """
    # Resolve all files first so we can report progress by byte position.
    file_entries: list[tuple[int, Path]] = []
    for source_index, source in enumerate(sources):
        validate_data_source(source)
        for file_path in resolve_jsonl_files([source.path]):
            file_entries.append((source_index, file_path))

    total_bytes = 0
    for _, file_path in file_entries:
        try:
            total_bytes += os.path.getsize(str(file_path))
        except OSError:
            pass

    index: list[_SampleIndex] = []
    cache: list[dict[str, object]] | None = [] if cache_in_memory else None
    bytes_read = 0
    last_pct = -1

    for source_index, file_path in file_entries:
        with open(file_path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                bytes_read += len(line)
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue

                sample: dict[str, object] | None = None
                if validate or cache_in_memory:
                    sample = json.loads(line_str)
                    if not isinstance(sample, dict):
                        raise ValueError(
                            f"sample must be a JSON object: {file_path}:{offset}"
                        )
                if validate and sample is not None:
                    validate_training_sample(sample, len(index))

                if cache is not None and sample is not None:
                    cache.append(sample)

                index.append(_SampleIndex(
                    file_path=str(file_path),
                    byte_offset=offset,
                    source_index=source_index,
                ))

                # Progress display (stderr so it doesn't pollute stdout).
                if show_progress and total_bytes > 0:
                    pct = int(bytes_read * 100 / total_bytes)
                    if pct - last_pct >= 5:  # every 5 % to limit noise
                        mb_read = bytes_read // (1024 * 1024)
                        mb_total = total_bytes // (1024 * 1024)
                        print(
                            f"\r  loading dataset: {pct}% ({mb_read}/{mb_total} MB)  ",
                            end="", file=sys.stderr, flush=True,
                        )
                        last_pct = pct

    if show_progress and total_bytes > 0:
        print(
            f"\r  loading dataset: 100% — {len(index)} samples  ",
            file=sys.stderr, flush=True,
        )

    return index, cache


def resolve_jsonl_files(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        else:
            files.append(path)
    return files


def iter_jsonl_samples(paths: Sequence[str]) -> Iterator[Dict[str, object]]:
    for path in resolve_jsonl_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError(f"sample must be a JSON object: {path}:{line_number}")
                yield sample


def iter_weighted_jsonl_samples(sources: Sequence[DataSource]) -> Iterator[Dict[str, object]]:
    for source_index, source in enumerate(sources):
        validate_data_source(source)
        for path in resolve_jsonl_files([source.path]):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    if not isinstance(sample, dict):
                        raise ValueError(f"sample must be a JSON object: {path}:{line_number}")
                    yield {
                        "sample": sample,
                        "source_index": source_index,
                        "source_name": source.name or str(path),
                        "source_path": str(path),
                        "policy_weight": source.policy_weight,
                        "value_weight": source.value_weight,
                    }


class DraughtsJsonlDataset:
    """Map-style dataset backed by on-disk JSONL files.

    By default samples are pre-loaded into memory during construction
    (``cache_in_memory=True``) so that ``__getitem__`` incurs no disk
    I/O or JSON parsing.  For large datasets (typical of MCTS self-play),
    always use ``cache_in_memory=False`` (via ``--no-cache`` in the trainer)
    to avoid exhausting system RAM.  The I/O overhead is negligible compared
    to GPU training time.  In that mode each ``__getitem__`` reads its line
    from disk on demand.

    Validation is opt-in via ``validate=True`` (or the ``.validate()``
    method) — useful for first-time data inspection but skipped by
    default to keep construction fast.
    """

    def __init__(
        self,
        paths: Sequence[str],
        validate: bool = False,
        cache_in_memory: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.sources = [DataSource(path=str(path)) for path in paths]
        self._index, self._cache = _build_sample_index(
            self.sources, validate=validate, cache_in_memory=cache_in_memory,
            show_progress=show_progress,
        )

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[DataSource],
        validate: bool = False,
        cache_in_memory: bool = True,
        show_progress: bool = True,
    ) -> "DraughtsJsonlDataset":
        dataset = cls.__new__(cls)
        dataset.sources = list(sources)
        dataset._index, dataset._cache = _build_sample_index(
            dataset.sources, validate=validate, cache_in_memory=cache_in_memory,
            show_progress=show_progress,
        )
        return dataset

    def validate(self) -> list[str]:
        """Validate all cached (or on-disk) samples and return error messages.

        Call this explicitly when you have new data and want to catch
        malformed samples before training.
        """
        errors: list[str] = []
        for i in range(len(self._index)):
            try:
                sample = self._load_raw_sample(i)
                validate_training_sample(sample, i)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"sample {i}: {exc}")
        return errors

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self._index[index]
        source = self.sources[entry.source_index]
        sample = self._load_raw_sample(index)
        return training_item_from_sample(
            sample,
            policy_weight=float(source.policy_weight),
            value_weight=float(source.value_weight),
            source_name=source.name or entry.file_path,
            source_path=entry.file_path,
        )

    def raw_sample(self, index: int) -> Dict[str, object]:
        """Return the raw JSON sample dict at *index* (for introspection)."""
        return self._load_raw_sample(index)

    def _load_raw_sample(self, index: int) -> Dict[str, object]:
        """Return the parsed JSON dict for *index*.

        Uses the in-memory cache when available; otherwise reads from disk.
        """
        if self._cache is not None:
            return self._cache[index]

        entry = self._index[index]
        with open(entry.file_path, 'rb') as f:
            f.seek(entry.byte_offset)
            line = f.readline()
        sample = json.loads(line.decode('utf-8'))
        if not isinstance(sample, dict):
            raise ValueError(
                f"sample must be a JSON object: {entry.file_path}:{entry.byte_offset}"
            )
        return sample


def training_item_from_sample(
    sample: Dict[str, object],
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    source_name: str = "",
    source_path: str = "",
) -> Dict[str, object]:
    legal_moves = sample["legal_moves"]
    selected_move = sample["selected_move"]
    side_to_move = int(sample["side_to_move"])
    policy_target_probs = sample.get("policy_target_probs")
    return {
        "observation": observation_from_sample(sample),
        "action_mask": sample["action_mask"],
        "policy_target": int(sample["selected_action_id"]),
        "move_features": encode_legal_moves(legal_moves, side_to_move=side_to_move),
        "policy_target_index": selected_move_index(legal_moves, selected_move),
        "policy_target_probs": (
            [float(p) for p in policy_target_probs]
            if policy_target_probs is not None
            else None
        ),
        "value_target": int(sample["value_target"]),    # 提取标签
        "selected_policy_prob": float(sample.get("selected_policy_prob", 1.0)),
        "behavior_temperature": float(sample.get("temperature", 1.0)),
        "behavior_value": float(sample.get("predicted_value", 0.0)),
        "policy_weight": float(policy_weight),
        "value_weight": float(value_weight),
        "metadata": {
            "game_id": int(sample["game_id"]),
            "ply": int(sample["ply"]),
            "side_to_move": int(sample["side_to_move"]),
            "agent": sample.get("agent", "unknown"),
            "final_result": int(sample["final_result"]),
            "max_plies_reached": bool(sample["max_plies_reached"]),
            "source_name": source_name,
            "source_path": source_path,
        },
    }


def observation_from_sample(sample: Dict[str, object]) -> object:
    if "raw_squares" in sample and "side_to_move" in sample:
        return encode_observation(
            {
                "squares": sample["raw_squares"],
                "side_to_move": int(sample["side_to_move"]),
            }
        )
    return sample["observation"]


def validate_data_source(source: DataSource) -> None:
    if source.policy_weight < 0:
        raise ValueError(f"policy_weight must be non-negative for data source: {source.path}")
    if source.value_weight < 0:
        raise ValueError(f"value_weight must be non-negative for data source: {source.path}")
    if source.policy_weight == 0 and source.value_weight == 0:
        raise ValueError(f"policy_weight and value_weight cannot both be zero: {source.path}")


def validate_training_sample(sample: Dict[str, object], index: int = -1) -> None:
    prefix = f"sample {index}: " if index >= 0 else ""
    observation = sample.get("observation")
    action_mask = sample.get("action_mask")
    selected_action_id = sample.get("selected_action_id")
    value_target = sample.get("value_target")
    legal_moves = sample.get("legal_moves")
    selected_move = sample.get("selected_move")

    if not has_observation_shape(observation, OBSERVATION_SHAPE):
        raise ValueError(f"{prefix}observation must be 6x10x10")
    if not isinstance(action_mask, list) or len(action_mask) != ACTION_MASK_LENGTH:
        raise ValueError(f"{prefix}action_mask must have length {ACTION_MASK_LENGTH}")
    if not isinstance(selected_action_id, int):
        raise ValueError(f"{prefix}selected_action_id must be an integer")
    if selected_action_id < 0 or selected_action_id >= ACTION_MASK_LENGTH:
        raise ValueError(f"{prefix}selected_action_id out of range")
    if int(action_mask[selected_action_id]) != 1:
        raise ValueError(f"{prefix}selected action must be enabled in action_mask")
    if int(value_target) not in {-1, 0, 1}:     # 对数据的局面进行硬校验，分别表示负、平、胜
        raise ValueError(f"{prefix}value_target must be -1, 0, or 1")
    if not isinstance(legal_moves, list) or not legal_moves:
        raise ValueError(f"{prefix}legal_moves must be a non-empty list")
    if not isinstance(selected_move, dict):
        raise ValueError(f"{prefix}selected_move must be an object")
    selected_move_index(legal_moves, selected_move)

    policy_target_probs = sample.get("policy_target_probs")
    if policy_target_probs is not None:
        if not isinstance(policy_target_probs, list) or len(policy_target_probs) != len(legal_moves):
            raise ValueError(
                f"{prefix}policy_target_probs must be a list with same length as legal_moves"
            )
        for prob in policy_target_probs:
            if not isinstance(prob, (int, float)) or float(prob) < -0.001:
                raise ValueError(f"{prefix}policy_target_probs values must be non-negative")
        prob_sum = sum(float(p) for p in policy_target_probs)
        if prob_sum < 0.99 or prob_sum > 1.01:
            raise ValueError(
                f"{prefix}policy_target_probs must sum to ~1.0, got {prob_sum:.6f}"
            )


def has_observation_shape(value: object, shape: tuple[int, int, int]) -> bool:
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


def collate_batch(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate samples into a batch and convert to tensors in the worker.

    Converting to tensors **inside** the DataLoader worker is critical
    for performance: torch Tensors use shared memory for IPC whereas
    nested Python lists would be pickled, which dominates training time
    for small models.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for collate_batch()") from exc

    max_moves = max((len(item["move_features"]) for item in items), default=0)
    has_soft_targets = any(item.get("policy_target_probs") is not None for item in items)

    batch: dict[str, object] = {
        "observation": torch.tensor(
            [item["observation"] for item in items], dtype=torch.float32,
        ),
        "action_mask": torch.tensor(
            [item["action_mask"] for item in items], dtype=torch.float32,
        ),
        "policy_target": torch.tensor(
            [item["policy_target"] for item in items], dtype=torch.long,
        ),
        "move_features": torch.tensor(
            [pad_move_features(item["move_features"], max_moves) for item in items],
            dtype=torch.float32,
        ),
        "move_mask": torch.tensor(
            [
                [1] * len(item["move_features"]) + [0] * (max_moves - len(item["move_features"]))
                for item in items
            ],
            dtype=torch.float32,
        ),
        "policy_target_index": torch.tensor(
            [item["policy_target_index"] for item in items], dtype=torch.long,
        ),
        "value_target": torch.tensor(
            [item["value_target"] for item in items], dtype=torch.float32,
        ),
        "selected_policy_prob": torch.tensor(
            [item.get("selected_policy_prob", 1.0) for item in items], dtype=torch.float32,
        ),
        "behavior_temperature": torch.tensor(
            [item.get("behavior_temperature", 1.0) for item in items], dtype=torch.float32,
        ),
        "behavior_value": torch.tensor(
            [item.get("behavior_value", 0.0) for item in items], dtype=torch.float32,
        ),
        "policy_weight": torch.tensor(
            [item.get("policy_weight", 1.0) for item in items], dtype=torch.float32,
        ),
        "value_weight": torch.tensor(
            [item.get("value_weight", 1.0) for item in items], dtype=torch.float32,
        ),
        "metadata": [item["metadata"] for item in items],
    }
    if has_soft_targets:
        batch["policy_target_probs"] = torch.tensor(
            [
                pad_probs(item.get("policy_target_probs"), len(item["move_features"]), max_moves)
                for item in items
            ],
            dtype=torch.float32,
        )
    return batch


def pad_move_features(move_features: Sequence[Sequence[float]], max_moves: int) -> List[List[float]]:
    padded = [list(features) for features in move_features]
    padded.extend([[0.0] * MOVE_FEATURE_DIM for _index in range(max_moves - len(padded))])
    return padded


def pad_probs(
    probs: Optional[Sequence[float]],
    num_legal: int,
    max_moves: int,
) -> List[float]:
    """Pad a probability distribution to max_moves, filling with zeros.

    Returns a zero-filled list when probs is None (backward compat with
    data that has no soft targets).
    """
    if probs is None:
        return [0.0] * max_moves
    padded = [float(p) for p in probs]
    padded.extend([0.0] * (max_moves - len(padded)))
    return padded


def to_torch_batch(batch: Dict[str, object]) -> Dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for to_torch_batch()") from exc

    result: dict[str, object] = {
        "observation": torch.tensor(batch["observation"], dtype=torch.float32),
        "action_mask": torch.tensor(batch["action_mask"], dtype=torch.float32),
        "policy_target": torch.tensor(batch["policy_target"], dtype=torch.long),
        "move_features": torch.tensor(batch["move_features"], dtype=torch.float32),
        "move_mask": torch.tensor(batch["move_mask"], dtype=torch.float32),
        "policy_target_index": torch.tensor(batch["policy_target_index"], dtype=torch.long),
        "value_target": torch.tensor(batch["value_target"], dtype=torch.float32),
        "selected_policy_prob": torch.tensor(batch["selected_policy_prob"], dtype=torch.float32),
        "behavior_temperature": torch.tensor(batch["behavior_temperature"], dtype=torch.float32),
        "behavior_value": torch.tensor(batch["behavior_value"], dtype=torch.float32),
        "policy_weight": torch.tensor(batch["policy_weight"], dtype=torch.float32),
        "value_weight": torch.tensor(batch["value_weight"], dtype=torch.float32),
        "metadata": batch["metadata"],
    }
    if batch.get("policy_target_probs") is not None:
        result["policy_target_probs"] = torch.tensor(
            batch["policy_target_probs"], dtype=torch.float32
        )
    return result


def batched(items: Iterable[Dict[str, object]], batch_size: int) -> Iterator[List[Dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: List[Dict[str, object]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
