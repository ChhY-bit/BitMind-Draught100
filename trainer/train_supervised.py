from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from draughts.dataset import DataSource, DraughtsJsonlDataset, collate_batch
from draughts.model import CandidateMoveNet, DEFAULT_MODEL_CONFIG_PATH, load_model_config


DEFAULT_SUPERVISED_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "supervised.yaml"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
TIME_LOG_FILENAME = "training_time.json"


@dataclass
class SupervisedConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    value_loss_weight: float = 0.25
    validation_split: float = 0.1
    seed: int = 1
    shuffle: int = 1
    log_interval: int = 20
    csv_log_interval_epochs: int = 1
    num_workers: int = 0


@dataclass
class EpochMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    policy_accuracy: float
    value_mae: float
    samples: int


def main() -> None:
    print("supervised: parsing arguments and loading configs...", flush=True)
    args = parse_args()
    print(f"supervised: loading train config from {args.config}", flush=True)
    config = load_supervised_config(args.config)
    config = apply_cli_overrides(config, args)
    print(f"supervised: loading model config from {args.model_config}", flush=True)
    model_config = load_model_config(args.model_config)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"supervised: using device={device}", flush=True)
    set_seed(config.seed)

    data_sources = resolve_data_sources(args.data, args.source_config, args.source)
    data_source_descriptions = describe_data_sources(data_sources)
    experiment_data_paths = experiment_data_identity(
        args.data,
        args.source_config,
        args.source,
        data_source_descriptions,
    )
    print(f"supervised: loading dataset from {', '.join(data_source_descriptions)}", flush=True)
    dataset = DraughtsJsonlDataset.from_sources(data_sources, cache_in_memory=not args.no_cache)
    print(f"supervised: loaded {len(dataset)} samples", flush=True)
    train_indices, val_indices = split_indices(len(dataset), config.validation_split, config.seed)
    print(
        f"supervised: split dataset train={len(train_indices)} val={len(val_indices)}",
        flush=True,
    )
    print("supervised: building dataloaders...", flush=True)
    train_loader = make_loader(dataset, train_indices, config, training=True)
    val_loader = make_loader(dataset, val_indices, config, training=False) if val_indices else None

    print("supervised: initializing model and optimizer...", flush=True)
    model = CandidateMoveNet(config=model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch = 0
    previous_cumulative_seconds = load_cumulative_seconds(
        checkpoint_dir,
        config,
        model_config,
        experiment_data_paths,
    )
    best_val_loss: Optional[float] = None

    if args.resume:
        print(f"supervised: loading checkpoint from {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        previous_cumulative_seconds = float(
            checkpoint.get("cumulative_train_seconds", previous_cumulative_seconds)
        )
        best_val_loss = checkpoint.get("best_val_loss")
        print(
            f"supervised: resumed from epoch={start_epoch} "
            f"cumulative_train_time={format_seconds(previous_cumulative_seconds)}",
            flush=True,
        )

    print(
        "supervised:"
        f" samples={len(dataset)} train={len(train_indices)} val={len(val_indices)}"
        f" epochs={config.epochs} batch_size={config.batch_size}"
        f" device={device} checkpoint_dir={checkpoint_dir}"
    )
    print("supervised: starting training loop...", flush=True)

    run_start = time.perf_counter()
    run_started_at_unix = time.time()
    key = experiment_key(config, model_config, experiment_data_paths)
    csv_log_path = checkpoint_dir / f"training_log_{key}.csv"
    latest_checkpoint: Optional[Path] = None
    final_epoch = start_epoch + config.epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device, epoch)
        val_metrics = evaluate(model, val_loader, config, device) if val_loader is not None else None
        elapsed_seconds = time.perf_counter() - run_start
        cumulative_seconds = previous_cumulative_seconds + elapsed_seconds
        is_best = val_metrics is None or best_val_loss is None or val_metrics.loss < best_val_loss
        if is_best:
            best_val_loss = val_metrics.loss if val_metrics is not None else train_metrics.loss

        print_epoch_summary(epoch, train_metrics, val_metrics, elapsed_seconds, cumulative_seconds)

        latest_checkpoint = save_checkpoint(
            checkpoint_dir / "candidate_move_net_latest.pt",
            model,
            optimizer,
            epoch,
            config,
            model_config,
            experiment_data_paths,
            train_metrics,
            val_metrics,
            elapsed_seconds,
            cumulative_seconds,
            best_val_loss,
        )
        if args.save_every_epoch or epoch == final_epoch:
            save_checkpoint(
                checkpoint_dir / f"candidate_move_net_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                model_config,
                experiment_data_paths,
                train_metrics,
                val_metrics,
                elapsed_seconds,
                cumulative_seconds,
                best_val_loss,
            )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "candidate_move_net_best.pt",
                model,
                optimizer,
                epoch,
                config,
                model_config,
                experiment_data_paths,
                train_metrics,
                val_metrics,
                elapsed_seconds,
                cumulative_seconds,
                best_val_loss,
            )

        update_time_log(
            checkpoint_dir,
            config,
            model_config,
            experiment_data_paths,
            cumulative_seconds,
            elapsed_seconds,
            epoch,
        )
        if should_write_csv_log(epoch, final_epoch, config.csv_log_interval_epochs):
            append_csv_log(
                csv_log_path,
                epoch,
                run_started_at_unix,
                train_metrics,
                val_metrics,
                elapsed_seconds,
                cumulative_seconds,
                best_val_loss,
            )

    if latest_checkpoint is not None:
        print(f"saved latest checkpoint: {latest_checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CandidateMoveNet with supervised data.")
    parser.add_argument("data", nargs="*", help="JSONL data file(s) or directory path(s).")
    parser.add_argument("--config", default=str(DEFAULT_SUPERVISED_CONFIG_PATH))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None, help="Checkpoint path to continue training from.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Defaults to auto.")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save candidate_move_net_epoch_XXX.pt after every epoch. Defaults to final epoch only.",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="YAML-like file that lists weighted data sources.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH:POLICY_WEIGHT:VALUE_WEIGHT[:NAME]",
        help="Add one weighted data source. Can be repeated.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config file).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size (overrides config file).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config file).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Do not preload entire JSONL into memory. Strongly recommended for large datasets to avoid OOM.",
    )
    args = parser.parse_args()
    if not args.data and not args.source_config and not args.source:
        parser.error("at least one data path, --source-config, or --source is required")
    return args


def apply_cli_overrides(config: SupervisedConfig, args: argparse.Namespace) -> SupervisedConfig:
    """Override SupervisedConfig fields with any non-None CLI values.

    Follows the same pattern as merge_selfplay_args() in scripts/selfplay.py.
    """
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    if not overrides:
        return config

    values = asdict(config)
    values.update(overrides)
    merged = SupervisedConfig(**values)
    validate_supervised_config(merged)
    if args.epochs is not None:
        print(f"supervised: CLI override epochs={args.epochs}", flush=True)
    if args.batch_size is not None:
        print(f"supervised: CLI override batch_size={args.batch_size}", flush=True)
    if args.lr is not None:
        print(f"supervised: CLI override lr={args.lr}", flush=True)
    return merged


def resolve_data_sources(
    data_paths: Sequence[str],
    source_config_path: Optional[str],
    source_args: Sequence[str],
) -> list[DataSource]:
    sources = [DataSource(path=str(path)) for path in data_paths]
    if source_config_path is not None:
        sources.extend(load_data_source_config(source_config_path))
    for raw_source in source_args:
        sources.append(parse_data_source_arg(raw_source))
    if not sources:
        raise ValueError("at least one data source is required")
    return sources


def parse_data_source_arg(raw_source: str) -> DataSource:
    parts = raw_source.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(
            "--source must use PATH:POLICY_WEIGHT:VALUE_WEIGHT[:NAME] format"
        )
    path, policy_weight, value_weight = parts[:3]
    name = parts[3] if len(parts) == 4 else ""
    return DataSource(
        path=path,
        policy_weight=float(policy_weight),
        value_weight=float(value_weight),
        name=name,
    )


def load_data_source_config(config_path: str | Path) -> list[DataSource]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"data source config not found: {path}")

    groups: dict[str, dict[str, str]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid data source config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        prefix, field = split_source_key(key, line_number)
        groups.setdefault(prefix, {})[field] = value

    sources: list[DataSource] = []
    for prefix in sorted(groups, key=source_sort_key):
        values = groups[prefix]
        if "path" not in values:
            raise ValueError(f"{prefix}_path is required in data source config")
        sources.append(
            DataSource(
                path=values["path"],
                policy_weight=float(values.get("policy_weight", "1.0")),
                value_weight=float(values.get("value_weight", "1.0")),
                name=values.get("name", ""),
            )
        )
    return sources


def split_source_key(key: str, line_number: int) -> tuple[str, str]:
    for suffix in ("_policy_weight", "_value_weight", "_path", "_name"):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix[1:]
    raise ValueError(f"unknown data source config key on line {line_number}: {key}")


def source_sort_key(prefix: str) -> tuple[str, int]:
    tail = prefix.rsplit("_", 1)[-1]
    return (prefix[: -len(tail)], int(tail) if tail.isdigit() else 0)


def describe_data_sources(sources: Sequence[DataSource]) -> list[str]:
    return [
        (
            f"{source.path}"
            f"|policy={source.policy_weight:g}"
            f"|value={source.value_weight:g}"
            f"|name={source.name or 'default'}"
        )
        for source in sources
    ]


def experiment_data_identity(
    data_paths: Sequence[str],
    source_config_path: Optional[str],
    source_args: Sequence[str],
    data_source_descriptions: Sequence[str],
) -> list[str]:
    if source_config_path is None and not source_args:
        return list(data_paths)
    return list(data_source_descriptions)


def load_supervised_config(config_path: str | Path) -> SupervisedConfig:
    path = Path(config_path)
    defaults = asdict(SupervisedConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(path, defaults, "supervised"))
    config = SupervisedConfig(**values)
    validate_supervised_config(config)
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
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in defaults:
            raise ValueError(f"unknown {config_name} config key on line {line_number}: {key}")
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


def validate_supervised_config(config: SupervisedConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.value_loss_weight < 0:
        raise ValueError("value_loss_weight must be non-negative")
    if not 0 <= config.validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    if config.shuffle not in {0, 1}:
        raise ValueError("shuffle must be 0 or 1")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.csv_log_interval_epochs <= 0:
        raise ValueError("csv_log_interval_epochs must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def split_indices(size: int, validation_split: float, seed: int) -> tuple[list[int], list[int]]:
    if size <= 0:
        raise ValueError("dataset must contain at least one sample")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    validation_size = int(size * validation_split)
    if validation_split > 0 and validation_size == 0 and size > 1:
        validation_size = 1
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    if not train_indices:
        train_indices = val_indices
        val_indices = []
    return train_indices, val_indices


def make_loader(
    dataset: DraughtsJsonlDataset,
    indices: Sequence[int],
    config: SupervisedConfig,
    training: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader_kwargs: dict[str, object] = dict(
        batch_size=config.batch_size,
        shuffle=bool(config.shuffle) if training else False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        generator=generator,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(Subset(dataset, list(indices)), **loader_kwargs)


def train_one_epoch(
    model: CandidateMoveNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: SupervisedConfig,
    device: torch.device,
    epoch: int,
) -> EpochMetrics:
    model.train()
    totals = MetricTotals()
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch_to_device(raw_batch, device)
        output = model(batch["observation"], batch["move_features"], batch["move_mask"])
        loss, policy_loss, value_loss = compute_loss(output, batch, config.value_loss_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals.update(output, batch, loss, policy_loss, value_loss)
        if batch_index % config.log_interval == 0:
            metrics = totals.to_metrics()
            print(
                f"epoch={epoch} batch={batch_index} loss={metrics.loss:.4f}"
                f" policy_loss={metrics.policy_loss:.4f}"
                f" value_loss={metrics.value_loss:.4f}"
                f" policy_acc={metrics.policy_accuracy:.3f}"
            )
    return totals.to_metrics()


def evaluate(
    model: CandidateMoveNet,
    loader: DataLoader,
    config: SupervisedConfig,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    totals = MetricTotals()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            output = model(batch["observation"], batch["move_features"], batch["move_mask"])
            loss, policy_loss, value_loss = compute_loss(output, batch, config.value_loss_weight)
            totals.update(output, batch, loss, policy_loss, value_loss)
    return totals.to_metrics()


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Policy loss: use soft targets (KL divergence) when available,
    # falling back to hard-label cross-entropy for legacy data.
    policy_logits = output["policy_logits"]  # [B, M]
    # 根据policy_target_probs选择硬标签或软标签
    if "policy_target_probs" in batch and batch["policy_target_probs"] is not None:
        policy_loss_per_sample = soft_cross_entropy(
            policy_logits,
            batch["policy_target_probs"],
            batch["move_mask"],
        )
    else:
        policy_loss_per_sample = nn.functional.cross_entropy(
            policy_logits,
            batch["policy_target_index"],
            reduction="none",
        )
    value_loss_per_sample = nn.functional.mse_loss(
        output["value"],
        batch["value_target"],
        reduction="none",
    )
    policy_loss = weighted_mean(policy_loss_per_sample, batch["policy_weight"])
    value_loss = weighted_mean(value_loss_per_sample, batch["value_weight"])
    loss = policy_loss + value_loss_weight * value_loss
    return loss, policy_loss, value_loss


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    move_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with a soft (distributional) target.

    Computes -sum(target_probs * log_softmax(logits)) per sample, masked
    so that padded move positions (where move_mask == 0) contribute zero.

    Args:
        logits: [B, M] raw policy logits (may contain MASKED_LOGIT for
                padded positions).
        target_probs: [B, M] teacher probability distribution, already
                      padded with zeros for invalid positions.
        move_mask: [B, M] 1 for legal moves, 0 for padded positions.

    Returns:
        [B] loss per sample.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)  # [B, M]
    # Masked positions have log_prob ≈ -inf after mask fill in forward();
    # zero them out so that target_probs * log_prob = 0 for those entries.
    safe_log_probs = torch.where(
        move_mask.bool(), log_probs, torch.zeros_like(log_probs)
    )
    # Sum over the move dimension: -target * log_prob for each legal move.
    per_sample = -(target_probs * safe_log_probs).sum(dim=-1)  # [B]
    return per_sample


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total_weight = weights.sum()
    if float(total_weight.item()) <= 0:
        return values.new_tensor(0.0)
    return (values * weights).sum() / total_weight


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


class MetricTotals:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.correct = 0
        self.value_abs_error_sum = 0.0
        self.samples = 0

    def update(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, object],
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
    ) -> None:
        targets = batch["policy_target_index"]
        value_targets = batch["value_target"]
        batch_size = int(targets.shape[0])
        predictions = output["policy_logits"].argmax(dim=1)
        self.loss_sum += float(loss.item()) * batch_size
        self.policy_loss_sum += float(policy_loss.item()) * batch_size
        self.value_loss_sum += float(value_loss.item()) * batch_size
        self.correct += int((predictions == targets).sum().item())
        self.value_abs_error_sum += float((output["value"] - value_targets).abs().sum().item())
        self.samples += batch_size

    def to_metrics(self) -> EpochMetrics:
        if self.samples <= 0:
            return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return EpochMetrics(
            loss=self.loss_sum / self.samples,
            policy_loss=self.policy_loss_sum / self.samples,
            value_loss=self.value_loss_sum / self.samples,
            policy_accuracy=self.correct / self.samples,
            value_mae=self.value_abs_error_sum / self.samples,
            samples=self.samples,
        )


def print_epoch_summary(
    epoch: int,
    train_metrics: EpochMetrics,
    val_metrics: Optional[EpochMetrics],
    elapsed_seconds: float,
    cumulative_seconds: float,
) -> None:
    message = (
        f"epoch={epoch} train_loss={train_metrics.loss:.4f}"
        f" train_policy_acc={train_metrics.policy_accuracy:.3f}"
        f" train_value_mae={train_metrics.value_mae:.4f}"
    )
    if val_metrics is not None:
        message += (
            f" val_loss={val_metrics.loss:.4f}"
            f" val_policy_acc={val_metrics.policy_accuracy:.3f}"
            f" val_value_mae={val_metrics.value_mae:.4f}"
        )
    message += (
        f" elapsed={format_seconds(elapsed_seconds)}"
        f" cumulative_train_time={format_seconds(cumulative_seconds)}"
    )
    print(message)


def save_checkpoint(
    path: Path,
    model: CandidateMoveNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: SupervisedConfig,
    model_config: object,
    data_paths: Sequence[str],
    train_metrics: EpochMetrics,
    val_metrics: Optional[EpochMetrics],
    elapsed_seconds: float,
    cumulative_seconds: float,
    best_val_loss: Optional[float],
) -> Path:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "supervised_config": asdict(config),
        "model_config": asdict(model_config),
        "data_paths": list(data_paths),
        "train_metrics": asdict(train_metrics),
        "val_metrics": asdict(val_metrics) if val_metrics is not None else None,
        "run_elapsed_seconds": elapsed_seconds,
        "cumulative_train_seconds": cumulative_seconds,
        "cumulative_train_time": format_seconds(cumulative_seconds),
        "best_val_loss": best_val_loss,
        "experiment_key": experiment_key(config, model_config, data_paths),
    }
    torch.save(checkpoint, path)
    return path


def load_cumulative_seconds(
    checkpoint_dir: Path,
    config: SupervisedConfig,
    model_config: object,
    data_paths: Sequence[str],
) -> float:
    time_log = read_time_log(checkpoint_dir)
    key = experiment_key(config, model_config, data_paths)
    return float(time_log.get(key, {}).get("cumulative_train_seconds", 0.0))


def update_time_log(
    checkpoint_dir: Path,
    config: SupervisedConfig,
    model_config: object,
    data_paths: Sequence[str],
    cumulative_seconds: float,
    run_elapsed_seconds: float,
    epoch: int,
) -> None:
    time_log = read_time_log(checkpoint_dir)
    key = experiment_key(config, model_config, data_paths)
    time_log[key] = {
        "cumulative_train_seconds": cumulative_seconds,
        "cumulative_train_time": format_seconds(cumulative_seconds),
        "last_run_elapsed_seconds": run_elapsed_seconds,
        "last_epoch": epoch,
        "supervised_config": asdict(config),
        "model_config": asdict(model_config),
        "data_paths": list(data_paths),
        "updated_at_unix": time.time(),
    }
    path = checkpoint_dir / TIME_LOG_FILENAME
    path.write_text(json.dumps(time_log, ensure_ascii=False, indent=2), encoding="utf-8")


def should_write_csv_log(epoch: int, final_epoch: int, interval_epochs: int) -> bool:
    return epoch == final_epoch or epoch % interval_epochs == 0


def append_csv_log(
    path: Path,
    epoch: int,
    run_started_at_unix: float,
    train_metrics: EpochMetrics,
    val_metrics: Optional[EpochMetrics],
    elapsed_seconds: float,
    cumulative_seconds: float,
    best_val_loss: Optional[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_log_fieldnames())
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "written_at_unix": f"{time.time():.6f}",
                "run_started_at_unix": f"{run_started_at_unix:.6f}",
                "epoch": epoch,
                "train_samples": train_metrics.samples,
                "train_loss": f"{train_metrics.loss:.8f}",
                "train_policy_loss": f"{train_metrics.policy_loss:.8f}",
                "train_value_loss": f"{train_metrics.value_loss:.8f}",
                "train_policy_accuracy": f"{train_metrics.policy_accuracy:.8f}",
                "train_value_mae": f"{train_metrics.value_mae:.8f}",
                "val_samples": val_metrics.samples if val_metrics is not None else 0,
                "val_loss": f"{val_metrics.loss:.8f}" if val_metrics is not None else "",
                "val_policy_loss": (
                    f"{val_metrics.policy_loss:.8f}" if val_metrics is not None else ""
                ),
                "val_value_loss": (
                    f"{val_metrics.value_loss:.8f}" if val_metrics is not None else ""
                ),
                "val_policy_accuracy": (
                    f"{val_metrics.policy_accuracy:.8f}" if val_metrics is not None else ""
                ),
                "val_value_mae": (
                    f"{val_metrics.value_mae:.8f}" if val_metrics is not None else ""
                ),
                "best_val_loss": f"{best_val_loss:.8f}" if best_val_loss is not None else "",
                "run_elapsed_seconds": f"{elapsed_seconds:.6f}",
                "run_elapsed_time": format_seconds(elapsed_seconds),
                "cumulative_train_seconds": f"{cumulative_seconds:.6f}",
                "cumulative_train_time": format_seconds(cumulative_seconds),
            }
        )


def csv_log_fieldnames() -> list[str]:
    return [
        "written_at_unix",
        "run_started_at_unix",
        "epoch",
        "train_samples",
        "train_loss",
        "train_policy_loss",
        "train_value_loss",
        "train_policy_accuracy",
        "train_value_mae",
        "val_samples",
        "val_loss",
        "val_policy_loss",
        "val_value_loss",
        "val_policy_accuracy",
        "val_value_mae",
        "best_val_loss",
        "run_elapsed_seconds",
        "run_elapsed_time",
        "cumulative_train_seconds",
        "cumulative_train_time",
    ]


def read_time_log(checkpoint_dir: Path) -> dict[str, object]:
    path = checkpoint_dir / TIME_LOG_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_key(config: SupervisedConfig, model_config: object, data_paths: Sequence[str]) -> str:
    config_dict = asdict(config)
    # epochs controls how long this run trains; it should not split cumulative time.
    config_dict.pop("epochs", None)
    # Logging cadence does not change the actual training setup.
    config_dict.pop("log_interval", None)
    config_dict.pop("csv_log_interval_epochs", None)
    payload = {
        "supervised_config": config_dict,
        "model_config": asdict(model_config),
        "data_paths": sorted(str(Path(path)) for path in data_paths),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def format_seconds(seconds: float) -> str:
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
