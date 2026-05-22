from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from agents import load_model_config_from_checkpoint
from dataset import DraughtsJsonlDataset, collate_batch, to_torch_batch
from model import CandidateMoveNet
from train_supervised import (
    DEFAULT_CHECKPOINT_DIR,
    TIME_LOG_FILENAME,
    format_seconds,
    load_scalar_yaml,
    move_batch_to_device,
    read_time_log,
    set_seed,
    should_write_csv_log,
    split_indices,
)


DEFAULT_RL_TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "rl_train.yaml"


@dataclass
class RlTrainConfig:
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.0001
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.0
    advantage_clip: float = 2.0
    ppo_clip_epsilon: float = 0.2
    validation_split: float = 0.1
    seed: int = 1
    shuffle: int = 1
    log_interval: int = 20
    csv_log_interval_epochs: int = 1
    num_workers: int = 0


@dataclass
class RlLossParts:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    advantage: torch.Tensor
    selected_prob: torch.Tensor
    ratio: torch.Tensor
    clipped: torch.Tensor


@dataclass
class RlEpochMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    advantage_mean: float
    advantage_abs_mean: float
    selected_prob_mean: float
    ratio_mean: float
    clip_fraction: float
    value_mae: float
    samples: int


def main() -> None:
    print("rl_train: parsing arguments and loading configs...", flush=True)
    args = parse_args()
    print(f"rl_train: loading train config from {args.config}", flush=True)
    config = load_rl_train_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"rl_train: using device={device}", flush=True)
    set_seed(config.seed)

    print(f"rl_train: loading dataset from {', '.join(args.data)}", flush=True)
    dataset = DraughtsJsonlDataset(args.data)
    validate_rl_dataset(dataset)
    print(f"rl_train: loaded {len(dataset)} RL samples", flush=True)
    train_indices, val_indices = split_indices(len(dataset), config.validation_split, config.seed)
    print(f"rl_train: split dataset train={len(train_indices)} val={len(val_indices)}", flush=True)
    print("rl_train: building dataloaders...", flush=True)
    train_loader = make_loader(dataset, train_indices, config, training=True)
    val_loader = make_loader(dataset, val_indices, config, training=False) if val_indices else None

    print(f"rl_train: loading base checkpoint from {args.resume}", flush=True)
    resume_checkpoint = torch.load(args.resume, map_location=device)
    model_config = load_model_config_from_checkpoint(resume_checkpoint)
    model = CandidateMoveNet(config=model_config).to(device)
    model.load_state_dict(resume_checkpoint.get("model_state_dict", resume_checkpoint))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch, best_val_loss, previous_cumulative_seconds = resume_rl_state(
        resume_checkpoint,
        optimizer,
        checkpoint_dir,
        config,
        model_config,
        args.data,
    )

    print(
        "rl_train:"
        f" samples={len(dataset)} train={len(train_indices)} val={len(val_indices)}"
        f" epochs={config.epochs} batch_size={config.batch_size}"
        f" resume={args.resume} device={device} checkpoint_dir={checkpoint_dir}"
    )
    print("rl_train: starting training loop...", flush=True)

    run_start = time.perf_counter()
    run_started_at_unix = time.time()
    key = experiment_key(config, model_config, args.data)
    csv_log_path = checkpoint_dir / f"rl_training_log_{key}.csv"
    latest_checkpoint: Optional[Path] = None
    final_epoch = start_epoch + config.epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device, epoch)
        val_metrics = evaluate(model, val_loader, config, device) if val_loader is not None else None
        elapsed_seconds = time.perf_counter() - run_start
        cumulative_seconds = previous_cumulative_seconds + elapsed_seconds
        compare_loss = val_metrics.loss if val_metrics is not None else train_metrics.loss
        is_best = best_val_loss is None or compare_loss < best_val_loss
        if is_best:
            best_val_loss = compare_loss

        print_epoch_summary(epoch, train_metrics, val_metrics, elapsed_seconds, cumulative_seconds)
        latest_checkpoint = save_checkpoint(
            checkpoint_dir / "candidate_move_net_rl_latest.pt",
            model,
            optimizer,
            epoch,
            config,
            model_config,
            args.data,
            args.resume,
            train_metrics,
            val_metrics,
            elapsed_seconds,
            cumulative_seconds,
            best_val_loss,
        )
        if args.save_every_epoch or epoch == final_epoch:
            save_checkpoint(
                checkpoint_dir / f"candidate_move_net_rl_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                model_config,
                args.data,
                args.resume,
                train_metrics,
                val_metrics,
                elapsed_seconds,
                cumulative_seconds,
                best_val_loss,
            )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "candidate_move_net_rl_best.pt",
                model,
                optimizer,
                epoch,
                config,
                model_config,
                args.data,
                args.resume,
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
            args.data,
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
        print(f"saved latest RL checkpoint: {latest_checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CandidateMoveNet from RL self-play JSONL data.")
    parser.add_argument("data", nargs="+", help="RL self-play JSONL file(s) or directory path(s).")
    parser.add_argument("--config", default=str(DEFAULT_RL_TRAIN_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume", required=True, help="Base or RL candidate checkpoint to update.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Defaults to auto.")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save candidate_move_net_rl_epoch_XXX.pt after every epoch.",
    )
    return parser.parse_args()


def load_rl_train_config(config_path: str | Path) -> RlTrainConfig:
    path = Path(config_path)
    defaults = asdict(RlTrainConfig())
    values = dict(defaults)
    values.update(load_scalar_yaml(path, defaults, "rl_train"))
    config = RlTrainConfig(**values)
    validate_rl_train_config(config)
    return config


def validate_rl_train_config(config: RlTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("rl_train epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("rl_train batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("rl_train learning_rate must be positive")
    if config.policy_loss_weight < 0:
        raise ValueError("rl_train policy_loss_weight must be non-negative")
    if config.value_loss_weight < 0:
        raise ValueError("rl_train value_loss_weight must be non-negative")
    if config.entropy_weight < 0:
        raise ValueError("rl_train entropy_weight must be non-negative")
    if config.advantage_clip < 0:
        raise ValueError("rl_train advantage_clip must be non-negative")
    if config.ppo_clip_epsilon <= 0:
        raise ValueError("rl_train ppo_clip_epsilon must be positive")
    if not 0 <= config.validation_split < 1:
        raise ValueError("rl_train validation_split must be in [0, 1)")
    if config.shuffle not in {0, 1}:
        raise ValueError("rl_train shuffle must be 0 or 1")
    if config.log_interval <= 0:
        raise ValueError("rl_train log_interval must be positive")
    if config.csv_log_interval_epochs <= 0:
        raise ValueError("rl_train csv_log_interval_epochs must be positive")
    if config.num_workers < 0:
        raise ValueError("rl_train num_workers must be non-negative")
    if config.policy_loss_weight == 0 and config.value_loss_weight == 0:
        raise ValueError("rl_train policy and value loss weights cannot both be zero")


def validate_rl_dataset(dataset: DraughtsJsonlDataset) -> None:
    for index, sample in enumerate(dataset.samples):
        if sample.get("sample_kind") != "rl_selfplay":
            raise ValueError(f"sample {index}: train_selfplay requires sample_kind=rl_selfplay")
        if not isinstance(sample.get("policy_probs"), list):
            raise ValueError(f"sample {index}: rl_selfplay sample must include policy_probs")
        if not isinstance(sample.get("selected_policy_prob"), (int, float)):
            raise ValueError(f"sample {index}: rl_selfplay sample must include selected_policy_prob")
        if not 0 < float(sample["selected_policy_prob"]) <= 1:
            raise ValueError(f"sample {index}: selected_policy_prob must be in (0, 1]")
        if not isinstance(sample.get("temperature"), (int, float)):
            raise ValueError(f"sample {index}: rl_selfplay sample must include temperature")
        if float(sample["temperature"]) <= 0:
            raise ValueError(f"sample {index}: temperature must be positive")
        if "predicted_value" not in sample:
            raise ValueError(f"sample {index}: rl_selfplay sample must include predicted_value")


def make_loader(
    dataset: DraughtsJsonlDataset,
    indices: Sequence[int],
    config: RlTrainConfig,
    training: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=config.batch_size,
        shuffle=bool(config.shuffle) if training else False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        generator=generator,
    )


def train_one_epoch(
    model: CandidateMoveNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: RlTrainConfig,
    device: torch.device,
    epoch: int,
) -> RlEpochMetrics:
    model.train()
    totals = RlMetricTotals()
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch_to_device(to_torch_batch(raw_batch), device)
        output = model(batch["observation"], batch["move_features"], batch["move_mask"])
        parts = compute_rl_loss(output, batch, config)

        optimizer.zero_grad()
        parts.loss.backward()
        optimizer.step()

        totals.update(output, batch, parts)
        if batch_index % config.log_interval == 0:
            metrics = totals.to_metrics()
            print(
                f"epoch={epoch} batch={batch_index} loss={metrics.loss:.4f}"
                f" policy_loss={metrics.policy_loss:.4f}"
                f" value_loss={metrics.value_loss:.4f}"
                f" entropy={metrics.entropy:.4f}"
                f" advantage_abs={metrics.advantage_abs_mean:.4f}"
                f" ratio={metrics.ratio_mean:.4f}"
                f" clip_fraction={metrics.clip_fraction:.3f}"
            )
    return totals.to_metrics()


def evaluate(
    model: CandidateMoveNet,
    loader: DataLoader,
    config: RlTrainConfig,
    device: torch.device,
) -> RlEpochMetrics:
    model.eval()
    totals = RlMetricTotals()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(to_torch_batch(raw_batch), device)
            output = model(batch["observation"], batch["move_features"], batch["move_mask"])
            totals.update(output, batch, compute_rl_loss(output, batch, config))
    return totals.to_metrics()


def compute_rl_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    config: RlTrainConfig,
) -> RlLossParts:
    temperatures = batch["behavior_temperature"].clamp_min(1.0e-6).unsqueeze(1)
    log_probs = nn.functional.log_softmax(output["policy_logits"] / temperatures, dim=1)
    probs = log_probs.exp()
    target_indices = batch["policy_target_index"]
    selected_log_prob = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
    selected_prob = probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)

    advantage = batch["value_target"] - batch["behavior_value"]
    if config.advantage_clip > 0:
        advantage = advantage.clamp(-config.advantage_clip, config.advantage_clip)
    old_prob = batch["selected_policy_prob"].clamp_min(1.0e-8)
    ratio = selected_prob / old_prob
    clipped_ratio = ratio.clamp(1.0 - config.ppo_clip_epsilon, 1.0 + config.ppo_clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
    value_loss = nn.functional.mse_loss(output["value"], batch["value_target"])
    entropy = -(probs * log_probs).sum(dim=1).mean()
    loss = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        - config.entropy_weight * entropy
    )
    clipped = (ratio - clipped_ratio).abs() > 1.0e-8
    return RlLossParts(loss, policy_loss, value_loss, entropy, advantage, selected_prob, ratio, clipped)


class RlMetricTotals:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.advantage_sum = 0.0
        self.advantage_abs_sum = 0.0
        self.selected_prob_sum = 0.0
        self.ratio_sum = 0.0
        self.clip_count = 0
        self.value_abs_error_sum = 0.0
        self.samples = 0

    def update(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, object],
        parts: RlLossParts,
    ) -> None:
        batch_size = int(batch["value_target"].shape[0])
        self.loss_sum += float(parts.loss.item()) * batch_size
        self.policy_loss_sum += float(parts.policy_loss.item()) * batch_size
        self.value_loss_sum += float(parts.value_loss.item()) * batch_size
        self.entropy_sum += float(parts.entropy.item()) * batch_size
        self.advantage_sum += float(parts.advantage.sum().item())
        self.advantage_abs_sum += float(parts.advantage.abs().sum().item())
        self.selected_prob_sum += float(parts.selected_prob.sum().item())
        self.ratio_sum += float(parts.ratio.sum().item())
        self.clip_count += int(parts.clipped.sum().item())
        self.value_abs_error_sum += float((output["value"] - batch["value_target"]).abs().sum().item())
        self.samples += batch_size

    def to_metrics(self) -> RlEpochMetrics:
        if self.samples <= 0:
            return RlEpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return RlEpochMetrics(
            loss=self.loss_sum / self.samples,
            policy_loss=self.policy_loss_sum / self.samples,
            value_loss=self.value_loss_sum / self.samples,
            entropy=self.entropy_sum / self.samples,
            advantage_mean=self.advantage_sum / self.samples,
            advantage_abs_mean=self.advantage_abs_sum / self.samples,
            selected_prob_mean=self.selected_prob_sum / self.samples,
            ratio_mean=self.ratio_sum / self.samples,
            clip_fraction=self.clip_count / self.samples,
            value_mae=self.value_abs_error_sum / self.samples,
            samples=self.samples,
        )


def print_epoch_summary(
    epoch: int,
    train_metrics: RlEpochMetrics,
    val_metrics: Optional[RlEpochMetrics],
    elapsed_seconds: float,
    cumulative_seconds: float,
) -> None:
    message = (
        f"epoch={epoch} train_loss={train_metrics.loss:.4f}"
        f" train_policy_loss={train_metrics.policy_loss:.4f}"
        f" train_value_mae={train_metrics.value_mae:.4f}"
        f" train_entropy={train_metrics.entropy:.4f}"
        f" train_advantage_abs={train_metrics.advantage_abs_mean:.4f}"
        f" train_ratio={train_metrics.ratio_mean:.4f}"
        f" train_clip_fraction={train_metrics.clip_fraction:.3f}"
    )
    if val_metrics is not None:
        message += (
            f" val_loss={val_metrics.loss:.4f}"
            f" val_value_mae={val_metrics.value_mae:.4f}"
            f" val_entropy={val_metrics.entropy:.4f}"
            f" val_advantage_abs={val_metrics.advantage_abs_mean:.4f}"
            f" val_ratio={val_metrics.ratio_mean:.4f}"
            f" val_clip_fraction={val_metrics.clip_fraction:.3f}"
        )
    message += (
        f" elapsed={format_seconds(elapsed_seconds)}"
        f" cumulative_train_time={format_seconds(cumulative_seconds)}"
    )
    print(message)


def resume_rl_state(
    checkpoint: object,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    config: RlTrainConfig,
    model_config: object,
    data_paths: Sequence[str],
) -> tuple[int, Optional[float], float]:
    previous_cumulative_seconds = load_cumulative_seconds(
        checkpoint_dir,
        config,
        model_config,
        data_paths,
    )
    if not isinstance(checkpoint, dict) or "rl_train_config" not in checkpoint:
        return 0, None, previous_cumulative_seconds

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return (
        int(checkpoint.get("epoch", 0)),
        checkpoint.get("best_val_loss"),
        float(checkpoint.get("cumulative_train_seconds", previous_cumulative_seconds)),
    )


def save_checkpoint(
    path: Path,
    model: CandidateMoveNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: RlTrainConfig,
    model_config: object,
    data_paths: Sequence[str],
    base_checkpoint: str,
    train_metrics: RlEpochMetrics,
    val_metrics: Optional[RlEpochMetrics],
    elapsed_seconds: float,
    cumulative_seconds: float,
    best_val_loss: Optional[float],
) -> Path:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rl_train_config": asdict(config),
            "model_config": asdict(model_config),
            "data_paths": list(data_paths),
            "base_checkpoint": base_checkpoint,
            "train_metrics": asdict(train_metrics),
            "val_metrics": asdict(val_metrics) if val_metrics is not None else None,
            "run_elapsed_seconds": elapsed_seconds,
            "cumulative_train_seconds": cumulative_seconds,
            "cumulative_train_time": format_seconds(cumulative_seconds),
            "best_val_loss": best_val_loss,
            "experiment_key": experiment_key(config, model_config, data_paths),
        },
        path,
    )
    return path


def load_cumulative_seconds(
    checkpoint_dir: Path,
    config: RlTrainConfig,
    model_config: object,
    data_paths: Sequence[str],
) -> float:
    key = experiment_key(config, model_config, data_paths)
    return float(read_time_log(checkpoint_dir).get(key, {}).get("cumulative_train_seconds", 0.0))


def update_time_log(
    checkpoint_dir: Path,
    config: RlTrainConfig,
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
        "rl_train_config": asdict(config),
        "model_config": asdict(model_config),
        "data_paths": list(data_paths),
        "updated_at_unix": time.time(),
    }
    (checkpoint_dir / TIME_LOG_FILENAME).write_text(
        json.dumps(time_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_csv_log(
    path: Path,
    epoch: int,
    run_started_at_unix: float,
    train_metrics: RlEpochMetrics,
    val_metrics: Optional[RlEpochMetrics],
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
                **csv_metric_values("train", train_metrics),
                **csv_metric_values("val", val_metrics),
                "best_val_loss": f"{best_val_loss:.8f}" if best_val_loss is not None else "",
                "run_elapsed_seconds": f"{elapsed_seconds:.6f}",
                "run_elapsed_time": format_seconds(elapsed_seconds),
                "cumulative_train_seconds": f"{cumulative_seconds:.6f}",
                "cumulative_train_time": format_seconds(cumulative_seconds),
            }
        )


def csv_metric_values(prefix: str, metrics: Optional[RlEpochMetrics]) -> dict[str, object]:
    if metrics is None:
        return {field: "" for field in csv_metric_fieldnames(prefix)}
    return {
        f"{prefix}_samples": metrics.samples,
        f"{prefix}_loss": f"{metrics.loss:.8f}",
        f"{prefix}_policy_loss": f"{metrics.policy_loss:.8f}",
        f"{prefix}_value_loss": f"{metrics.value_loss:.8f}",
        f"{prefix}_entropy": f"{metrics.entropy:.8f}",
        f"{prefix}_advantage_mean": f"{metrics.advantage_mean:.8f}",
        f"{prefix}_advantage_abs_mean": f"{metrics.advantage_abs_mean:.8f}",
        f"{prefix}_selected_prob_mean": f"{metrics.selected_prob_mean:.8f}",
        f"{prefix}_ratio_mean": f"{metrics.ratio_mean:.8f}",
        f"{prefix}_clip_fraction": f"{metrics.clip_fraction:.8f}",
        f"{prefix}_value_mae": f"{metrics.value_mae:.8f}",
    }


def csv_log_fieldnames() -> list[str]:
    return [
        "written_at_unix",
        "run_started_at_unix",
        "epoch",
        *csv_metric_fieldnames("train"),
        *csv_metric_fieldnames("val"),
        "best_val_loss",
        "run_elapsed_seconds",
        "run_elapsed_time",
        "cumulative_train_seconds",
        "cumulative_train_time",
    ]


def csv_metric_fieldnames(prefix: str) -> list[str]:
    return [
        f"{prefix}_samples",
        f"{prefix}_loss",
        f"{prefix}_policy_loss",
        f"{prefix}_value_loss",
        f"{prefix}_entropy",
        f"{prefix}_advantage_mean",
        f"{prefix}_advantage_abs_mean",
        f"{prefix}_selected_prob_mean",
        f"{prefix}_ratio_mean",
        f"{prefix}_clip_fraction",
        f"{prefix}_value_mae",
    ]


def experiment_key(config: RlTrainConfig, model_config: object, data_paths: Sequence[str]) -> str:
    config_dict = asdict(config)
    config_dict.pop("epochs", None)
    config_dict.pop("log_interval", None)
    config_dict.pop("csv_log_interval_epochs", None)
    payload = {
        "rl_train_config": config_dict,
        "model_config": asdict(model_config),
        "data_paths": sorted(str(Path(path)) for path in data_paths),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
