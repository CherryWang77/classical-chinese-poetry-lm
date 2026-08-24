from __future__ import annotations

import csv
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from .data import batch_from_starts
from .model import DecoderOnlyLM, cross_entropy_loss


METRIC_FIELDS = [
    "step",
    "train_loss",
    "train_bpc",
    "avg_val_loss",
    "avg_val_bpc",
    "step_time_ms",
    "peak_memory_mb",
]


def loss_to_bpc(loss: float) -> float:
    return loss / math.log(2.0)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_fixed(
    model: DecoderOnlyLM,
    validation_ids: Sequence[int],
    batches_of_starts: Sequence[Sequence[int]],
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for starts in batches_of_starts:
            features, targets = batch_from_starts(
                validation_ids, starts, model.config.context_length, device
            )
            losses.append(float(cross_entropy_loss(model(features), targets).item()))
    return sum(losses) / len(losses)


class CsvMetricLogger:
    def __init__(self, path: Path, append: bool):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        self.handle = path.open("a" if append else "w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=METRIC_FIELDS)
        if not append or not file_exists:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "CsvMetricLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def save_checkpoint(
    path: Path,
    model: DecoderOnlyLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    experiment_config: dict[str, Any],
    vocab: list[str],
    train_rng: random.Random,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "experiment_config": experiment_config,
        "model_config": model.config.to_dict(),
        "vocab": vocab,
        "train_rng_state": train_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(checkpoint: dict[str, Any], train_rng: random.Random) -> None:
    train_rng.setstate(checkpoint["train_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
