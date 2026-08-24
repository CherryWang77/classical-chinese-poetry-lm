#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from poetry_lm.config import ExperimentConfig  # noqa: E402
from poetry_lm.model import MultiHeadCausalSelfAttention  # noqa: E402
from poetry_lm.training import select_device, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark manual causal attention against PyTorch SDPA"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def one_step(module: torch.nn.Module, inputs: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    output = module(inputs)
    output.square().mean().backward()


def benchmark_backend(
    backend: str,
    base_state: dict[str, torch.Tensor],
    experiment: ExperimentConfig,
    batch_size: int,
    warmup: int,
    steps: int,
    device: torch.device,
) -> dict[str, object]:
    model_config = replace(experiment.model, attention_backend=backend)
    module = MultiHeadCausalSelfAttention(model_config).to(device)
    module.load_state_dict(base_state)
    module.train()
    inputs = torch.randn(
        batch_size,
        model_config.context_length,
        model_config.embed_dim,
        device=device,
        requires_grad=True,
    )

    for _ in range(warmup):
        one_step(module, inputs)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        one_step(module, inputs)
        synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - start))

    peak_memory: float | str = ""
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "backend": backend,
        "device": str(device),
        "batch_size": batch_size,
        "context_length": model_config.context_length,
        "embed_dim": model_config.embed_dim,
        "num_heads": model_config.num_heads,
        "mean_step_ms": statistics.fmean(timings),
        "median_step_ms": statistics.median(timings),
        "peak_memory_mb": peak_memory,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch-size and steps must be positive; warmup must be non-negative")
    experiment = ExperimentConfig.from_json(args.config)
    device = select_device(args.device)
    set_global_seed(experiment.training.seed)

    manual_config = replace(experiment.model, attention_backend="manual")
    reference = MultiHeadCausalSelfAttention(manual_config).to(device)
    base_state = reference.state_dict()
    rows = [
        benchmark_backend(
            backend,
            base_state,
            experiment,
            args.batch_size,
            args.warmup,
            args.steps,
            device,
        )
        for backend in ("manual", "sdpa")
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manual_ms = float(rows[0]["median_step_ms"])
    sdpa_ms = float(rows[1]["median_step_ms"])
    print(f"device: {device}")
    print(f"manual median: {manual_ms:.3f} ms")
    print(f"sdpa median: {sdpa_ms:.3f} ms")
    print(f"speedup (manual / sdpa): {manual_ms / sdpa_ms:.3f}x")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
