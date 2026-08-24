#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from poetry_lm.config import ExperimentConfig  # noqa: E402
from poetry_lm.data import fixed_validation_starts, load_corpus, sample_batch  # noqa: E402
from poetry_lm.generation import generate_ids, make_generator  # noqa: E402
from poetry_lm.model import DecoderOnlyLM, cross_entropy_loss  # noqa: E402
from poetry_lm.training import (  # noqa: E402
    CsvMetricLogger,
    load_checkpoint,
    loss_to_bpc,
    restore_rng_state,
    save_checkpoint,
    select_device,
    set_global_seed,
    evaluate_fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume a poetry language model")
    parser.add_argument("--config", type=Path, required=True, help="experiment JSON config")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="base directory for relative data and output paths",
    )
    parser.add_argument("--resume", type=Path, help="checkpoint to resume")
    parser.add_argument("--max-steps", type=int, help="override target training step")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing metrics file when starting from step zero",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = ExperimentConfig.from_json(config_path)
    target_step = args.max_steps or config.training.max_steps
    if target_step <= 0:
        raise ValueError("target step must be positive")

    device = select_device(args.device)
    set_global_seed(config.training.seed)
    train_rng = random.Random(config.training.seed)

    corpus_path = Path(config.data.input_file)
    if not corpus_path.is_absolute():
        corpus_path = project_root / corpus_path
    corpus = load_corpus(corpus_path, config.data.validation_ratio)

    model = DecoderOnlyLM(len(corpus.vocab), config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    start_step = 0

    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else project_root / args.resume
        checkpoint = load_checkpoint(resume_path, device)
        if checkpoint["model_config"] != config.model.to_dict():
            raise ValueError("checkpoint model configuration does not match the selected config")
        if checkpoint["vocab"] != corpus.vocab:
            raise ValueError("checkpoint vocabulary does not match the selected corpus")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        restore_rng_state(checkpoint, train_rng)

    if start_step >= target_step:
        raise ValueError(f"checkpoint step {start_step} is already >= target {target_step}")

    run_dir = project_root / "artifacts" / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if start_step == 0 and metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"metrics already exist at {metrics_path}; use --overwrite or --resume"
        )

    validation_starts = fixed_validation_starts(
        corpus.val_ids,
        config.training.eval_batch_size,
        config.model.context_length,
        config.training.num_val_batches,
        config.training.eval_seed,
    )
    prompt_ids = corpus.encode(config.generation.prompt)
    experiment_payload = config.to_dict()
    (run_dir / "resolved_config.json").write_text(
        json.dumps(experiment_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"run: {config.run_name}")
    print(f"device: {device}")
    print(f"corpus: {corpus_path}")
    print(f"characters: {len(corpus.text):,}; vocabulary: {len(corpus.vocab):,}")
    print(f"training steps: {start_step + 1:,}..{target_step:,}")
    print(f"attention backend: {config.model.attention_backend}")
    print(f"outputs: {run_dir}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    interval_start = time.perf_counter()
    last_timed_step = start_step

    with CsvMetricLogger(metrics_path, append=start_step > 0) as metric_logger:
        for step in range(start_step + 1, target_step + 1):
            model.train()
            features, targets = sample_batch(
                corpus.train_ids,
                config.training.batch_size,
                config.model.context_length,
                device,
                train_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            train_loss = cross_entropy_loss(model(features), targets)
            train_loss.backward()
            optimizer.step()

            train_loss_value = float(train_loss.item())
            validation_loss: float | str = ""
            validation_bpc: float | str = ""
            step_time_ms: float | str = ""
            peak_memory_mb: float | str = ""

            if step % config.training.eval_every == 0:
                validation_loss = evaluate_fixed(
                    model, corpus.val_ids, validation_starts, device
                )
                validation_bpc = loss_to_bpc(validation_loss)

            if step % config.training.log_every == 0:
                synchronize(device)
                elapsed = time.perf_counter() - interval_start
                step_time_ms = 1000.0 * elapsed / (step - last_timed_step)
                if device.type == "cuda":
                    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                    torch.cuda.reset_peak_memory_stats(device)
                interval_start = time.perf_counter()
                last_timed_step = step

            metric_logger.write(
                {
                    "step": step,
                    "train_loss": train_loss_value,
                    "train_bpc": loss_to_bpc(train_loss_value),
                    "avg_val_loss": validation_loss,
                    "avg_val_bpc": validation_bpc,
                    "step_time_ms": step_time_ms,
                    "peak_memory_mb": peak_memory_mb,
                }
            )

            if step % config.training.log_every == 0:
                message = f"step {step:05d} | train {train_loss_value:.6f}"
                if validation_loss != "":
                    message += f" | val {validation_loss:.6f}"
                if step_time_ms != "":
                    message += f" | {step_time_ms:.2f} ms/step"
                print(message)
                metric_logger.flush()

            if step % config.training.save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    step,
                    experiment_payload,
                    corpus.vocab,
                    train_rng,
                )

            if step % config.training.sample_every == 0:
                sample_generator = make_generator(device, config.generation.seed)
                generated = generate_ids(
                    model,
                    prompt_ids,
                    config.generation.max_new_chars,
                    config.generation.temperature,
                    config.generation.top_k,
                    device,
                    sample_generator,
                )
                (sample_dir / f"step_{step}.txt").write_text(
                    corpus.decode(generated), encoding="utf-8"
                )

        save_checkpoint(
            checkpoint_dir / "final.pt",
            model,
            optimizer,
            target_step,
            experiment_payload,
            corpus.vocab,
            train_rng,
        )
        metric_logger.flush()

    print(f"completed: {config.run_name} at step {target_step}")


if __name__ == "__main__":
    main()
