from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .data import fixed_validation_starts, sample_batch
from .manifests import (
    artifact_hashes,
    git_commit,
    package_versions,
    sha256_file,
    write_manifest,
)
from .model import DecoderOnlyLM, cross_entropy_loss
from .prompts import character_control_prefix
from .research_data import read_poems
from .research_types import RunManifest
from .templates import load_templates
from .training import evaluate_fixed, select_device, set_global_seed


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[character] for character in text]


def _scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _save_research_checkpoint(
    path: Path,
    model: DecoderOnlyLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    stage: str,
    step: int,
    vocab: list[str],
    train_rng: random.Random,
    config_sha256: str,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "step": step,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "vocab": vocab,
        "train_rng_state": train_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "config_sha256": config_sha256,
        "best_validation_loss": best_validation_loss,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng(checkpoint: dict[str, Any], train_rng: random.Random) -> None:
    train_rng.setstate(checkpoint["train_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def _stage_data(
    stage: str,
    all_training: Sequence[str],
    selected_training: Sequence[str],
    selected_validation: Sequence[str],
    stoi: dict[str, int],
) -> tuple[list[int], list[int]]:
    training_text = (
        "\n\n".join(all_training) if stage == "pretrain" else "\n\n".join(selected_training)
    )
    validation_text = "\n\n".join(selected_validation)
    return _encode(training_text, stoi), _encode(validation_text, stoi)


def train_character_research(
    config_path: Path,
    project_root: Path,
    seed: int,
    device_name: str = "auto",
    resume: Path | None = None,
    pretrain_tokens_override: int | None = None,
    finetune_tokens_override: int | None = None,
) -> Path:
    config = _load_json(config_path)
    if seed not in config["training_seeds"]:
        raise ValueError(f"seed {seed} is not pre-registered")
    dataset_dir = project_root / config["dataset_dir"]
    all_training_records = read_poems(dataset_dir / "splits/train.jsonl")
    selected_training_records = read_poems(dataset_dir / "selected/train.jsonl")
    selected_validation_records = read_poems(dataset_dir / "selected/validation.jsonl")
    templates = load_templates(dataset_dir / "templates.json")

    all_training = [record.normalized_text for record in all_training_records]
    selected_training = [
        character_control_prefix(templates[record.template_id]) + "\n" + record.normalized_text
        for record in selected_training_records
        if record.template_id
    ]
    selected_validation = [
        character_control_prefix(templates[record.template_id]) + "\n" + record.normalized_text
        for record in selected_validation_records
        if record.template_id
    ]
    if not all_training or not selected_training or not selected_validation:
        raise ValueError("research dataset does not contain all required splits")

    conditioning_text = "".join(str(value) for value in config.get("conditioning_themes", ()))
    complete_text = "\n\n".join(
        [*all_training, *selected_training, *selected_validation, conditioning_text]
    )
    vocab = sorted(set(complete_text))
    stoi = {character: index for index, character in enumerate(vocab)}
    model_config = ModelConfig(**config["model"])
    device = select_device(device_name)
    set_global_seed(seed)
    model = DecoderOnlyLM(len(vocab), model_config).to(device)
    run_dir = project_root / config["output_dir"] / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_digest = sha256_file(config_path)
    resume_payload: dict[str, Any] | None = None
    if resume is not None:
        resume_payload = torch.load(resume, map_location=device, weights_only=False)
        if resume_payload["config_sha256"] != config_digest:
            raise ValueError("resume checkpoint was created with a different config")
        if resume_payload["vocab"] != vocab:
            raise ValueError("resume checkpoint vocabulary does not match the dataset")
        model.load_state_dict(resume_payload["model_state_dict"])

    pretrain_tokens = pretrain_tokens_override or int(config["pretrain_tokens"])
    finetune_tokens = finetune_tokens_override or int(config["finetune_tokens"])
    stage_specs = [
        ("pretrain", pretrain_tokens, float(config["learning_rate"])),
        ("finetune", finetune_tokens, float(config["finetune_learning_rate"])),
    ]
    resume_stage = resume_payload["stage"] if resume_payload else None
    metrics_path = run_dir / "metrics.jsonl"
    if resume is None and metrics_path.exists():
        raise FileExistsError(f"existing run at {run_dir}; pass --resume or choose another seed")

    for stage, token_budget, learning_rate in stage_specs:
        if resume_stage == "finetune" and stage == "pretrain":
            continue
        train_ids, validation_ids = _stage_data(
            stage, all_training, selected_training, selected_validation, stoi
        )
        total_steps = max(
            1,
            math.ceil(
                token_budget / (int(config["batch_size"]) * model_config.context_length)
            ),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=float(config["weight_decay"])
        )
        warmup = min(int(config["warmup_steps"]), max(total_steps // 10, 1))
        scheduler = _scheduler(optimizer, warmup, total_steps)
        train_rng = random.Random(seed + (0 if stage == "pretrain" else 100_000))
        start_step = 0
        best_validation_loss = float("inf")
        if resume_payload is not None and resume_payload["stage"] == stage:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            start_step = int(resume_payload["step"])
            best_validation_loss = float(resume_payload["best_validation_loss"])
            _restore_rng(resume_payload, train_rng)
            resume_payload = None
        elif stage == "finetune" and (checkpoint_dir / "pretrain_best.pt").exists():
            best_pretrain = torch.load(
                checkpoint_dir / "pretrain_best.pt", map_location=device, weights_only=False
            )
            model.load_state_dict(best_pretrain["model_state_dict"])

        validation_starts = fixed_validation_starts(
            validation_ids,
            batch_size=min(int(config["batch_size"]), 32),
            context_length=model_config.context_length,
            num_batches=20,
            seed=1729,
        )
        for step in range(start_step + 1, total_steps + 1):
            model.train()
            features, targets = sample_batch(
                train_ids,
                int(config["batch_size"]),
                model_config.context_length,
                device,
                train_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy_loss(model(features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            validation_loss: float | None = None
            if step % int(config["eval_every"]) == 0 or step == total_steps:
                validation_loss = evaluate_fixed(model, validation_ids, validation_starts, device)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    _save_research_checkpoint(
                        checkpoint_dir / f"{stage}_best.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage,
                        step,
                        vocab,
                        train_rng,
                        config_digest,
                        best_validation_loss,
                    )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "stage": stage,
                            "step": step,
                            "train_loss": float(loss.item()),
                            "validation_loss": validation_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if step % int(config["save_every"]) == 0 or step == total_steps:
                _save_research_checkpoint(
                    checkpoint_dir / f"{stage}_step_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    stage,
                    step,
                    vocab,
                    train_rng,
                    config_digest,
                    best_validation_loss,
                )

    artifacts = [metrics_path, *checkpoint_dir.glob("*_best.pt")]
    manifest = RunManifest(
        command="poetry-lm train char",
        git_commit=git_commit(project_root),
        resolved_config=copy.deepcopy(config),
        dataset_sha256=sha256_file(dataset_dir / "dataset_manifest.json"),
        model_id="poetry_lm.DecoderOnlyLM",
        model_revision=None,
        seed=seed,
        device=str(device),
        versions=package_versions(["torch", "numpy"]),
        artifacts=artifact_hashes(artifacts, project_root),
    )
    write_manifest(run_dir / "run_manifest.json", manifest)
    return run_dir
