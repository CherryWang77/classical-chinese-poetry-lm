from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .manifests import (
    artifact_hashes,
    git_commit,
    package_versions,
    sha256_file,
    write_manifest,
)
from .prompts import SYSTEM_PROMPT, user_prompt
from .research_data import read_poems
from .research_types import RunManifest
from .templates import load_templates


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in rows)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for row in rows:
            padding = maximum - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


class TokenizedDataset(torch.utils.data.Dataset[dict[str, list[int]]]):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _tokenize_records(tokenizer: Any, records: list[Any], templates: dict[str, Any], max_length: int) -> TokenizedDataset:
    rows: list[dict[str, list[int]]] = []
    for record in records:
        template = templates[record.template_id]
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(template)},
        ]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": record.normalized_text},
        ]
        kwargs = {"enable_thinking": False}
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            **kwargs,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template prompt is not a prefix of the supervised example")
        full_ids = full_ids[:max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if all(value == -100 for value in labels):
            continue
        rows.append({"input_ids": full_ids, "labels": labels})
    if not rows:
        raise ValueError("no QLoRA training examples survived tokenization")
    return TokenizedDataset(rows)


def _training_arguments(TrainingArguments: Any, payload: dict[str, Any]) -> Any:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        payload["eval_strategy"] = "steps"
    else:
        payload["evaluation_strategy"] = "steps"
    return TrainingArguments(**payload)


def train_lora(
    config_path: Path,
    project_root: Path,
    seed: int,
    resume: str | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 QLoRA requires a CUDA GPU; use the Kaggle launcher")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("QLoRA dependencies require `pip install -e '.[research]'`") from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if seed not in config["training_seeds"]:
        raise ValueError(f"seed {seed} is not pre-registered")
    set_seed(seed)
    dataset_dir = project_root / config["dataset_dir"]
    templates = load_templates(dataset_dir / "templates.json")
    train_records = read_poems(dataset_dir / "selected/train.jsonl")
    validation_records = read_poems(dataset_dir / "selected/validation.jsonl")
    if config.get("train_limit") is not None:
        train_records = sorted(train_records, key=lambda record: record.id)[
            : int(config["train_limit"])
        ]
    if config.get("validation_limit") is not None:
        validation_records = sorted(validation_records, key=lambda record: record.id)[
            : int(config["validation_limit"])
        ]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = _tokenize_records(tokenizer, train_records, templates, int(config["max_length"]))
    validation_dataset = _tokenize_records(
        tokenizer, validation_records, templates, int(config["max_length"])
    )
    collator = CompletionCollator(int(tokenizer.pad_token_id))
    run_dir = project_root / config["output_dir"] / f"seed_{seed}"
    resume_from: str | bool = False
    if resume == "auto":
        candidates = sorted(
            (run_dir / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        resume_from = str(candidates[-1]) if candidates else False
    elif resume:
        resume_from = resume

    def run_attempt(batch_size: int, gradient_accumulation: int) -> Any:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(config["lora_rank"]),
                lora_alpha=int(config["lora_alpha"]),
                lora_dropout=float(config["lora_dropout"]),
                target_modules=list(config["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        argument_payload: dict[str, Any] = {
            "output_dir": str(run_dir / "checkpoints"),
            "num_train_epochs": float(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "warmup_ratio": float(config["warmup_ratio"]),
            "weight_decay": float(config["weight_decay"]),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": gradient_accumulation,
            "lr_scheduler_type": "cosine",
            "save_strategy": "steps",
            "save_steps": int(config["save_steps"]),
            "eval_steps": int(config["eval_steps"]),
            "logging_steps": 10,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": compute_dtype == torch.float16,
            "bf16": compute_dtype == torch.bfloat16,
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "report_to": [],
            "seed": seed,
            "data_seed": seed,
            "remove_unused_columns": False,
        }
        if config.get("max_steps") is not None:
            argument_payload["max_steps"] = int(config["max_steps"])
        arguments = _training_arguments(
            TrainingArguments,
            argument_payload,
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
        )
        trainer.train(resume_from_checkpoint=resume_from)
        best_dir = run_dir / "best_adapter"
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
        trainer.save_state()
        return trainer

    try:
        trainer = run_attempt(
            int(config["micro_batch_size"]), int(config["gradient_accumulation_steps"])
        )
        effective_batch = int(config["micro_batch_size"]) * int(
            config["gradient_accumulation_steps"]
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        trainer = run_attempt(1, 16)
        effective_batch = 16

    state_path = run_dir / "trainer_state_summary.json"
    state_path.write_text(
        json.dumps(
            {
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "global_step": trainer.state.global_step,
                "effective_batch_size": effective_batch,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = [state_path, *(run_dir / "best_adapter").glob("*")]
    manifest = RunManifest(
        command="poetry-lm train lora",
        git_commit=git_commit(project_root),
        resolved_config=config,
        dataset_sha256=sha256_file(dataset_dir / "dataset_manifest.json"),
        model_id=config["model_id"],
        model_revision=config["model_revision"],
        seed=seed,
        device=torch.cuda.get_device_name(torch.cuda.current_device()),
        versions=package_versions(
            ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
        ),
        artifacts=artifact_hashes(artifacts, project_root),
    )
    write_manifest(run_dir / "run_manifest.json", manifest)
    return run_dir
