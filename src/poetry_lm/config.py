from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    embed_dim: int
    num_layers: int
    num_heads: int
    dropout: float = 0.1
    attention_backend: str = "manual"
    positional_encoding: str = "learned"

    def validate(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.embed_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")
        if self.positional_encoding != "learned":
            raise ValueError("this implementation currently supports learned positions only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    input_file: str
    validation_ratio: float = 0.1

    def validate(self) -> None:
        if not self.input_file:
            raise ValueError("input_file must not be empty")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    max_steps: int
    seed: int = 42
    eval_seed: int = 1729
    num_val_batches: int = 5
    log_every: int = 100
    eval_every: int = 200
    save_every: int = 1000
    sample_every: int = 1000

    def validate(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "num_val_batches": self.num_val_batches,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "sample_every": self.sample_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    max_new_chars: int = 200
    temperature: float = 1.0
    top_k: int | None = 50
    seed: int = 2026

    def validate(self) -> None:
        if not self.prompt:
            raise ValueError("generation prompt must not be empty")
        if self.max_new_chars <= 0:
            raise ValueError("max_new_chars must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or null")


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        config = cls(
            run_name=payload["run_name"],
            data=DataConfig(**payload["data"]),
            model=ModelConfig(**payload["model"]),
            training=TrainingConfig(**payload["training"]),
            generation=GenerationConfig(**payload["generation"]),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if not self.run_name or any(ch.isspace() for ch in self.run_name):
            raise ValueError("run_name must be non-empty and contain no whitespace")
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.generation.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
