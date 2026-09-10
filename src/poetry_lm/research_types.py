from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SegmentSpec:
    char_count: int
    punctuation: str
    stanza_end: bool = False
    rhyme_slot: bool = False
    rhyme_chain: int | None = None
    required_tone: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SegmentSpec:
        return cls(**payload)


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    cipai: str
    variant: int
    segments: tuple[SegmentSpec, ...]
    source_commit: str
    support_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TemplateSpec:
        data = dict(payload)
        data["segments"] = tuple(SegmentSpec.from_dict(row) for row in data["segments"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def visible_characters(self) -> int:
        return sum(segment.char_count for segment in self.segments)

    def format_string(self) -> str:
        pieces: list[str] = []
        for segment in self.segments:
            pieces.append(f"{segment.char_count}{segment.punctuation}")
            if segment.stanza_end:
                pieces.append("//")
        return "".join(pieces).removesuffix("//")


@dataclass(frozen=True)
class PoemRecord:
    id: str
    author: str
    rhythmic: str
    paragraphs: tuple[str, ...]
    text: str
    normalized_text: str
    template_id: str | None
    template_exact: bool
    split: str
    fold: int
    source_file: str
    source_index: int
    sha256: str
    tags: tuple[str, ...] = ()
    group_id: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PoemRecord:
        data = dict(payload)
        data["paragraphs"] = tuple(data.get("paragraphs", ()))
        data["tags"] = tuple(data.get("tags", ()))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRecord:
    system: str
    template_id: str
    theme: str
    training_seed: int | None
    decoding_seed: int
    prompt: str
    text: str
    elapsed_seconds: float
    generated_tokens: int
    peak_memory_mb: float | None
    dead_end: bool
    metrics: dict[str, float | int | bool]
    model_id: str
    model_revision: str
    artifact: str | None = None
    dataset_split: str = "test"
    rhyme_lambda: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GenerationRecord:
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    command: str
    git_commit: str
    resolved_config: dict[str, Any]
    dataset_sha256: str | None
    model_id: str | None
    model_revision: str | None
    seed: int | None
    device: str
    versions: dict[str, str]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
