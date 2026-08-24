from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class CorpusData:
    text: str
    vocab: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]
    train_ids: list[int]
    val_ids: list[int]

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text).difference(self.stoi))
        if missing:
            raise ValueError(f"characters absent from vocabulary: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(index)] for index in ids)


def load_corpus(path: Path, validation_ratio: float = 0.1) -> CorpusData:
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"corpus is empty: {path}")

    vocab = sorted(set(text))
    stoi = {character: index for index, character in enumerate(vocab)}
    itos = {index: character for character, index in stoi.items()}
    encoded = [stoi[character] for character in text]
    split_index = int(len(encoded) * (1.0 - validation_ratio))

    return CorpusData(
        text=text,
        vocab=vocab,
        stoi=stoi,
        itos=itos,
        train_ids=encoded[:split_index],
        val_ids=encoded[split_index:],
    )


def max_start_index(ids: Sequence[int], context_length: int) -> int:
    result = len(ids) - context_length - 1
    if result < 0:
        raise ValueError("sequence is too short for the selected context length")
    return result


def sample_starts(
    ids: Sequence[int],
    batch_size: int,
    context_length: int,
    rng: random.Random,
) -> list[int]:
    maximum = max_start_index(ids, context_length)
    return [rng.randint(0, maximum) for _ in range(batch_size)]


def fixed_validation_starts(
    ids: Sequence[int],
    batch_size: int,
    context_length: int,
    num_batches: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        sample_starts(ids, batch_size, context_length, rng)
        for _ in range(num_batches)
    ]


def batch_from_starts(
    ids: Sequence[int],
    starts: Sequence[int],
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [ids[start : start + context_length] for start in starts]
    targets = [ids[start + 1 : start + context_length + 1] for start in starts]
    return (
        torch.tensor(features, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def sample_batch(
    ids: Sequence[int],
    batch_size: int,
    context_length: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = sample_starts(ids, batch_size, context_length, rng)
    return batch_from_starts(ids, starts, context_length, device)
