from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .model import DecoderOnlyLM


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def sample_next_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / temperature

    if top_k is not None:
        effective_top_k = min(top_k, logits.numel())
        threshold = torch.topk(logits, k=effective_top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probabilities = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_ids(
    model: DecoderOnlyLM,
    prompt_ids: Sequence[int],
    max_new_chars: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
    generator: torch.Generator,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one character")

    model.eval()
    generated = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(max_new_chars):
            conditioned = generated[-model.config.context_length :]
            tokens = torch.tensor([conditioned], dtype=torch.long, device=device)
            logits = model(tokens)[0, -1]
            generated.append(sample_next_id(logits, temperature, top_k, generator))
    return generated
