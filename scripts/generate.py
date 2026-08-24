#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from poetry_lm.config import ModelConfig  # noqa: E402
from poetry_lm.generation import generate_ids, make_generator  # noqa: E402
from poetry_lm.model import DecoderOnlyLM  # noqa: E402
from poetry_lm.training import load_checkpoint, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    vocab: list[str] = checkpoint["vocab"]
    stoi = {character: index for index, character in enumerate(vocab)}
    itos = {index: character for index, character in enumerate(vocab)}

    missing = sorted(set(args.prompt).difference(stoi))
    if missing:
        raise ValueError(f"prompt contains characters absent from the vocabulary: {missing!r}")

    model_config = ModelConfig(**checkpoint["model_config"])
    model = DecoderOnlyLM(len(vocab), model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt_ids = [stoi[character] for character in args.prompt]
    generator = make_generator(device, args.seed)
    generated = generate_ids(
        model,
        prompt_ids,
        args.max_new_chars,
        args.temperature,
        args.top_k,
        device,
        generator,
    )
    text = "".join(itos[index] for index in generated)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
