#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MEBIBYTE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an edited Song-ci character corpus")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="ci.song.*.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-mib",
        type=float,
        default=5.0,
        help="approximate target size while preserving whole poems",
    )
    return parser.parse_args()


def load_poems(input_dir: Path, pattern: str) -> list[str]:
    poems: list[str] = []
    # Lexicographic order intentionally reproduces the historical project export.
    for path in sorted(input_dir.glob(pattern), key=lambda item: item.name):
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            paragraphs = record.get("paragraphs", [])
            if paragraphs:
                poems.append("\n".join(paragraphs))
    return poems


def subset_nearest_target(poems: list[str], target_bytes: int) -> str:
    selected: list[str] = []
    current = ""
    current_size = 0
    for poem in poems:
        candidate = poem if not selected else current + "\n\n" + poem
        candidate_size = len(candidate.encode("utf-8"))
        if candidate_size <= target_bytes:
            selected.append(poem)
            current = candidate
            current_size = candidate_size
            continue
        if abs(target_bytes - candidate_size) < abs(target_bytes - current_size):
            current = candidate
        break
    return current


def main() -> None:
    args = parse_args()
    poems = load_poems(args.input_dir, args.pattern)
    if not poems:
        raise FileNotFoundError(
            f"no non-empty poems matched {args.pattern!r} under {args.input_dir}"
        )
    corpus = subset_nearest_target(poems, int(args.target_mib * MEBIBYTE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    print(f"matched poems: {len(poems):,}")
    print(f"output bytes: {args.output.stat().st_size:,}")
    print(f"sha256: {digest}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
