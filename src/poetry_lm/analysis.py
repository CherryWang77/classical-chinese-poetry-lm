from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics
from typing import Any, Iterable


PUNCTUATION = set("，。！？；：、（）《》〈〉“”‘’〔〕【】—…·,.!?;:()[]\"' ")


def split_poems(text: str) -> list[list[str]]:
    return [
        [line for line in block.splitlines() if line.strip()]
        for block in text.strip().split("\n\n")
        if block.strip()
    ]


def strip_punctuation(text: str) -> str:
    return "".join(character for character in text if character not in PUNCTUATION)


def counter_to_dict(counter: Counter[int] | Counter[str]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: item[0])}


def text_statistics(text: str, byte_size: int | None = None) -> dict[str, Any]:
    poems = split_poems(text)
    lines = [line for poem in poems for line in poem]
    stripped_lines = [strip_punctuation(line) for line in lines]
    line_lengths = [len(line) for line in stripped_lines]
    lines_per_poem = [len(poem) for poem in poems]

    visible_characters = [character for line in lines for character in line if not character.isspace()]
    punctuation_count = sum(character in PUNCTUATION for character in visible_characters)
    endings = [line[-1] for line in stripped_lines if line]

    return {
        "byte_size": byte_size if byte_size is not None else len(text.encode("utf-8")),
        "num_poems": len(poems),
        "num_lines": len(lines),
        "num_visible_characters": len(visible_characters),
        "mean_lines_per_poem": statistics.fmean(lines_per_poem) if lines_per_poem else 0.0,
        "mean_characters_per_line_without_punctuation": (
            statistics.fmean(line_lengths) if line_lengths else 0.0
        ),
        "median_characters_per_line_without_punctuation": (
            statistics.median(line_lengths) if line_lengths else 0.0
        ),
        "punctuation_rate": (
            punctuation_count / len(visible_characters) if visible_characters else 0.0
        ),
        "line_length_distribution": counter_to_dict(Counter(line_lengths)),
        "lines_per_poem_distribution": counter_to_dict(Counter(lines_per_poem)),
        "top_line_endings": Counter(endings).most_common(20),
    }


def file_statistics(path: Path) -> dict[str, Any]:
    return text_statistics(path.read_text(encoding="utf-8"), path.stat().st_size)


def normalized_distribution(distribution: dict[str, int], support: Iterable[str]) -> list[float]:
    total = sum(distribution.values())
    if total == 0:
        return [0.0 for _ in support]
    return [distribution.get(key, 0) / total for key in support]


def total_variation_distance(left: dict[str, int], right: dict[str, int]) -> float:
    support = sorted(set(left).union(right))
    left_probabilities = normalized_distribution(left, support)
    right_probabilities = normalized_distribution(right, support)
    return 0.5 * sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left_probabilities, right_probabilities)
    )


def compare_statistics(reference: dict[str, Any], generated: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_line_length_difference": (
            generated["mean_characters_per_line_without_punctuation"]
            - reference["mean_characters_per_line_without_punctuation"]
        ),
        "mean_lines_per_poem_difference": (
            generated["mean_lines_per_poem"] - reference["mean_lines_per_poem"]
        ),
        "punctuation_rate_difference": (
            generated["punctuation_rate"] - reference["punctuation_rate"]
        ),
        "line_length_total_variation": total_variation_distance(
            reference["line_length_distribution"], generated["line_length_distribution"]
        ),
        "lines_per_poem_total_variation": total_variation_distance(
            reference["lines_per_poem_distribution"],
            generated["lines_per_poem_distribution"],
        ),
    }
