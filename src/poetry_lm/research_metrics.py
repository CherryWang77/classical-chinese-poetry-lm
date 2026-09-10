from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from .constraints import RhymeLexicon
from .research_types import TemplateSpec
from .templates import split_segments


def structural_metrics(text: str, template: TemplateSpec) -> dict[str, float | bool]:
    rows = split_segments(text)
    expected = template.segments
    if rows is None:
        return {
            "structural_exact_match": False,
            "segment_length_accuracy": 0.0,
            "punctuation_accuracy": 0.0,
            "stanza_accuracy": 0.0,
        }
    comparable = min(len(rows), len(expected))
    correct_lengths = sum(
        len(rows[index][0]) == expected[index].char_count for index in range(comparable)
    )
    correct_punctuation = sum(
        rows[index][1] == expected[index].punctuation for index in range(comparable)
    )
    denominator = max(len(rows), len(expected), 1)
    segment_exact = len(rows) == len(expected) and all(
        len(content) == segment.char_count and punctuation == segment.punctuation
        for (content, punctuation), segment in zip(rows, expected, strict=True)
    )

    expected_boundaries = {
        index + 1
        for index, segment in enumerate(expected)
        if segment.stanza_end and index + 1 < len(expected)
    }
    observed_boundaries: set[int] = set()
    cumulative = 0
    blocks = [block for block in text.strip().split("\n\n") if block.strip()]
    for block in blocks[:-1]:
        parsed = split_segments(block)
        cumulative += len(parsed or ())
        observed_boundaries.add(cumulative)
    boundary_denominator = max(len(expected_boundaries | observed_boundaries), 1)
    stanza_accuracy = (
        len(expected_boundaries & observed_boundaries) / boundary_denominator
        if expected_boundaries or observed_boundaries
        else 1.0
    )
    exact = segment_exact and stanza_accuracy == 1.0
    return {
        "structural_exact_match": exact,
        "segment_length_accuracy": correct_lengths / denominator,
        "punctuation_accuracy": correct_punctuation / denominator,
        "stanza_accuracy": stanza_accuracy,
    }


def rhyme_consistency(text: str, template: TemplateSpec, lexicon: RhymeLexicon) -> float:
    rows = split_segments(text)
    if rows is None:
        return 0.0
    by_chain: dict[int, list[frozenset[str]]] = defaultdict(list)
    for index, segment in enumerate(template.segments):
        if not segment.rhyme_slot or segment.rhyme_chain is None or index >= len(rows):
            continue
        content = rows[index][0]
        if len(content) != segment.char_count or not content:
            continue
        by_chain[segment.rhyme_chain].append(lexicon.groups(content[-1]))
    numerator = 0.0
    denominator = 0
    for slots in by_chain.values():
        if len(slots) < 2:
            continue
        counts: Counter[str] = Counter()
        for groups in slots:
            for group in groups:
                counts[group] += 1
        denominator += len(slots)
        numerator += max(counts.values(), default=0)
    return numerator / denominator if denominator else 0.0


def visible_characters(text: str) -> list[str]:
    return [character for character in text if not character.isspace() and character not in "，。！？；"]


def ngrams(text: str, width: int) -> list[str]:
    characters = visible_characters(text)
    return ["".join(characters[index : index + width]) for index in range(len(characters) - width + 1)]


def distinct_n(texts: Sequence[str], width: int) -> float:
    grams = [gram for text in texts for gram in ngrams(text, width)]
    return len(set(grams)) / len(grams) if grams else 0.0


def consecutive_repetition_rate(text: str) -> float:
    characters = visible_characters(text)
    if len(characters) < 2:
        return 0.0
    repetitions = sum(
        left == right for left, right in zip(characters, characters[1:], strict=False)
    )
    return repetitions / (len(characters) - 1)


def corpus_ngram_overlap(text: str, corpus_ngrams: set[str], width: int = 8) -> float:
    grams = ngrams(text, width)
    return sum(gram in corpus_ngrams for gram in grams) / len(grams) if grams else 0.0


def longest_common_substring(left: str, right: str) -> int:
    a, b = visible_characters(left), visible_characters(right)
    previous = [0] * (len(b) + 1)
    best = 0
    for left_char in a:
        current = [0]
        for index, right_char in enumerate(b, start=1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


class CorpusNoveltyIndex:
    def __init__(self, corpus: Sequence[str], width: int = 8):
        self.corpus = tuple(corpus)
        self.width = width
        index: dict[str, set[int]] = defaultdict(set)
        for document_id, text in enumerate(corpus):
            for gram in set(ngrams(text, width)):
                index[gram].add(document_id)
        self.index = index

    @property
    def all_ngrams(self) -> set[str]:
        return set(self.index)

    def normalized_nearest_lcs(self, text: str, max_candidates: int = 20) -> float:
        votes: Counter[int] = Counter()
        for gram in set(ngrams(text, self.width)):
            votes.update(self.index.get(gram, ()))
        candidates = [document_id for document_id, _ in votes.most_common(max_candidates)]
        if not candidates:
            return 0.0
        denominator = max(len(visible_characters(text)), 1)
        return max(
            longest_common_substring(text, self.corpus[document_id]) / denominator
            for document_id in candidates
        )


def _modified_precision(candidate: str, references: Sequence[str], width: int) -> float:
    candidate_counts = Counter(ngrams(candidate, width))
    if not candidate_counts:
        return 0.0
    reference_max: Counter[str] = Counter()
    for reference in references:
        counts = Counter(ngrams(reference, width))
        for gram, count in counts.items():
            reference_max[gram] = max(reference_max[gram], count)
    clipped = sum(min(count, reference_max[gram]) for gram, count in candidate_counts.items())
    return clipped / sum(candidate_counts.values())


def self_bleu(texts: Sequence[str], max_order: int = 4) -> float:
    if len(texts) < 2:
        return 0.0
    scores: list[float] = []
    for index, candidate in enumerate(texts):
        references = [text for other, text in enumerate(texts) if other != index]
        precisions = [max(_modified_precision(candidate, references, n), 1e-9) for n in range(1, max_order + 1)]
        candidate_length = len(visible_characters(candidate))
        reference_lengths = [len(visible_characters(reference)) for reference in references]
        closest = min(reference_lengths, key=lambda length: (abs(length - candidate_length), length))
        brevity = 1.0 if candidate_length > closest else math.exp(1 - closest / max(candidate_length, 1))
        scores.append(brevity * math.exp(sum(math.log(value) for value in precisions) / max_order))
    return statistics.fmean(scores)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired inputs must have equal non-zero length")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    samples = [
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(resamples)
    ]
    samples.sort()
    lower = samples[int(0.025 * resamples)]
    upper = samples[min(resamples - 1, int(0.975 * resamples))]
    mean = statistics.fmean(differences)
    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return {
        "mean_difference": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "paired_effect_size": mean / deviation if deviation else 0.0,
    }


def choose_rhyme_lambda(
    rows: dict[float, dict[str, float]],
    baseline_lambda: float = 0.0,
) -> float:
    if baseline_lambda not in rows:
        raise ValueError("lambda=0 validation result is required")
    baseline = rows[baseline_lambda]
    eligible = [
        value
        for value, metrics in rows.items()
        if metrics["distinct_2"] >= 0.95 * baseline["distinct_2"]
        and metrics["repetition_rate"] <= baseline["repetition_rate"] + 0.02
    ]
    if not eligible:
        return 0.5
    best_rhyme = max(rows[value]["rhyme_consistency"] for value in eligible)
    return min(value for value in eligible if rows[value]["rhyme_consistency"] == best_rhyme)


def aggregate_metric(rows: Iterable[dict[str, float | int | bool]], name: str) -> float:
    values = [float(row[name]) for row in rows if name in row]
    return statistics.fmean(values) if values else 0.0
