from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .constraints import RhymeLexicon
from .manifests import sha256_file
from .research_data import read_poems
from .research_metrics import (
    CorpusNoveltyIndex,
    aggregate_metric,
    choose_rhyme_lambda,
    consecutive_repetition_rate,
    distinct_n,
    paired_bootstrap_difference,
    rhyme_consistency,
    self_bleu,
    structural_metrics,
    wilson_interval,
)
from .research_types import GenerationRecord
from .templates import load_templates

PRIMARY_METRICS = (
    "structural_exact_match",
    "segment_length_accuracy",
    "punctuation_accuracy",
    "stanza_accuracy",
    "rhyme_consistency",
    "repetition_rate",
    "train_8gram_overlap",
    "normalized_nearest_train_lcs",
)

PAIRED_COMPARISONS = (
    ("char_hard_structure", "char_unconstrained"),
    ("qwen_lora_hard_structure", "qwen_lora_unconstrained"),
    ("qwen_lora_hard_structure", "qwen_lora_best_of_8"),
    ("qwen_lora_hard_structure_rhyme", "qwen_lora_hard_structure"),
)


def read_generations(path: Path) -> list[GenerationRecord]:
    with path.open(encoding="utf-8") as handle:
        return [
            GenerationRecord.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


def _quartiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0}
    ordered = sorted(values)
    if len(ordered) == 1:
        q1 = median = q3 = ordered[0]
    else:
        q1, median, q3 = statistics.quantiles(
            ordered, n=4, method="inclusive"
        )[0], statistics.median(ordered), statistics.quantiles(
            ordered, n=4, method="inclusive"
        )[2]
    return {"median": median, "q1": q1, "q3": q3, "iqr": q3 - q1}


def _system_key(record: GenerationRecord) -> str:
    if record.system == "qwen_lora_hard_structure_rhyme":
        return f"{record.system}@lambda={record.rhyme_lambda:g}"
    return record.system


def _prompt_id(record: GenerationRecord) -> str:
    return f"{record.template_id}|{record.theme}"


def _enrich(
    records: Sequence[GenerationRecord],
    templates_path: Path,
    rhyme_path: Path,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], CorpusNoveltyIndex]:
    templates = load_templates(templates_path)
    lexicon = RhymeLexicon.from_json(rhyme_path)
    training_texts = [
        record.normalized_text
        for record in read_poems(dataset_dir / "splits/train.jsonl")
    ]
    novelty = CorpusNoveltyIndex(training_texts, width=8)
    training_ngrams = novelty.all_ngrams
    rows: list[dict[str, Any]] = []
    for record in records:
        template = templates[record.template_id]
        metrics = {
            **structural_metrics(record.text, template),
            "rhyme_consistency": rhyme_consistency(record.text, template, lexicon),
            "repetition_rate": consecutive_repetition_rate(record.text),
            "train_8gram_overlap": _overlap(
                _character_ngrams(record.text, 8), training_ngrams
            ),
            "normalized_nearest_train_lcs": novelty.normalized_nearest_lcs(record.text),
        }
        rows.append({**record.to_dict(), "metrics": metrics})
    return rows, novelty


def _character_ngrams(text: str, width: int) -> list[str]:
    compact = "".join(
        character
        for character in text
        if not character.isspace() and character not in "，。！？；"
    )
    return [compact[index : index + width] for index in range(len(compact) - width + 1)]


def _overlap(grams: Sequence[str], reference: set[str]) -> float:
    return sum(gram in reference for gram in grams) / len(grams) if grams else 0.0


def _aggregate_system(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    texts = [str(row["text"]) for row in rows if row["text"]]
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    token_rates = [
        float(row["generated_tokens"]) / float(row["elapsed_seconds"])
        for row in rows
        if float(row["elapsed_seconds"]) > 0
    ]
    memory = [
        float(row["peak_memory_mb"])
        for row in rows
        if row["peak_memory_mb"] is not None
    ]
    exact = sum(bool(row["structural_exact_match"]) for row in metrics)
    exact_interval = wilson_interval(exact, len(rows))
    dead_ends = sum(bool(row["dead_end"]) for row in rows)
    dead_end_interval = wilson_interval(dead_ends, len(rows))
    total_minutes = sum(elapsed) / 60
    return {
        "outputs": len(rows),
        "independent_prompts": len({_prompt_id(GenerationRecord.from_dict(row)) for row in rows}),
        "structural_exact_match": exact / len(rows) if rows else 0.0,
        "structural_exact_match_wilson95": list(exact_interval),
        "segment_length_accuracy": aggregate_metric(metrics, "segment_length_accuracy"),
        "punctuation_accuracy": aggregate_metric(metrics, "punctuation_accuracy"),
        "stanza_accuracy": aggregate_metric(metrics, "stanza_accuracy"),
        "rhyme_consistency": aggregate_metric(metrics, "rhyme_consistency"),
        "constraint_dead_end_rate": dead_ends / len(rows) if rows else 0.0,
        "constraint_dead_end_wilson95": list(dead_end_interval),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_4": distinct_n(texts, 4),
        "self_bleu_4": self_bleu(texts, 4),
        "repetition_rate": aggregate_metric(metrics, "repetition_rate"),
        "train_8gram_overlap": aggregate_metric(metrics, "train_8gram_overlap"),
        "normalized_nearest_train_lcs": aggregate_metric(
            metrics, "normalized_nearest_train_lcs"
        ),
        "valid_outputs_per_gpu_minute": exact / total_minutes if total_minutes else 0.0,
        "latency_seconds": _quartiles(elapsed),
        "tokens_per_second": _quartiles(token_rates),
        "peak_cuda_memory_mb": _quartiles(memory),
    }


def _prompt_aggregates(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record = GenerationRecord.from_dict(row)
        grouped[(_system_key(record), _prompt_id(record))].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (system, prompt), members in grouped.items():
        result[system][prompt] = {
            metric: statistics.fmean(float(row["metrics"][metric]) for row in members)
            for metric in PRIMARY_METRICS
        }
    return dict(result)


def _bootstrap_comparisons(
    prompt_rows: dict[str, dict[str, dict[str, float]]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for left, right in PAIRED_COMPARISONS:
        matching_left = [name for name in prompt_rows if name == left or name.startswith(f"{left}@")]
        matching_right = [name for name in prompt_rows if name == right or name.startswith(f"{right}@")]
        if len(matching_left) != 1 or len(matching_right) != 1:
            continue
        left_name, right_name = matching_left[0], matching_right[0]
        prompts = sorted(set(prompt_rows[left_name]).intersection(prompt_rows[right_name]))
        if not prompts:
            continue
        comparisons[f"{left_name}_minus_{right_name}"] = {
            "paired_prompts": len(prompts),
            "metrics": {
                metric: paired_bootstrap_difference(
                    [prompt_rows[left_name][prompt][metric] for prompt in prompts],
                    [prompt_rows[right_name][prompt][metric] for prompt in prompts],
                    resamples=10_000,
                    seed=2026,
                )
                for metric in PRIMARY_METRICS
            },
        }
    return comparisons


def _write_flat_csv(path: Path, systems: dict[str, dict[str, Any]]) -> None:
    columns = [
        "system",
        "outputs",
        "independent_prompts",
        "structural_exact_match",
        "segment_length_accuracy",
        "punctuation_accuracy",
        "stanza_accuracy",
        "rhyme_consistency",
        "constraint_dead_end_rate",
        "distinct_1",
        "distinct_2",
        "distinct_4",
        "self_bleu_4",
        "repetition_rate",
        "train_8gram_overlap",
        "normalized_nearest_train_lcs",
        "valid_outputs_per_gpu_minute",
        "median_latency_seconds",
        "median_tokens_per_second",
        "peak_cuda_memory_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for system, row in sorted(systems.items()):
            writer.writerow(
                {
                    **{name: row.get(name) for name in columns},
                    "system": system,
                    "median_latency_seconds": row["latency_seconds"]["median"],
                    "median_tokens_per_second": row["tokens_per_second"]["median"],
                    "peak_cuda_memory_mb": row["peak_cuda_memory_mb"]["median"],
                }
            )


def evaluate_generations(
    manifest_path: Path,
    config_path: Path,
    project_root: Path,
    output_dir: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = read_generations(manifest_path)
    if not records:
        raise ValueError("generation manifest is empty")
    templates_path = project_root / config["templates_file"]
    rhyme_path = project_root / config["rhyme_file"]
    dataset_dir = project_root / config["dataset_dir"]
    enriched, _ = _enrich(records, templates_path, rhyme_path, dataset_dir)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_split[str(row["dataset_split"])].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for split, split_rows in sorted(by_split.items()):
        systems_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            systems_grouped[_system_key(GenerationRecord.from_dict(row))].append(row)
        systems = {
            system: _aggregate_system(members)
            for system, members in sorted(systems_grouped.items())
        }
        prompts = _prompt_aggregates(split_rows)
        comparisons = _bootstrap_comparisons(prompts)
        summaries[split] = {"systems": systems, "paired_comparisons": comparisons}
        _write_flat_csv(output_dir / f"main_results_{split}.csv", systems)

    summary = {
        "protocol_version": 1,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": sha256_file(manifest_path),
        "benchmark_config_sha256": sha256_file(config_path),
        "splits": summaries,
        "resume_claim_policy": (
            "Use improved/outperformed only when the paired 95% CI excludes zero."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "generations_enriched.jsonl").open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary_path


def select_validation_lambda(
    manifest_path: Path,
    output_path: Path,
) -> Path:
    records = [
        record
        for record in read_generations(manifest_path)
        if record.system == "qwen_lora_hard_structure_rhyme"
    ]
    if not records or any(record.dataset_split != "validation" for record in records):
        raise ValueError("lambda selection accepts validation generations only")
    grouped: dict[float, list[GenerationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.rhyme_lambda].append(record)
    required = {0.0, 0.5, 1.0, 2.0, 4.0}
    if set(grouped) != required:
        raise ValueError(f"lambda sweep is incomplete; observed {sorted(grouped)}")
    rows: dict[float, dict[str, float]] = {}
    for value, members in grouped.items():
        texts = [record.text for record in members]
        rows[value] = {
            "distinct_2": distinct_n(texts, 2),
            "repetition_rate": statistics.fmean(
                consecutive_repetition_rate(text) for text in texts
            ),
            "rhyme_consistency": statistics.fmean(
                float(record.metrics["rhyme_consistency"]) for record in members
            ),
        }
    selected = choose_rhyme_lambda(rows)
    payload = {
        "selected_lambda": selected,
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": sha256_file(manifest_path),
        "selection_rule": (
            "highest rhyme consistency with Distinct-2 >= 95% of lambda=0 and "
            "repetition increase <= 0.02; ties choose smallest lambda"
        ),
        "validation_metrics": {str(key): value for key, value in sorted(rows.items())},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def generation_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
