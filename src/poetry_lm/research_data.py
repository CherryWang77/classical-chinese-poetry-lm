from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .research_types import PoemRecord, TemplateSpec
from .templates import (
    aliases,
    exact_template_match,
    finalize_template,
    format_from_resource,
    match_lengths,
    normalize_text,
    render_with_template_breaks,
    split_segments,
)

INVALID_TEXT_MARKERS = {"□", "�"}
class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class WorkingRecord:
    author: str
    rhythmic: str
    paragraphs: tuple[str, ...]
    text: str
    normalized_text: str
    template_id: str | None
    source_file: str
    source_index: int
    tags: tuple[str, ...]
    sha256: str
    fold: int = -1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_sources(config: dict[str, Any], project_root: Path) -> None:
    source = config["source"]
    raw_root = project_root / "data/research/raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            source["songci_repository"],
            source["songci_commit"],
            raw_root / "chinese-poetry",
            "宋词",
        ),
        (
            source["rhyme_repository"],
            source["rhyme_commit"],
            raw_root / "chinese_word_rhyme",
            None,
        ),
    ]
    for repository, commit, destination, sparse_path in specifications:
        if not destination.exists():
            command = ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)]
            subprocess.run(command, check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit], check=True)
        if sparse_path:
            subprocess.run(["git", "-C", str(destination), "sparse-checkout", "set", sparse_path], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)


def _resolve(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_base_templates(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, TemplateSpec], dict[str, str], Path, Path]:
    tunes_path = _resolve(config["input"]["tunes_file"], project_root)
    rhyme_path = _resolve(config["input"]["rhyme_file"], project_root)
    tune_payload = json.loads(tunes_path.read_text(encoding="utf-8"))
    source_commit = config["source"]["rhyme_commit"]
    templates = {
        f"{cipai}:v{variant}": format_from_resource(
            cipai, int(variant), tune_payload, source_commit
        )
        for cipai, variant in config["target_templates"].items()
    }
    alias_to_template: dict[str, str] = {}
    for template in templates.values():
        alias_to_template[template.cipai] = template.template_id
    return templates, alias_to_template, tunes_path, rhyme_path


def _raw_records(
    songci_dir: Path,
    templates: dict[str, TemplateSpec],
    alias_to_template: dict[str, str],
) -> list[WorkingRecord]:
    rows: list[WorkingRecord] = []
    for source_path in sorted(songci_dir.glob("ci.song.*.json"), key=lambda path: path.name):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        for source_index, record in enumerate(payload):
            paragraphs = tuple(str(value).strip() for value in record.get("paragraphs", []) if str(value).strip())
            if not paragraphs:
                continue
            original = "\n".join(paragraphs)
            normalized = normalize_text(original)
            if not normalized or any(marker in normalized for marker in INVALID_TEXT_MARKERS):
                continue
            template_id = next(
                (
                    alias_to_template[name]
                    for name in aliases(str(record.get("rhythmic", "")))
                    if name in alias_to_template
                ),
                None,
            )
            if template_id is not None and not match_lengths(normalized, templates[template_id]):
                template_id = None
            rows.append(
                WorkingRecord(
                    author=str(record.get("author") or "<unknown>").strip(),
                    rhythmic=str(record.get("rhythmic") or "").strip(),
                    paragraphs=paragraphs,
                    text=original,
                    normalized_text=normalized,
                    template_id=template_id,
                    source_file=source_path.name,
                    source_index=source_index,
                    tags=tuple(str(tag) for tag in record.get("tags", [])),
                    sha256=sha256_text(normalized),
                )
            )
    return rows


def remove_exact_duplicates(records: list[WorkingRecord]) -> list[WorkingRecord]:
    unique: dict[str, WorkingRecord] = {}
    for record in records:
        unique.setdefault(record.sha256, record)
    return list(unique.values())


def character_shingles(text: str, width: int) -> set[str]:
    compact = text.replace("\n", "")
    if len(compact) <= width:
        return {compact}
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def join_near_duplicates(
    records: list[WorkingRecord],
    groups: UnionFind,
    width: int,
    threshold: float,
    num_perm: int = 128,
) -> None:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as error:
        raise RuntimeError("near-duplicate detection requires `pip install -e '.[data]'`") from error

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    shingle_sets: list[set[str]] = []
    for index, record in enumerate(records):
        shingles = character_shingles(record.normalized_text, width)
        signature = MinHash(num_perm=num_perm, seed=2026)
        for shingle in sorted(shingles):
            signature.update(shingle.encode("utf-8"))
        for candidate_key in lsh.query(signature):
            candidate = int(candidate_key)
            other = shingle_sets[candidate]
            union = len(shingles | other)
            similarity = len(shingles & other) / union if union else 1.0
            if similarity >= threshold:
                groups.union(index, candidate)
        lsh.insert(str(index), signature)
        shingle_sets.append(shingles)


def join_authors(records: list[WorkingRecord], groups: UnionFind) -> None:
    first: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.author == "<unknown>":
            continue
        if record.author in first:
            groups.union(index, first[record.author])
        else:
            first[record.author] = index


def assign_folds(
    records: list[WorkingRecord], groups: UnionFind, num_folds: int, seed: int
) -> None:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:
        raise RuntimeError("group splitting requires `pip install -e '.[data]'`") from error

    labels = [record.template_id or "__other__" for record in records]
    group_ids = [str(groups.find(index)) for index in range(len(records))]
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    assigned = [-1] * len(records)
    dummy = [0] * len(records)
    for fold, (_, held_out) in enumerate(splitter.split(dummy, labels, group_ids)):
        for index in held_out:
            assigned[int(index)] = fold
    if any(fold < 0 for fold in assigned):
        raise AssertionError("not every record received a fold")
    for record, fold in zip(records, assigned, strict=True):
        record.fold = fold


def _split_name(fold: int, validation_fold: int, test_fold: int) -> str:
    if fold == test_fold:
        return "test"
    if fold == validation_fold:
        return "validation"
    return "train"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify_no_group_leakage(records: list[PoemRecord]) -> None:
    authors: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.author != "<unknown>":
            authors[record.author].add(record.split)
        hashes[record.sha256].add(record.split)
        groups[record.group_id].add(record.split)
    leaked_authors = [author for author, splits in authors.items() if len(splits) > 1]
    leaked_hashes = [digest for digest, splits in hashes.items() if len(splits) > 1]
    leaked_groups = [group for group, splits in groups.items() if len(splits) > 1]
    if leaked_authors or leaked_hashes or leaked_groups:
        raise AssertionError(
            "split leakage: "
            f"{len(leaked_authors)} authors, {len(leaked_hashes)} exact texts, "
            f"{len(leaked_groups)} union-find groups"
        )


def audit_templates(
    records: list[PoemRecord],
    templates: dict[str, TemplateSpec],
    rhyme_path: Path,
) -> dict[str, Any]:
    from .constraints import RhymeLexicon, SongCiAutomaton

    lexicon = RhymeLexicon.from_json(rhyme_path)
    audit: dict[str, Any] = {"protocol_version": 1, "samples_per_template": 3, "templates": {}}
    for template_id, template in templates.items():
        candidates = sorted(
            (
                record
                for record in records
                if record.split == "train"
                and record.template_id == template_id
                and record.template_exact
            ),
            key=lambda record: record.id,
        )[:3]
        if len(candidates) != 3:
            raise AssertionError(f"{template_id} has fewer than three audit examples")
        automaton = SongCiAutomaton(template, lexicon)
        examples: list[dict[str, Any]] = []
        for record in candidates:
            rendered = render_with_template_breaks(record.normalized_text, template)
            automaton.require_complete(rendered)
            rows = split_segments(record.normalized_text)
            assert rows is not None
            rhyme_slots = []
            for index, segment in enumerate(template.segments):
                if not segment.rhyme_slot:
                    continue
                character = rows[index][0][-1]
                rhyme_slots.append(
                    {
                        "segment": index,
                        "character": character,
                        "chain": segment.rhyme_chain,
                        "required_tone": segment.required_tone,
                        "lexicon_groups": sorted(lexicon.groups(character)),
                    }
                )
            examples.append(
                {
                    "id": record.id,
                    "author": record.author,
                    "source_file": record.source_file,
                    "source_index": record.source_index,
                    "normalized_text": record.normalized_text,
                    "canonical_fsa_text": rendered,
                    "accepted": True,
                    "rhyme_slots": rhyme_slots,
                }
            )
        audit["templates"][template_id] = {
            "support_counts": template.support_counts,
            "segment_count": len(template.segments),
            "rhyme_chain_count": len(
                {
                    segment.rhyme_chain
                    for segment in template.segments
                    if segment.rhyme_chain is not None
                }
            ),
            "examples": examples,
        }
    return audit


def build_dataset(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    templates, alias_to_template, tunes_path, rhyme_path = _load_base_templates(config, project_root)
    songci_dir = _resolve(config["input"]["songci_dir"], project_root)
    output_dir = _resolve(config["output_dir"], project_root)
    records = remove_exact_duplicates(_raw_records(songci_dir, templates, alias_to_template))

    groups = UnionFind(len(records))
    join_authors(records, groups)
    join_near_duplicates(
        records,
        groups,
        int(config["near_duplicate_ngram"]),
        float(config["near_duplicate_threshold"]),
    )
    assign_folds(records, groups, int(config["num_folds"]), int(config["split_seed"]))

    validation_fold = int(config["validation_fold"])
    test_fold = int(config["test_fold"])
    final_templates: dict[str, TemplateSpec] = {}
    for template_id, base in templates.items():
        training = [
            record.normalized_text
            for record in records
            if record.template_id == template_id
            and _split_name(record.fold, validation_fold, test_fold) == "train"
        ]
        provisional = finalize_template(base, training, {})
        support = Counter(
            _split_name(record.fold, validation_fold, test_fold)
            for record in records
            if record.template_id == template_id
            and exact_template_match(record.normalized_text, provisional)
        )
        final_templates[template_id] = finalize_template(base, training, dict(support))

    final_records: list[PoemRecord] = []
    for index, record in enumerate(records):
        split = _split_name(record.fold, validation_fold, test_fold)
        template_exact = bool(
            record.template_id
            and exact_template_match(record.normalized_text, final_templates[record.template_id])
        )
        final_records.append(
            PoemRecord(
                id=f"songci-{index:05d}",
                author=record.author,
                rhythmic=record.rhythmic,
                paragraphs=record.paragraphs,
                text=record.text,
                normalized_text=record.normalized_text,
                template_id=record.template_id,
                template_exact=template_exact,
                split=split,
                fold=record.fold,
                source_file=record.source_file,
                source_index=record.source_index,
                sha256=record.sha256,
                tags=record.tags,
                group_id=str(groups.find(index)),
            )
        )
    verify_no_group_leakage(final_records)

    poems_path = output_dir / "poems.jsonl"
    selected_path = output_dir / "selected_poems.jsonl"
    templates_path = output_dir / "templates.json"
    audit_path = output_dir / "template_audit.json"
    _write_jsonl(poems_path, (record.to_dict() for record in final_records))
    _write_jsonl(
        selected_path,
        (
            record.to_dict()
            for record in final_records
            if record.template_id is not None and record.template_exact
        ),
    )
    split_paths: dict[str, Path] = {}
    selected_split_paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        split_paths[split] = output_dir / "splits" / f"{split}.jsonl"
        selected_split_paths[split] = output_dir / "selected" / f"{split}.jsonl"
        _write_jsonl(
            split_paths[split],
            (record.to_dict() for record in final_records if record.split == split),
        )
        _write_jsonl(
            selected_split_paths[split],
            (
                record.to_dict()
                for record in final_records
                if record.split == split
                and record.template_id is not None
                and record.template_exact
            ),
        )
    _write_json(
        templates_path,
        {
            "source_commit": config["source"]["rhyme_commit"],
            "templates": [template.to_dict() for template in final_templates.values()],
        },
    )
    _write_json(audit_path, audit_templates(final_records, final_templates, rhyme_path))
    manifest = {
        "protocol_version": 1,
        "config_sha256": sha256_file(config_path),
        "source_commits": {
            "songci": config["source"]["songci_commit"],
            "rhyme": config["source"]["rhyme_commit"],
        },
        "source_repositories": {
            "songci": config["source"]["songci_repository"],
            "rhyme": config["source"]["rhyme_repository"],
        },
        "licenses": {"songci": "MIT", "rhyme": "MIT"},
        "source_file_sha256": {
            "Ci_Tunes.json": sha256_file(tunes_path),
            "Ci_Word_Tune.json": sha256_file(rhyme_path),
        },
        "counts": {
            "all": len(final_records),
            "selected": sum(record.template_exact for record in final_records),
            "splits": dict(Counter(record.split for record in final_records)),
            "selected_splits": dict(
                Counter(record.split for record in final_records if record.template_exact)
            ),
        },
        "artifacts": {
            "poems.jsonl": sha256_file(poems_path),
            "selected_poems.jsonl": sha256_file(selected_path),
            "templates.json": sha256_file(templates_path),
            "template_audit.json": sha256_file(audit_path),
            **{
                f"splits/{split}.jsonl": sha256_file(path)
                for split, path in split_paths.items()
            },
            **{
                f"selected/{split}.jsonl": sha256_file(path)
                for split, path in selected_split_paths.items()
            },
        },
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def read_poems(path: Path) -> list[PoemRecord]:
    with path.open(encoding="utf-8") as handle:
        return [PoemRecord.from_dict(json.loads(line)) for line in handle if line.strip()]
