from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .research_types import SegmentSpec, TemplateSpec

BOUNDARY_PATTERN = re.compile(r"([，。！？；])")
PUNCTUATION_MAP = str.maketrans({",": "，", ".": "。", "!": "！", "?": "？", ";": "；"})
RHYME_MARKERS = {"韵", "叶", "叠"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(PUNCTUATION_MAP)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def aliases(name: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", name).strip()
    values = [normalized, *re.split(r"[・·/]", normalized)]
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def split_segments(text: str) -> tuple[tuple[str, str], ...] | None:
    compact = normalize_text(text).replace("\n", "")
    if not compact:
        return None
    rows: list[tuple[str, str]] = []
    start = 0
    for match in BOUNDARY_PATTERN.finditer(compact):
        content = compact[start : match.start()]
        if not content:
            return None
        rows.append((content, match.group(1)))
        start = match.end()
    if compact[start:]:
        return None
    return tuple(rows)


def format_from_resource(
    cipai: str,
    variant: int,
    payload: dict[str, Any],
    source_commit: str,
) -> TemplateSpec:
    try:
        selected = payload[cipai]["formats"][variant]
    except (KeyError, IndexError) as error:
        raise ValueError(f"unknown template {cipai!r} variant {variant}") from error

    segments: list[SegmentSpec] = []
    count = 0
    rhyme_chain = -1
    previous_rhyme_tone: str | None = None
    for position in selected["tunes"]:
        count += 1
        marker = position.get("rhythm")
        if marker is None:
            continue
        rhyme_slot = str(marker) in RHYME_MARKERS
        tone = str(position.get("tune", "")) or None
        if rhyme_slot and (previous_rhyme_tone is None or tone != previous_rhyme_tone):
            rhyme_chain += 1
        if rhyme_slot:
            previous_rhyme_tone = tone
        segments.append(
            SegmentSpec(
                char_count=count,
                punctuation="。",
                stanza_end=bool(position.get("shift")),
                rhyme_slot=rhyme_slot,
                rhyme_chain=rhyme_chain if rhyme_slot else None,
                required_tone=tone if rhyme_slot else None,
            )
        )
        count = 0
    if count:
        raise ValueError(f"template {cipai!r} variant {variant} has an unterminated segment")
    if not segments:
        raise ValueError(f"template {cipai!r} variant {variant} has no segments")
    return TemplateSpec(
        template_id=f"{cipai}:v{variant}",
        cipai=cipai,
        variant=variant,
        segments=tuple(segments),
        source_commit=source_commit,
    )


def template_lengths(template: TemplateSpec) -> tuple[int, ...]:
    return tuple(segment.char_count for segment in template.segments)


def match_lengths(text: str, template: TemplateSpec) -> bool:
    rows = split_segments(text)
    return rows is not None and tuple(len(content) for content, _ in rows) == template_lengths(template)


def punctuation_pattern(text: str) -> tuple[str, ...] | None:
    rows = split_segments(text)
    return None if rows is None else tuple(punctuation for _, punctuation in rows)


def finalize_template(
    template: TemplateSpec,
    training_texts: Iterable[str],
    support_counts: dict[str, int],
) -> TemplateSpec:
    patterns = Counter(
        pattern
        for text in training_texts
        if (pattern := punctuation_pattern(text)) is not None
        and len(pattern) == len(template.segments)
    )
    if not patterns:
        raise ValueError(f"no training punctuation pattern for {template.template_id}")
    selected, _ = patterns.most_common(1)[0]
    segments = tuple(
        replace(segment, punctuation=punctuation)
        for segment, punctuation in zip(template.segments, selected, strict=True)
    )
    return replace(template, segments=segments, support_counts=dict(support_counts))


def exact_template_match(text: str, template: TemplateSpec) -> bool:
    rows = split_segments(text)
    if rows is None or len(rows) != len(template.segments):
        return False
    return all(
        len(content) == segment.char_count and punctuation == segment.punctuation
        for (content, punctuation), segment in zip(
            rows, template.segments, strict=True
        )
    )


def render_with_template_breaks(text: str, template: TemplateSpec) -> str:
    """Render a structurally matching corpus poem into the FSA's canonical layout."""
    rows = split_segments(text)
    if rows is None or len(rows) != len(template.segments):
        raise ValueError(f"text does not match the segment count for {template.template_id}")
    pieces: list[str] = []
    for (content, punctuation), segment in zip(rows, template.segments, strict=True):
        if len(content) != segment.char_count or punctuation != segment.punctuation:
            raise ValueError(f"text does not match {template.template_id}")
        pieces.append(content + punctuation)
        pieces.append("\n\n" if segment.stanza_end else "\n")
    return "".join(pieces)


def load_templates(path: Path) -> dict[str, TemplateSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["template_id"]: TemplateSpec.from_dict(row) for row in payload["templates"]}
