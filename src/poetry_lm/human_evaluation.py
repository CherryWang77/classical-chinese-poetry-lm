from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .research_evaluation import read_generations

PAIRINGS = (
    ("structure", "qwen_lora_unconstrained", "qwen_lora_hard_structure"),
    ("rhyme", "qwen_lora_hard_structure", "qwen_lora_hard_structure_rhyme"),
)


def _match_key(record: Any) -> tuple[Any, ...]:
    return (
        record.template_id,
        record.theme,
        record.training_seed,
        record.decoding_seed,
    )


def build_blind_evaluation_pack(
    generation_path: Path,
    form_path: Path,
    key_path: Path,
    seed: int = 31_415,
) -> tuple[Path, Path]:
    records = [record for record in read_generations(generation_path) if record.dataset_split == "test"]
    by_system = {
        system: {_match_key(record): record for record in records if record.system == system}
        for _, left, right in PAIRINGS
        for system in (left, right)
    }
    rng = random.Random(seed)
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for category, first_system, second_system in PAIRINGS:
        common = set(by_system[first_system]).intersection(by_system[second_system])
        templates = sorted({key[0] for key in common})
        if len(templates) != 10:
            raise ValueError(f"{category} pairing does not cover all ten templates")
        for template_id in templates:
            candidates = sorted(key for key in common if key[0] == template_id)
            themes: dict[str, list[tuple[Any, ...]]] = {}
            for key in candidates:
                themes.setdefault(str(key[1]), []).append(key)
            if len(themes) < 3:
                raise ValueError(f"{template_id} has fewer than three common themes")
            selected_themes = rng.sample(sorted(themes), 3)
            selected = [rng.choice(themes[theme]) for theme in selected_themes]
            for key in selected:
                first = by_system[first_system][key]
                second = by_system[second_system][key]
                pair_id = f"{category}-{len(public_rows) + 1:03d}"
                swapped = bool(rng.getrandbits(1))
                left, right = (second, first) if swapped else (first, second)
                public_rows.append(
                    {
                        "pair_id": pair_id,
                        "comparison": category,
                        "cipai": template_id.split(":", 1)[0],
                        "theme": first.theme,
                        "left_text": left.text,
                        "right_text": right.text,
                        "fluency": "",
                        "coherence": "",
                        "imagery_poetic_quality": "",
                        "overall_preference": "",
                        "notes": "",
                    }
                )
                private_rows.append(
                    {
                        "pair_id": pair_id,
                        "left_system": left.system,
                        "right_system": right.system,
                        "training_seed": first.training_seed,
                        "decoding_seed": first.decoding_seed,
                    }
                )
    if len(public_rows) != 60:
        raise AssertionError(f"expected 60 pairs, built {len(public_rows)}")
    form_path.parent.mkdir(parents=True, exist_ok=True)
    with form_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "instructions": "Responses must be left, right, or tie.",
                "pairs": private_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return form_path, key_path
