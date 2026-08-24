#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from poetry_lm.analysis import compare_statistics, file_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generated poems with the edited training corpus"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--generated", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = file_statistics(args.reference)
    payload: dict[str, object] = {
        "reference": {"path": str(args.reference), "statistics": reference},
        "generated": [],
    }
    generated_rows: list[dict[str, object]] = []
    for path in args.generated:
        statistics = file_statistics(path)
        generated_rows.append(
            {
                "path": str(path),
                "statistics": statistics,
                "comparison_to_reference": compare_statistics(reference, statistics),
            }
        )
    payload["generated"] = generated_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"analysed generated files: {len(generated_rows)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
