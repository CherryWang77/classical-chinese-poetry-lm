#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from poetry_lm.analysis import file_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute structural corpus statistics")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    statistics = file_statistics(args.input)
    rendered = json.dumps(statistics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
