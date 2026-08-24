#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = {
    "Real Song-ci baseline": "real_songci_baseline.csv",
    "6L / BS32 / Dim256 / H4": "6l_bs32_d256_h4.csv",
    "6L / BS64 / Dim256 / H4": "6l_bs64_d256_h4.csv",
    "6L / BS32 / Dim256 / H8": "6l_bs32_d256_h8.csv",
    "6L / BS32 / Dim384 / H4": "6l_bs32_d384_h4.csv",
    "6L / BS32 / Dim384 / H8": "6l_bs32_d384_h8.csv",
}


def read_validation_points(path: Path) -> list[tuple[int, float]]:
    points: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("avg_val_loss", "").strip()
            if value:
                points[int(row["step"])] = float(value)
    if not points:
        raise ValueError(f"no validation values found in {path}")
    return sorted(points.items())


def selected_points(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    preferred = {5000, 7000, 10000, 11000, 15000, 20000}
    chosen = [point for point in points if point[0] in preferred]
    if points[0] not in chosen:
        chosen.insert(0, points[0])
    if points[-1] not in chosen:
        chosen.append(points[-1])
    return sorted(set(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild final report figures")
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_points = {
        label: read_validation_points(args.metrics_dir / filename)
        for label, filename in EXPERIMENTS.items()
    }
    summary: dict[str, dict[str, float | int]] = {}
    for label, points in all_points.items():
        best_step, best_loss = min(points, key=lambda point: point[1])
        final_step, final_loss = points[-1]
        summary[label] = {
            "best_step": best_step,
            "best_validation_loss": best_loss,
            "final_step": final_step,
            "final_validation_loss": final_loss,
        }

    labels = list(summary)
    x_positions = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(12, 6))
    plt.bar(
        [position - width / 2 for position in x_positions],
        [float(summary[label]["best_validation_loss"]) for label in labels],
        width,
        label="Best validation loss",
    )
    plt.bar(
        [position + width / 2 for position in x_positions],
        [float(summary[label]["final_validation_loss"]) for label in labels],
        width,
        label="Final validation loss",
    )
    plt.xticks(x_positions, labels, rotation=25, ha="right")
    plt.ylabel("Validation loss")
    plt.title("Best vs Final Validation Loss Across Main Experiment Lines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "best_vs_final_validation_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    for label, points in all_points.items():
        sparse = selected_points(points)
        plt.plot(
            [point[0] for point in sparse],
            [point[1] for point in sparse],
            marker="o",
            label=label,
        )
    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.title("Sparse Validation Trajectories Across Main Experiment Lines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "sparse_validation_trajectories.png", dpi=300)
    plt.close()

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
