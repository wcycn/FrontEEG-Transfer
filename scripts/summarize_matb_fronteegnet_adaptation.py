#!/usr/bin/env python3
"""Aggregate FrontEEGNet low-shot adaptation with participant-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("head", "partial", "full", "scratch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def distribution(values: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "bootstrap_95_ci_mean": bootstrap_ci(values, seed),
    }


def paired(values: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        **distribution(values, seed),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
    }


def main() -> None:
    args = parse_args()
    input_directory = Path(args.input_directory).expanduser().resolve()
    paths = sorted((input_directory / "subjects").glob("sub-*.json"))
    if len(paths) != 26:
        raise ValueError(f"expected 26 target reports, found {len(paths)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    subjects = [str(report["target_subject"]) for report in reports]
    if len(set(subjects)) != 26:
        raise ValueError("target reports contain duplicate subjects")
    budgets = reports[0]["budgets_per_class"]
    if any(report["budgets_per_class"] != budgets for report in reports):
        raise ValueError("inconsistent budgets across target reports")
    source = np.asarray(
        [report["source_only_target_test"]["balanced_accuracy"] for report in reports]
    )
    cells: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        cells[method] = {}
        for budget_index, budget in enumerate(budgets):
            participant_values = []
            trainable_counts = set()
            training_times = []
            for report in reports:
                selected = [
                    record
                    for record in report["records"]
                    if record["method"] == method
                    and record["budget_per_class"] == budget
                ]
                if len(selected) != int(report["repeats"]):
                    raise ValueError(
                        f"incomplete repeats for {report['target_subject']} {method}@{budget}"
                    )
                participant_values.append(
                    np.mean(
                        [
                            record["target_test"]["balanced_accuracy"]
                            for record in selected
                        ]
                    )
                )
                trainable_counts.update(
                    int(record["training"]["trainable_parameters"])
                    for record in selected
                )
                training_times.extend(
                    float(record["training"]["training_seconds"])
                    for record in selected
                )
            if len(trainable_counts) != 1:
                raise ValueError(f"inconsistent parameter count for {method}@{budget}")
            values = np.asarray(participant_values)
            cells[method][str(budget)] = {
                "trainable_parameters": trainable_counts.pop(),
                "balanced_accuracy": distribution(
                    values, args.seed + method_index * 100 + budget_index
                ),
                "paired_vs_source_only": paired(
                    values - source, args.seed + 1_000 + method_index * 100 + budget_index
                ),
                "mean_training_seconds_per_repeat": float(np.mean(training_times)),
                "per_subject_balanced_accuracy": dict(
                    zip(subjects, (float(value) for value in values), strict=True)
                ),
            }
    for budget_index, budget in enumerate(budgets):
        scratch_values = np.asarray(
            list(cells["scratch"][str(budget)]["per_subject_balanced_accuracy"].values())
        )
        for method_index, method in enumerate(("head", "partial", "full")):
            values = np.asarray(
                list(cells[method][str(budget)]["per_subject_balanced_accuracy"].values())
            )
            cells[method][str(budget)]["paired_vs_scratch"] = paired(
                values - scratch_values,
                args.seed + 2_000 + method_index * 100 + budget_index,
            )
    summary = {
        "schema_version": 1,
        "dataset": "COG-BCI MATB",
        "protocol": "26-fold participant-held-out; target S1 low-shot adaptation; target S3 test",
        "participants": subjects,
        "budgets_per_class": budgets,
        "source_only": distribution(source, args.seed),
        "methods": cells,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    source_result = summary["source_only"]
    lines = [
        "# FrontEEGNet nested low-shot target adaptation",
        "",
        "All conditions reuse the identical strict-LOSO source checkpoints and nested, "
        "balanced, non-overlapping target-S1 calibration windows. Target S3 is used only "
        "for final scoring. Repeats are averaged within participant before inference.",
        "",
        f"Source-only: {100 * source_result['mean']:.2f}% ± "
        f"{100 * source_result['std']:.2f}% balanced accuracy.",
        "",
        "| Method | Trainable parameters | Labels/class | Balanced accuracy | "
        "Difference from source-only | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "head": "Head-only",
        "partial": "Partial last-block",
        "full": "Full fine-tuning",
        "scratch": "Target-only scratch",
    }
    for method in METHODS:
        for budget in budgets:
            cell = cells[method][str(budget)]
            accuracy = cell["balanced_accuracy"]
            difference = cell["paired_vs_source_only"]
            interval = difference["bootstrap_95_ci_mean"]
            lines.append(
                f"| {labels[method]} | {cell['trainable_parameters']:,} | {budget} | "
                f"{100 * accuracy['mean']:.2f}% ± {100 * accuracy['std']:.2f}% | "
                f"{100 * difference['mean']:+.2f} pp | "
                f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] pp |"
            )
    lines.append("")
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output_json), "source_only": source_result}, indent=2))


if __name__ == "__main__":
    main()
