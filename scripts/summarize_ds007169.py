#!/usr/bin/env python3
"""Summarize the 18-fold participant-disjoint ds007169 experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def distribution(values: list[float], seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "bootstrap_95_ci_mean": bootstrap_ci(array, seed),
    }


def paired(values: list[float], seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        **distribution(values, seed),
        "wins": int(np.sum(array > 1e-12)),
        "ties": int(np.sum(np.abs(array) <= 1e-12)),
        "losses": int(np.sum(array < -1e-12)),
    }


def pooled_confusion(matrices: list[list[list[int]]]) -> dict[str, object]:
    matrix = np.sum([np.asarray(value, dtype=np.int64) for value in matrices], axis=0)
    return {
        "matrix": matrix.tolist(),
        "class_recall": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
    }


def format_distribution(result: dict[str, object]) -> str:
    return f"{100 * float(result['mean']):.2f}% ± {100 * float(result['std']):.2f}%"


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.input_directory).expanduser().resolve().glob("sub-*.json"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 subject files, found {len(paths)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    budgets = reports[0]["budgets_per_class"]
    repeats = reports[0]["repeats"]
    methods = ("scratch", "linear", "full")
    per_subject = {}
    for report in reports:
        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for record in report["records"]:
            grouped[(record["method"], record["budget_per_class"])].append(record)
        per_subject[report["target_subject"]] = {
            "source_only": report["source_only_target_test"],
            "methods": {
                method: {
                    str(budget): {
                        metric: float(
                            np.mean([item["test"][metric] for item in grouped[(method, budget)]])
                        )
                        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                    }
                    for budget in budgets
                }
                for method in methods
            },
            "target_test_per_class": report["target_test_per_class"],
        }
    subjects = sorted(per_subject)
    source_values = [
        float(per_subject[subject]["source_only"]["balanced_accuracy"]) for subject in subjects
    ]
    aggregate = {
        "source_only": {
            "balanced_accuracy": distribution(source_values, args.seed),
            "macro_f1": distribution(
                [float(per_subject[subject]["source_only"]["macro_f1"]) for subject in subjects],
                args.seed + 1,
            ),
            "paired_vs_chance": paired([value - 0.25 for value in source_values], args.seed + 2),
            "pooled_confusion": pooled_confusion(
                [report["source_only_target_test"]["confusion_matrix"] for report in reports]
            ),
        },
        "methods": {},
    }
    for method_index, method in enumerate(methods):
        aggregate["methods"][method] = {}
        for budget_index, budget in enumerate(budgets):
            values = [
                float(per_subject[subject]["methods"][method][str(budget)]["balanced_accuracy"])
                for subject in subjects
            ]
            matrices = []
            for report in reports:
                matrices.extend(
                    record["test"]["confusion_matrix"]
                    for record in report["records"]
                    if record["method"] == method and record["budget_per_class"] == budget
                )
            aggregate["methods"][method][str(budget)] = {
                "balanced_accuracy": distribution(
                    values, args.seed + 10 + method_index * 10 + budget_index
                ),
                "paired_vs_source_only": paired(
                    [values[index] - source_values[index] for index in range(len(subjects))],
                    args.seed + 100 + method_index * 10 + budget_index,
                ),
                "pooled_confusion_across_repeats": pooled_confusion(matrices),
            }
    for budget_index, budget in enumerate(budgets):
        scratch = [
            float(per_subject[subject]["methods"]["scratch"][str(budget)]["balanced_accuracy"])
            for subject in subjects
        ]
        for method_index, method in enumerate(("linear", "full")):
            values = [
                float(per_subject[subject]["methods"][method][str(budget)]["balanced_accuracy"])
                for subject in subjects
            ]
            aggregate["methods"][method][str(budget)]["paired_vs_scratch"] = paired(
                [values[index] - scratch[index] for index in range(len(subjects))],
                args.seed + 200 + method_index * 10 + budget_index,
            )
    summary = {
        "schema_version": 1,
        "dataset": "OpenNeuro ds007169",
        "subjects": 18,
        "classes": ["1-back", "2-back", "3-back", "4-back"],
        "chance_balanced_accuracy": 0.25,
        "budgets_per_class": budgets,
        "repeats": repeats,
        "per_subject": per_subject,
        "aggregate": aggregate,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# OpenNeuro ds007169 external validation",
        "",
        "The same Fp1/Fp2 four-band PSD MLP is rerun as a four-class model in 18 "
        "participant-held-out folds. Each source recording is temporally split for "
        "training and early stopping. For the target participant, nested non-overlapping "
        "calibration windows precede a later test segment; target test data are never "
        "used for training or selection.",
        "",
        "| Labels/class | Source-only | Scratch | Linear head | Full fine-tuning |",
        "|---:|---:|---:|---:|---:|",
    ]
    source = aggregate["source_only"]["balanced_accuracy"]
    for budget in budgets:
        scratch = aggregate["methods"]["scratch"][str(budget)]["balanced_accuracy"]
        linear = aggregate["methods"]["linear"][str(budget)]["balanced_accuracy"]
        full = aggregate["methods"]["full"][str(budget)]["balanced_accuracy"]
        lines.append(
            f"| {budget} | {format_distribution(source)} | "
            f"{format_distribution(scratch)} | {format_distribution(linear)} | "
            f"{format_distribution(full)} |"
        )
    chance = aggregate["source_only"]["paired_vs_chance"]
    lines.extend(
        [
            "",
            "Source-only exceeded the 25% four-class chance level by "
            f"{100 * float(chance['mean']):.2f} percentage points (participant-level "
            f"bootstrap 95% CI {100 * float(chance['bootstrap_95_ci_mean'][0]):.2f} to "
            f"{100 * float(chance['bootstrap_95_ci_mean'][1]):.2f}).",
            "",
            "No calibration result should be described as superior to source-only or "
            "scratch unless its participant-paired interval excludes zero. The external "
            "dataset has one recording session, so this experiment tests cross-participant "
            "transportability but not independent cross-session generalization.",
        ]
    )
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_json), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
