#!/usr/bin/env python3
"""Validate and summarize all controlled cross-subject low-shot result files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ("scratch", "linear", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--expected-subjects", type=int, default=26)
    return parser.parse_args()


def validate(report: dict, path: Path) -> None:
    budgets = report.get("budgets_per_class")
    repeats = report.get("repeats")
    records = report.get("records", [])
    if not budgets or not repeats:
        raise ValueError(f"missing budgets/repeats in {path}")
    if len(report.get("source_subjects", [])) != 25:
        raise ValueError(f"expected 25 source subjects in {path}")
    if len(records) != len(budgets) * repeats * len(METHODS):
        raise ValueError(f"incomplete records in {path}: {len(records)}")
    expected = {
        (repeat, budget, method)
        for repeat in range(repeats)
        for budget in budgets
        for method in METHODS
    }
    actual = {
        (record["repeat"], record["budget_per_class"], record["method"]) for record in records
    }
    if actual != expected:
        raise ValueError(f"record grid mismatch in {path}")
    if "controlled_randomness" not in report:
        raise ValueError(f"uncontrolled experiment output in {path}")
    if report.get("schema_version", 1) >= 2:
        if "training_seconds" not in report.get("source_training", {}):
            raise ValueError(f"missing source timing in {path}")
        if any("training" not in record for record in records):
            raise ValueError(f"missing target timing in {path}")


def subject_metrics(report: dict) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in report["records"]:
        grouped[(record["method"], record["budget_per_class"])].append(record)
    methods = {}
    for key, records in grouped.items():
        method, budget = key
        methods.setdefault(method, {})[str(budget)] = {
            metric: {
                "mean": float(np.mean([record["test"][metric] for record in records])),
                "std": float(np.std([record["test"][metric] for record in records])),
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        if all("training" in record for record in records):
            methods[method][str(budget)]["training"] = {
                "training_seconds": {
                    "mean": float(
                        np.mean([record["training"]["training_seconds"] for record in records])
                    ),
                    "std": float(
                        np.std([record["training"]["training_seconds"] for record in records])
                    ),
                },
                "trainable_parameters": records[0]["training"]["trainable_parameters"],
                "optimizer_steps": records[0]["training"]["optimizer_steps"],
                "minimum_labeled_signal_seconds": records[0][
                    "minimum_labeled_signal_seconds"
                ],
            }
    return {
        "subject": report["target_subject"],
        "source_only": report["source_only_target_test"],
        "source_training": report.get("source_training", {}),
        "source_only_inference": report.get("source_only_inference", {}),
        "methods": methods,
    }


def bootstrap_ci(values: np.ndarray, seed: int = 12345) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "bootstrap_95_ci_mean": bootstrap_ci(array),
    }


def summarize(subjects: list[dict], budgets: list[int]) -> dict:
    aggregate = {
        "source_only": {
            metric: distribution([subject["source_only"][metric] for subject in subjects])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        },
        "methods": {},
        "paired_vs_scratch": {},
        "paired_vs_source_only": {},
        "paired_transfer_vs_scratch_grid": {},
    }
    if all("training_seconds" in subject["source_training"] for subject in subjects):
        aggregate["source_training"] = {
            "training_seconds": distribution(
                [subject["source_training"]["training_seconds"] for subject in subjects]
            ),
            "best_epoch": distribution(
                [subject["source_training"]["best_epoch"] for subject in subjects]
            ),
            "trainable_parameters": subjects[0]["source_training"]["trainable_parameters"],
        }
    if all("test_windows_per_second" in subject["source_only_inference"] for subject in subjects):
        aggregate["source_only_inference"] = {
            "test_windows_per_second": distribution(
                [
                    subject["source_only_inference"]["test_windows_per_second"]
                    for subject in subjects
                ]
            )
        }
    for method in METHODS:
        aggregate["methods"][method] = {}
        for budget in budgets:
            aggregate["methods"][method][str(budget)] = {
                metric: distribution(
                    [
                        subject["methods"][method][str(budget)][metric]["mean"]
                        for subject in subjects
                    ]
                )
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }
            if all(
                "training" in subject["methods"][method][str(budget)] for subject in subjects
            ):
                aggregate["methods"][method][str(budget)]["training"] = {
                    "training_seconds": distribution(
                        [
                            subject["methods"][method][str(budget)]["training"][
                                "training_seconds"
                            ]["mean"]
                            for subject in subjects
                        ]
                    ),
                    "trainable_parameters": subjects[0]["methods"][method][str(budget)][
                        "training"
                    ]["trainable_parameters"],
                    "optimizer_steps": subjects[0]["methods"][method][str(budget)][
                        "training"
                    ]["optimizer_steps"],
                    "minimum_labeled_signal_seconds": subjects[0]["methods"][method][
                        str(budget)
                    ]["training"]["minimum_labeled_signal_seconds"],
                }
    for method in ("linear", "full"):
        aggregate["paired_vs_scratch"][method] = {}
        for budget in budgets:
            gains = np.asarray(
                [
                    subject["methods"][method][str(budget)]["accuracy"]["mean"]
                    - subject["methods"]["scratch"][str(budget)]["accuracy"]["mean"]
                    for subject in subjects
                ]
            )
            aggregate["paired_vs_scratch"][method][str(budget)] = {
                **distribution(gains.tolist()),
                "wins": int((gains > 1e-12).sum()),
                "ties": int((np.abs(gains) <= 1e-12).sum()),
                "losses": int((gains < -1e-12).sum()),
                "win_rate": float((gains > 1e-12).mean()),
                "negative_transfer_rate": float((gains < -1e-12).mean()),
            }
            source_gains = np.asarray(
                [
                    subject["methods"][method][str(budget)]["accuracy"]["mean"]
                    - subject["source_only"]["accuracy"]
                    for subject in subjects
                ]
            )
            aggregate["paired_vs_source_only"].setdefault(method, {})[str(budget)] = {
                **distribution(source_gains.tolist()),
                "wins": int((source_gains > 1e-12).sum()),
                "ties": int((np.abs(source_gains) <= 1e-12).sum()),
                "losses": int((source_gains < -1e-12).sum()),
            }
            scratch_grid = {}
            for scratch_budget in budgets:
                cross_budget_gains = np.asarray(
                    [
                        subject["methods"][method][str(budget)]["accuracy"]["mean"]
                        - subject["methods"]["scratch"][str(scratch_budget)]["accuracy"]["mean"]
                        for subject in subjects
                    ]
                )
                scratch_grid[str(scratch_budget)] = {
                    **distribution(cross_budget_gains.tolist()),
                    "transfer_wins": int((cross_budget_gains > 1e-12).sum()),
                    "ties": int((np.abs(cross_budget_gains) <= 1e-12).sum()),
                    "transfer_losses": int((cross_budget_gains < -1e-12).sum()),
                }
            aggregate["paired_transfer_vs_scratch_grid"].setdefault(method, {})[
                str(budget)
            ] = scratch_grid

    scratch_population = {
        budget: aggregate["methods"]["scratch"][str(budget)]["accuracy"]["mean"]
        for budget in budgets
    }
    benchmarks = {"source_only": aggregate["source_only"]["accuracy"]["mean"]}
    for method in ("linear", "full"):
        for budget in budgets:
            benchmarks[f"{method}@{budget}"] = aggregate["methods"][method][str(budget)][
                "accuracy"
            ]["mean"]
    aggregate["population_label_equivalence"] = {}
    for name, accuracy in benchmarks.items():
        matching = [budget for budget in budgets if scratch_population[budget] >= accuracy]
        aggregate["population_label_equivalence"][name] = {
            "benchmark_accuracy": accuracy,
            "minimum_observed_scratch_budget_per_class": matching[0] if matching else None,
            "censored_above_budget_per_class": None if matching else max(budgets),
        }
    return aggregate


def write_csv(path: Path, subjects: list[dict], budgets: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject",
        "method",
        "budget_per_class",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "training_seconds_mean",
        "training_seconds_std",
        "trainable_parameters",
        "minimum_labeled_signal_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subject in subjects:
            source = subject["source_only"]
            writer.writerow(
                {
                    "subject": subject["subject"],
                    "method": "source_only",
                    "budget_per_class": 0,
                    "accuracy_mean": source["accuracy"],
                    "accuracy_std": 0.0,
                    "balanced_accuracy_mean": source["balanced_accuracy"],
                    "balanced_accuracy_std": 0.0,
                    "macro_f1_mean": source["macro_f1"],
                    "macro_f1_std": 0.0,
                    "training_seconds_mean": subject["source_training"].get(
                        "training_seconds", ""
                    ),
                    "training_seconds_std": 0.0,
                    "trainable_parameters": subject["source_training"].get(
                        "trainable_parameters", ""
                    ),
                    "minimum_labeled_signal_seconds": 0.0,
                }
            )
            for budget in budgets:
                for method in METHODS:
                    metrics = subject["methods"][method][str(budget)]
                    training = metrics.get("training", {})
                    training_seconds = training.get("training_seconds", {})
                    writer.writerow(
                        {
                            "subject": subject["subject"],
                            "method": method,
                            "budget_per_class": budget,
                            "accuracy_mean": metrics["accuracy"]["mean"],
                            "accuracy_std": metrics["accuracy"]["std"],
                            "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
                            "balanced_accuracy_std": metrics["balanced_accuracy"]["std"],
                            "macro_f1_mean": metrics["macro_f1"]["mean"],
                            "macro_f1_std": metrics["macro_f1"]["std"],
                            "training_seconds_mean": training_seconds.get("mean", ""),
                            "training_seconds_std": training_seconds.get("std", ""),
                            "trainable_parameters": training.get("trainable_parameters", ""),
                            "minimum_labeled_signal_seconds": training.get(
                                "minimum_labeled_signal_seconds", 3 * budget * 4.0
                            ),
                        }
                    )


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(path: Path, subjects: list[dict], budgets: list[int], aggregate: dict) -> None:
    lines = [
        "# Complete cross-subject low-shot results",
        "",
        "## Protocol",
        "",
        f"- Outer LOSO targets: {len(subjects)} subjects.",
        "- Each target uses the other 25 subjects as sources.",
        "- Target S1 supplies unlabeled normalization and nested labeled calibration subsets.",
        "- Target S3 is the independent test session; target S2 is unused.",
        "- Each low-shot cell is the mean of three controlled nested draws.",
        "",
        "## Population summary",
        "",
        f"Source-only accuracy: {percentage(aggregate['source_only']['accuracy']['mean'])} ± "
        f"{percentage(aggregate['source_only']['accuracy']['std'])}.",
        "",
        "| Labels/class | Scratch accuracy | Linear accuracy | Full accuracy | "
        "Linear−Scratch | Linear wins | Full−Scratch | Full wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        scratch = aggregate["methods"]["scratch"][str(budget)]["accuracy"]
        linear = aggregate["methods"]["linear"][str(budget)]["accuracy"]
        full = aggregate["methods"]["full"][str(budget)]["accuracy"]
        linear_gain = aggregate["paired_vs_scratch"]["linear"][str(budget)]
        full_gain = aggregate["paired_vs_scratch"]["full"][str(budget)]
        lines.append(
            f"| {budget} | {percentage(scratch['mean'])} ± {percentage(scratch['std'])} "
            f"| {percentage(linear['mean'])} ± {percentage(linear['std'])} "
            f"| {percentage(full['mean'])} ± {percentage(full['std'])} "
            f"| {percentage(linear_gain['mean'])} | {linear_gain['wins']}/{len(subjects)} "
            f"| {percentage(full_gain['mean'])} | {full_gain['wins']}/{len(subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Per-subject accuracy",
            "",
            "`S/L/F` denotes Scratch / Linear / Full calibration accuracy.",
            "",
            "| Subject | Source-only | "
            + " | ".join(f"{budget} labels/class S/L/F" for budget in budgets)
            + " |",
            "|---|---:|" + "---:|" * len(budgets),
        ]
    )
    for subject in subjects:
        cells = []
        for budget in budgets:
            values = [
                percentage(subject["methods"][method][str(budget)]["accuracy"]["mean"])
                for method in METHODS
            ]
            cells.append(" / ".join(values))
        lines.append(
            f"| {subject['subject']} | {percentage(subject['source_only']['accuracy'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Label-efficiency interpretation",
            "",
            "The table below reports the first observed Scratch budget whose population "
            "mean accuracy reaches each benchmark. A censored result means Scratch did "
            "not catch the benchmark by the largest tested budget; it is not an exact "
            "equivalence estimate.",
            "",
            "| Benchmark | Accuracy | First Scratch budget/class |",
            "|---|---:|---:|",
        ]
    )
    for name, result in aggregate["population_label_equivalence"].items():
        matched = result["minimum_observed_scratch_budget_per_class"]
        display = str(matched) if matched is not None else f">{max(budgets)} (censored)"
        lines.append(f"| {name} | {percentage(result['benchmark_accuracy'])} | {display} |")
    if "source_training" in aggregate:
        lines.extend(
            [
                "",
                "## Computation and acquisition cost",
                "",
                f"Source training wall time: "
                f"{aggregate['source_training']['training_seconds']['mean']:.2f} ± "
                f"{aggregate['source_training']['training_seconds']['std']:.2f} seconds "
                "per LOSO model. This is a one-time offline cost, not per-person "
                "calibration time.",
                "",
                "| Labels/class | Minimum signal | Scratch train | Linear train | Full train |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for budget in budgets:
            cells = []
            for method in METHODS:
                timing = aggregate["methods"][method][str(budget)]["training"][
                    "training_seconds"
                ]
                cells.append(f"{timing['mean']:.3f} ± {timing['std']:.3f} s")
            lines.append(
                f"| {budget} | {3 * budget * 4.0 / 60.0:.2f} min | "
                + " | ".join(cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are public-dataset offline results. They estimate cross-subject and "
            "cross-session label efficiency for Fp1/Fp2 MATB features; they are not yet "
            "real-device online results.",
            "The label-equivalence table in this file is conditional on the fixed "
            "50-epoch Scratch schedule. Use the separate Scratch convergence control "
            "before making label-saving claims.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_directory).expanduser().resolve()
    paths = sorted(input_root.glob("sub-*_sources25_controlled.json"))
    if len(paths) != args.expected_subjects:
        raise ValueError(f"expected {args.expected_subjects} result files, found {len(paths)}")
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        validate(report, path)
        reports.append(report)
    targets = [report["target_subject"] for report in reports]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate target-subject output")
    budgets = reports[0]["budgets_per_class"]
    if any(report["budgets_per_class"] != budgets for report in reports):
        raise ValueError("budget mismatch across subjects")
    subjects = sorted(
        (subject_metrics(report) for report in reports), key=lambda item: item["subject"]
    )
    aggregate = summarize(subjects, budgets)
    summary = {
        "subjects": len(subjects),
        "budgets_per_class": budgets,
        "methods": list(METHODS),
        "per_subject": subjects,
        "aggregate": aggregate,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv).expanduser().resolve(), subjects, budgets)
    write_markdown(Path(args.output_markdown).expanduser().resolve(), subjects, budgets, aggregate)
    print(
        json.dumps(
            {
                "subjects": len(subjects),
                "source_only_accuracy": aggregate["source_only"]["accuracy"],
                "outputs": {
                    "json": str(output_json),
                    "csv": str(Path(args.output_csv).expanduser().resolve()),
                    "markdown": str(Path(args.output_markdown).expanduser().resolve()),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
