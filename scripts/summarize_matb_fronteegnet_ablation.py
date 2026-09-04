#!/usr/bin/env python3
"""Summarize FrontEEGNet MATB architecture ablations against the full model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from run_matb_fronteegnet_ablation_subject import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--full-reference-directory", required=True)
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


def load_reports(directory: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(directory.glob("sub-*.json"))
    if len(paths) != 26:
        raise ValueError(f"expected 26 reports in {directory}, found {len(paths)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = {str(report["target_subject"]): report for report in reports}
    if len(result) != len(reports):
        raise ValueError(f"duplicate target subject in {directory}")
    return result


def summarize_variant(
    reports: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    subjects = sorted(full)
    if set(reports) != set(subjects):
        raise ValueError("ablation and full-model subject sets differ")
    values = np.asarray(
        [reports[subject]["target_test"]["balanced_accuracy"] for subject in subjects]
    )
    full_values = np.asarray(
        [full[subject]["target_test"]["balanced_accuracy"] for subject in subjects]
    )
    macro_f1 = np.asarray(
        [reports[subject]["target_test"]["macro_f1"] for subject in subjects]
    )
    confusion = np.sum(
        [
            np.asarray(reports[subject]["target_test"]["confusion_matrix"])
            for subject in subjects
        ],
        axis=0,
    )
    parameter_counts = {
        int(report["source_training"]["parameters"]) for report in reports.values()
    }
    interventions = {str(report["intervention"]) for report in reports.values()}
    if len(parameter_counts) != 1 or len(interventions) != 1:
        raise ValueError("inconsistent ablation metadata")
    return {
        "intervention": interventions.pop(),
        "parameters": parameter_counts.pop(),
        "balanced_accuracy": distribution(values, seed),
        "macro_f1": distribution(macro_f1, seed + 1),
        "paired_difference_from_full": paired(values - full_values, seed + 2),
        "pooled_confusion_matrix": confusion.tolist(),
        "pooled_class_recall": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        "per_subject_balanced_accuracy": {
            subject: float(value) for subject, value in zip(subjects, values, strict=True)
        },
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_directory).expanduser().resolve()
    full = load_reports(Path(args.full_reference_directory).expanduser().resolve())
    full_values = np.asarray(
        [full[subject]["target_test"]["balanced_accuracy"] for subject in sorted(full)]
    )
    full_parameters = {
        int(report["source_training"]["parameters"]) for report in full.values()
    }
    if len(full_parameters) != 1:
        raise ValueError("inconsistent full-model parameter count")
    variants = {
        variant: summarize_variant(
            load_reports(input_root / variant / "subjects"),
            full,
            args.seed + 100 * index,
        )
        for index, variant in enumerate(VARIANTS)
    }
    summary = {
        "schema_version": 1,
        "dataset": "COG-BCI MATB",
        "protocol": "26-fold participant-held-out and cross-session source-only",
        "full_model": {
            "parameters": full_parameters.pop(),
            "balanced_accuracy": distribution(full_values, args.seed),
        },
        "variants": variants,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    full_result = summary["full_model"]["balanced_accuracy"]
    lines = [
        "# FrontEEGNet architecture ablation on COG-BCI MATB",
        "",
        "All variants use the same 26 strict LOSO folds, source sessions, validation "
        "sessions, optimizer, stopping rule and seed as the full model.",
        "",
        "| Variant | Parameters | Balanced accuracy | Difference from full | 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| Full FrontEEGNet | {summary['full_model']['parameters']:,} | "
        f"{100 * full_result['mean']:.2f}% ± {100 * full_result['std']:.2f}% | 0.00 pp | — |",
    ]
    for variant in VARIANTS:
        result = variants[variant]
        accuracy = result["balanced_accuracy"]
        difference = result["paired_difference_from_full"]
        interval = difference["bootstrap_95_ci_mean"]
        lines.append(
            f"| {variant} | {result['parameters']:,} | "
            f"{100 * accuracy['mean']:.2f}% ± {100 * accuracy['std']:.2f}% | "
            f"{100 * difference['mean']:+.2f} pp | "
            f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "The fixed projection tests learned cross-channel weighting against predefined "
            "common/difference components. Single-channel variants test whether both frontal "
            "sensors contribute beyond model capacity alone.",
            "",
        ]
    )
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output_json), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
