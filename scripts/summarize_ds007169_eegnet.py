#!/usr/bin/env python3
"""Summarize ds007169 FrontEEGNet folds and the matched PSD-MLP baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--psd-reference", required=True)
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
    paths = sorted(Path(args.input_directory).expanduser().resolve().glob("sub-*.json"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 subject reports, found {len(paths)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_subject = {str(report["target_subject"]): report for report in reports}
    if len(by_subject) != len(reports):
        raise ValueError("duplicate target subjects")
    subjects = sorted(by_subject)
    bacc = np.asarray(
        [by_subject[subject]["target_test"]["balanced_accuracy"] for subject in subjects]
    )
    macro_f1 = np.asarray(
        [by_subject[subject]["target_test"]["macro_f1"] for subject in subjects]
    )
    confusion = np.sum(
        [
            np.asarray(by_subject[subject]["target_test"]["confusion_matrix"])
            for subject in subjects
        ],
        axis=0,
    )

    psd = json.loads(
        Path(args.psd_reference).expanduser().resolve().read_text(encoding="utf-8")
    )
    if set(psd["per_subject"]) != set(subjects):
        raise ValueError("FrontEEGNet and PSD-MLP subject sets differ")
    psd_bacc = np.asarray(
        [
            psd["per_subject"][subject]["source_only"]["balanced_accuracy"]
            for subject in subjects
        ]
    )
    difference = bacc - psd_bacc
    parameters = {int(report["source_training"]["parameters"]) for report in reports}
    if len(parameters) != 1:
        raise ValueError(f"inconsistent parameter counts: {parameters}")
    summary = {
        "schema_version": 1,
        "dataset": "OpenNeuro ds007169",
        "method": "FrontEEGNet source-only",
        "subjects": 18,
        "classes": ["1-back", "2-back", "3-back", "4-back"],
        "chance_balanced_accuracy": 0.25,
        "parameters": parameters.pop(),
        "aggregate": {
            "balanced_accuracy": distribution(bacc, args.seed),
            "macro_f1": distribution(macro_f1, args.seed + 1),
            "paired_vs_chance": paired(bacc - 0.25, args.seed + 2),
            "paired_vs_psd_mlp": paired(difference, args.seed + 3),
            "pooled_confusion_matrix": confusion.tolist(),
            "pooled_class_recall": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        },
        "per_subject": {
            subject: {
                "front_eegnet": by_subject[subject]["target_test"],
                "psd_mlp_balanced_accuracy": float(
                    psd["per_subject"][subject]["source_only"]["balanced_accuracy"]
                ),
            }
            for subject in subjects
        },
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    front = summary["aggregate"]["balanced_accuracy"]
    f1 = summary["aggregate"]["macro_f1"]
    comparison = summary["aggregate"]["paired_vs_psd_mlp"]
    lines = [
        "# FrontEEGNet external validation on OpenNeuro ds007169",
        "",
        "Eighteen participant-held-out folds use the same two-channel FrontEEGNet "
        "architecture after polyphase resampling from 200 to 250 Hz. The target prefix "
        "is excluded to match the existing PSD-MLP test segments. FrontEEGNet does not "
        "use this prefix; PSD-MLP uses only its unlabeled features for min-max normalization.",
        "",
        "| Method | Balanced accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| FrontEEGNet | {100 * front['mean']:.2f}% ± {100 * front['std']:.2f}% | "
        f"{100 * f1['mean']:.2f}% ± {100 * f1['std']:.2f}% |",
        f"| PSD-MLP | {100 * psd_bacc.mean():.2f}% ± {100 * psd_bacc.std():.2f}% | — |",
        "",
        f"The paired FrontEEGNet minus PSD-MLP difference was "
        f"{100 * comparison['mean']:.2f} percentage points "
        f"(bootstrap 95% CI {100 * comparison['bootstrap_95_ci_mean'][0]:.2f} to "
        f"{100 * comparison['bootstrap_95_ci_mean'][1]:.2f}; "
        f"wins/ties/losses {comparison['wins']}/{comparison['ties']}/{comparison['losses']}).",
        "",
        "This is an exploratory evaluation on a second participant-disjoint benchmark; "
        "it neither transfers MATB weights nor establishes superiority over PSD.",
        "",
    ]
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output_json), "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
