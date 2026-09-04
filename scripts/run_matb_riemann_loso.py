#!/usr/bin/env python3
"""Run 26-subject raw-covariance baselines under the strict MATB LOSO protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyriemann.classification import MDM
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from fronteeg_transfer.riemann_transfer import (
    CovarianceDomain,
    align_to_reference,
    estimate_covariances,
    fit_source_classifier,
    predict_target,
)

METHODS = ("plain_tangent", "plain_mdm", "ea_s1_tangent", "ra_s1_tangent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def discover_subjects(root: Path) -> list[str]:
    suffix = "_ses-S1_matb_raw.npz"
    return sorted(path.name.removesuffix(suffix) for path in root.glob(f"*{suffix}"))


def load_session(root: Path, subject: str, session: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{subject}_ses-{session}_matb_raw.npz"
    with np.load(path, allow_pickle=False) as values:
        eeg = np.asarray(values["eeg"], dtype=np.float32)
        labels = np.asarray(values["label"], dtype=np.int64)
    return estimate_covariances(eeg), labels


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def summarize(subjects: list[dict[str, object]], seed: int) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in METHODS:
        method_summary = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray(
                [subject["methods"][method][metric] for subject in subjects],
                dtype=np.float64,
            )
            method_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "bootstrap_95_ci_mean": bootstrap_ci(values, seed),
            }
        matrix = np.sum(
            [np.asarray(subject["methods"][method]["confusion_matrix"]) for subject in subjects],
            axis=0,
        )
        method_summary["pooled_confusion_matrix"] = matrix.tolist()
        method_summary["pooled_class_recall"] = (
            np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
        ).tolist()
        summary[method] = method_summary
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.raw_root).expanduser().resolve()
    subjects = discover_subjects(root)
    if len(subjects) != 26:
        raise ValueError(f"expected 26 subjects, found {len(subjects)}")
    loaded = {
        subject: {
            session: load_session(root, subject, session)
            for session in ("S1", "S2", "S3")
        }
        for subject in subjects
    }
    aligned_source = {mean: {} for mean in ("euclidean", "riemann")}
    for mean in aligned_source:
        for subject in subjects:
            subject_covariances = np.concatenate(
                [loaded[subject][session][0] for session in ("S1", "S2")]
            )
            aligned_source[mean][subject] = align_to_reference(
                subject_covariances, subject_covariances, mean=mean
            )
    results = []
    for target_subject in subjects:
        source_subjects = [subject for subject in subjects if subject != target_subject]
        source_covariances = np.concatenate(
            [loaded[subject][session][0] for subject in source_subjects for session in ("S1", "S2")]
        )
        source_labels = np.concatenate(
            [loaded[subject][session][1] for subject in source_subjects for session in ("S1", "S2")]
        )
        source_groups = np.concatenate(
            [
                np.full(len(loaded[subject][session][1]), subject, dtype=object)
                for subject in source_subjects
                for session in ("S1", "S2")
            ]
        )
        target_reference = loaded[target_subject]["S1"][0]
        target_test, target_labels = loaded[target_subject]["S3"]

        plain = fit_source_classifier(
            CovarianceDomain(source_covariances, source_labels, source_groups),
            recenter=False,
        )
        plain_predictions = predict_target(
            plain, CovarianceDomain(target_test), recenter=False
        ).argmax(axis=1)
        mdm = MDM(metric="riemann").fit(source_covariances, source_labels)
        methods = {
            "plain_tangent": metrics(target_labels, plain_predictions),
            "plain_mdm": metrics(target_labels, mdm.predict(target_test)),
        }
        for name, mean in (("ea_s1_tangent", "euclidean"), ("ra_s1_tangent", "riemann")):
            source_aligned = np.concatenate(
                [aligned_source[mean][subject] for subject in source_subjects]
            )
            aligned_target = align_to_reference(target_test, target_reference, mean=mean)
            classifier = fit_source_classifier(
                CovarianceDomain(source_aligned, source_labels), recenter=False
            )
            predictions = predict_target(
                classifier, CovarianceDomain(aligned_target), recenter=False
            ).argmax(axis=1)
            methods[name] = metrics(target_labels, predictions)
        row = {"subject": target_subject, "methods": methods}
        results.append(row)
        print(
            f"target={target_subject} "
            + " ".join(
                f"{method}={methods[method]['balanced_accuracy']:.4f}" for method in METHODS
            ),
            flush=True,
        )

    report = {
        "schema_version": 1,
        "experiment": "26-subject strict LOSO raw covariance baselines",
        "protocol": {
            "source_train": "25 subjects S1+S2 labels",
            "target_alignment_reference": "held-out target S1 signal without labels",
            "target_test": "held-out target S3 labels used only for final scoring",
            "target_S2_used": False,
            "window": "Fp1/Fp2, 1-40Hz, 250Hz, 4s, 50% overlap",
        },
        "methods": list(METHODS),
        "subjects": results,
        "summary": summarize(results, args.seed),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
