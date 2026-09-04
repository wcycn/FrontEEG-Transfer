#!/usr/bin/env python3
"""Evaluate trained LOSO EEGNet models under synthetic blink and motion corruption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_matb_eegnet_subject import (
    MODEL_CONFIG,
    evaluate,
    make_loader,
    normalize_windows,
)

from fronteeg_transfer.artifact_stress import add_blink_artifacts, add_motion_artifacts
from fronteeg_transfer.baselines import EEGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--checkpoint-directory", required=True)
    parser.add_argument("--reference-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def summarize(subjects: list[dict], conditions: list[str], seed: int) -> dict:
    result = {}
    clean = np.asarray(
        [subject["conditions"]["clean"]["balanced_accuracy"] for subject in subjects]
    )
    for index, condition in enumerate(conditions):
        values = np.asarray(
            [subject["conditions"][condition]["balanced_accuracy"] for subject in subjects]
        )
        difference = values - clean
        matrix = np.sum(
            [
                np.asarray(subject["conditions"][condition]["confusion_matrix"])
                for subject in subjects
            ],
            axis=0,
        )
        result[condition] = {
            "balanced_accuracy": {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "bootstrap_95_ci_mean": bootstrap_ci(values, seed + index),
            },
            "paired_difference_from_clean": {
                "mean": float(difference.mean()),
                "std": float(difference.std()),
                "bootstrap_95_ci_mean": bootstrap_ci(difference, seed + 100 + index),
                "wins": int(np.sum(difference > 1e-12)),
                "ties": int(np.sum(np.abs(difference) <= 1e-12)),
                "losses": int(np.sum(difference < -1e-12)),
            },
            "pooled_confusion_matrix": matrix.tolist(),
            "pooled_class_recall": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
        }
    return result


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_directory).expanduser().resolve()
    reference_root = Path(args.reference_directory).expanduser().resolve()
    subjects = sorted(path.stem for path in checkpoint_root.glob("sub-*.pt"))
    if len(subjects) != 26:
        raise ValueError(f"expected 26 checkpoints, found {len(subjects)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conditions = ["clean"]
    conditions.extend(f"blink_{amplitude}uv" for amplitude in (50, 100, 200))
    conditions.extend(f"motion_{amplitude}uv" for amplitude in (25, 50, 100))
    reports = []
    for subject_index, subject in enumerate(subjects):
        with np.load(raw_root / f"{subject}_ses-S3_matb_raw.npz", allow_pickle=False) as loaded:
            raw = loaded["eeg"].astype(np.float32)
            labels = loaded["label"].astype(np.int64)
        payload = torch.load(
            checkpoint_root / f"{subject}.pt", map_location="cpu", weights_only=True
        )
        model = EEGNet(MODEL_CONFIG, classes=3).to(device)
        model.load_state_dict(payload["model_state"])
        result = {}
        for condition in conditions:
            if condition == "clean":
                corrupted = raw
            elif condition.startswith("blink_"):
                amplitude = float(condition.removeprefix("blink_").removesuffix("uv"))
                corrupted = add_blink_artifacts(
                    raw,
                    sampling_rate=250.0,
                    amplitude_uv=amplitude,
                    seed=args.seed + subject_index * 100 + int(amplitude),
                )
            else:
                amplitude = float(condition.removeprefix("motion_").removesuffix("uv"))
                corrupted = add_motion_artifacts(
                    raw,
                    sampling_rate=250.0,
                    amplitude_uv=amplitude,
                    seed=args.seed + 50_000 + subject_index * 100 + int(amplitude),
                )
            loader = make_loader(
                (normalize_windows(corrupted), labels),
                batch_size=args.batch_size,
                shuffle=False,
            )
            result[condition] = evaluate(model, loader, device)
        reference = json.loads((reference_root / f"{subject}.json").read_text(encoding="utf-8"))[
            "target_test"
        ]["balanced_accuracy"]
        clean_error = abs(float(result["clean"]["balanced_accuracy"]) - float(reference))
        if clean_error > 1e-8:
            raise ValueError(f"clean result mismatch for {subject}: {clean_error}")
        reports.append({"subject": subject, "conditions": result})
        print(f"complete subject={subject}", flush=True)
    report = {
        "schema_version": 1,
        "experiment": "synthetic artifact stress test on strict-LOSO EEGNet",
        "boundary": (
            "The perturbations are signal-level simulations, not recorded or annotated "
            "human artifacts. They quantify model sensitivity but do not estimate "
            "real-device rates."
        ),
        "artifact_models": {
            "blink": "common frontal Gaussian pulse, Fp2 amplitude 0.85*Fp1",
            "motion": "channel-asymmetric electrode step with exponential recovery",
        },
        "conditions": conditions,
        "subjects": reports,
        "summary": summarize(reports, conditions, args.seed),
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
