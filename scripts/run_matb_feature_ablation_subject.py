#!/usr/bin/env python3
"""Run one target's frequency-band and channel ablations under strict MATB LOSO."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import run_cross_subject_lowshot_matb as experiment
import run_matb_ot_gnn_paper as baseline
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BANDS = ("theta", "alpha", "low_beta", "high_beta")
CONDITIONS = (
    "full",
    "only_theta",
    "only_alpha",
    "only_low_beta",
    "only_high_beta",
    "drop_theta",
    "drop_alpha",
    "drop_low_beta",
    "drop_high_beta",
    "only_fp1",
    "only_fp2",
)


class MaskedMlp(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(inputs, 53)
        self.classifier = nn.Linear(53, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = functional.relu(self.hidden(values))
        values = functional.dropout(
            values, p=experiment.MODEL_CONFIG.dropout, training=self.training
        )
        return self.classifier(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def feature_indices(condition: str) -> list[int]:
    grid = np.arange(8).reshape(2, 4)
    if condition == "full":
        return grid.ravel().tolist()
    if condition.startswith("only_") and condition.removeprefix("only_") in BANDS:
        band = BANDS.index(condition.removeprefix("only_"))
        return grid[:, band].tolist()
    if condition.startswith("drop_"):
        band = BANDS.index(condition.removeprefix("drop_"))
        return np.delete(grid, band, axis=1).ravel().tolist()
    if condition == "only_fp1":
        return grid[0].tolist()
    if condition == "only_fp2":
        return grid[1].tolist()
    raise ValueError(f"unknown ablation condition {condition}")


def select(features: np.ndarray, indices: list[int]) -> np.ndarray:
    return features.reshape(len(features), -1)[:, indices].astype(np.float32)


def make_loader(
    features: np.ndarray, labels: np.ndarray, *, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=min(experiment.MODEL_CONFIG.batch_size, len(dataset)),
        shuffle=shuffle,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            logits = model(features)
            total_loss += float(functional.cross_entropy(logits, target)) * len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    target = np.asarray(labels)
    prediction = np.asarray(predictions)
    return {
        "loss": total_loss / len(target),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1, 2]).tolist(),
    }


def run_condition(
    condition: str,
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    indices = feature_indices(condition)
    train_loader = make_loader(select(train[0], indices), train[1], shuffle=True)
    validation_loader = make_loader(
        select(validation[0], indices), validation[1], shuffle=False
    )
    test_loader = make_loader(select(test[0], indices), test[1], shuffle=False)
    baseline.seed_everything(seed)
    model = MaskedMlp(len(indices)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-3, weight_decay=experiment.MODEL_CONFIG.weight_decay
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
        validation_metrics = evaluate(model, validation_loader, device)
        if validation_metrics["loss"] < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("feature ablation produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "feature_indices": indices,
        "input_features": len(indices),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "validation": evaluate(model, validation_loader, device),
        "target_test": evaluate(model, test_loader, device),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    subjects = experiment.discover_subjects(root)
    if len(subjects) != 26 or args.target_subject not in subjects:
        raise ValueError("expected one target among 26 valid subjects")
    source_subjects = [subject for subject in subjects if subject != args.target_subject]
    sources = {subject: experiment.load_subject(root, subject) for subject in source_subjects}
    target = experiment.load_subject(root, args.target_subject)
    train = experiment.concatenate(
        [sources[subject][session] for subject in source_subjects for session in ("S1", "S2")]
    )
    validation = experiment.concatenate(
        [sources[subject]["S3"] for subject in source_subjects]
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for condition in CONDITIONS:
        results[condition] = run_condition(
            condition,
            train,
            validation,
            target["S3"],
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device=device,
        )
        print(
            f"target={args.target_subject} condition={condition} "
            f"bacc={results[condition]['target_test']['balanced_accuracy']:.4f}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "experiment": "Fp1/Fp2 PSD frequency-band and channel ablation",
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "protocol": {
            "source_train": "25 subjects S1+S2",
            "source_validation": "25 subjects S3",
            "target_test": "held-out target S3",
            "normalization": "each subject S1 MinMax, matching main experiment",
            "learning_rate": 0.003,
        },
        "bands": list(BANDS),
        "conditions": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
