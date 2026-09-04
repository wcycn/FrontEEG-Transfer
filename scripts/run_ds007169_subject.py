#!/usr/bin/env python3
"""Run one strict participant-held-out ds007169 PSD transfer experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fronteeg_transfer.external_ds007169 import (
    apply_minmax,
    minmax_statistics,
    source_temporal_partition,
    temporal_partition,
)

DROPOUT = 0.10387442330843398
WEIGHT_DECAY = 9.964185030586058e-4


class ExternalMlp(nn.Module):
    """The MATB deployment MLP with a four-class output layer."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(8, 53)
        self.classifier = nn.Linear(53, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = functional.relu(self.hidden(features))
        features = functional.dropout(features, p=DROPOUT, training=self.training)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject(root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / f"{subject}_fp1_fp2_psd.npz"
    with np.load(path, allow_pickle=False) as loaded:
        features = loaded["features"].astype(np.float32)
        labels = loaded["label"].astype(np.int64)
        within = loaded["within_class_index"].astype(np.int64)
    if features.ndim != 3 or features.shape[1:] != (2, 4):
        raise ValueError(f"unexpected features in {path}: {features.shape}")
    if set(np.unique(labels).tolist()) != {0, 1, 2, 3}:
        raise ValueError(f"unexpected labels in {path}")
    return features, labels, within


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    shuffle: bool,
    batch_size: int = 256,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.reshape(len(features), -1).astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)


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
    truth = np.asarray(labels)
    predicted = np.asarray(predictions)
    return {
        "loss": total_loss / len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2, 3]).tolist(),
    }


def train_source(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    *,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    seed_everything(seed)
    train_loader = make_loader(*train, shuffle=True)
    validation_loader = make_loader(*validation, shuffle=False)
    model = ExternalMlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(features), target)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if float(metrics["loss"]) < best_loss:
            best_loss = float(metrics["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("source training produced no checkpoint")
    model.load_state_dict(best_state)
    return best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "training_seconds": time.perf_counter() - started,
        "train_windows": len(train[1]),
        "validation_windows": len(validation[1]),
        "validation": evaluate(model, validation_loader, device),
    }


def train_target(
    calibration: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    source_state: dict[str, torch.Tensor] | None,
    method: str,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    seed_everything(seed)
    model = ExternalMlp().to(device)
    if source_state is not None:
        model.load_state_dict(source_state)
    if method == "linear":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        learning_rate = 1e-3
    elif method == "full":
        learning_rate = 1e-4
    elif method == "scratch":
        learning_rate = 3e-3
    else:
        raise ValueError(f"unknown method {method}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    calibration_loader = make_loader(*calibration, shuffle=True)
    test_loader = make_loader(*test, shuffle=False)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for features, target in calibration_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(features), target)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    return evaluate(model, test_loader, device), {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_seconds": elapsed,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


def normalize_source(
    features: np.ndarray,
    labels: np.ndarray,
    within: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    split = source_temporal_partition(labels, within)
    minimum, span = minmax_statistics(features[split["train"]])
    normalized = apply_minmax(features, minimum, span)
    return (
        (normalized[split["train"]], labels[split["train"]]),
        (normalized[split["validation"]], labels[split["validation"]]),
    )


def concatenate(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def main() -> None:
    args = parse_args()
    if args.budgets != sorted(set(args.budgets)) or min(args.budgets) <= 0:
        raise ValueError("budgets must be positive, unique, and increasing")
    root = Path(args.feature_root).expanduser().resolve()
    subjects = sorted(
        path.name.removesuffix("_fp1_fp2_psd.npz") for path in root.glob("sub-*_fp1_fp2_psd.npz")
    )
    if args.target_subject not in subjects:
        raise ValueError(f"target {args.target_subject} not found")
    sources_train, sources_validation = [], []
    for subject in subjects:
        if subject == args.target_subject:
            continue
        source = normalize_source(*load_subject(root, subject))
        sources_train.append(source[0])
        sources_validation.append(source[1])
    target_features, target_labels, target_within = load_subject(root, args.target_subject)
    target_split = temporal_partition(
        target_labels, target_within, maximum_budget=max(args.budgets)
    )
    normalization_indices = np.flatnonzero(target_within < 2 * max(args.budgets))
    minimum, span = minmax_statistics(target_features[normalization_indices])
    target_features = apply_minmax(target_features, minimum, span)
    target_test = (
        target_features[target_split["test"]],
        target_labels[target_split["test"]],
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_state, source_training = train_source(
        concatenate(sources_train),
        concatenate(sources_validation),
        epochs=args.source_epochs,
        patience=args.source_patience,
        seed=args.seed,
        device=device,
    )
    source_model = ExternalMlp().to(device)
    source_model.load_state_dict(source_state)
    source_only = evaluate(source_model, make_loader(*target_test, shuffle=False), device)
    records = []
    nested_indices: dict[str, dict[str, list[int]]] = {}
    candidates = target_split["calibration_candidates"]
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + 1000 + repeat)
        ordered = []
        for label in range(4):
            class_candidates = candidates[target_labels[candidates] == label].copy()
            rng.shuffle(class_candidates)
            ordered.append(class_candidates)
        nested_indices[str(repeat)] = {}
        for budget in args.budgets:
            indices = np.asarray(
                sorted(int(index) for values in ordered for index in values[:budget]),
                dtype=np.int64,
            )
            nested_indices[str(repeat)][str(budget)] = indices.tolist()
            calibration = (target_features[indices], target_labels[indices])
            for method, offset in (("scratch", 100), ("linear", 200), ("full", 300)):
                epochs = args.scratch_epochs if method == "scratch" else args.adapt_epochs
                metrics, training = train_target(
                    calibration,
                    target_test,
                    source_state=None if method == "scratch" else source_state,
                    method=method,
                    epochs=epochs,
                    seed=args.seed + repeat * 10000 + budget * 10 + offset,
                    device=device,
                )
                records.append(
                    {
                        "repeat": repeat,
                        "budget_per_class": budget,
                        "method": method,
                        "test": metrics,
                        "training": training,
                    }
                )
    report = {
        "schema_version": 1,
        "experiment": "ds007169 participant-disjoint Fp1/Fp2 PSD transfer",
        "target_subject": args.target_subject,
        "source_subjects": [subject for subject in subjects if subject != args.target_subject],
        "classes": ["1-back", "2-back", "3-back", "4-back"],
        "protocol": {
            "source_train": "early temporal 75% of every source recording",
            "source_validation": (
                "late temporal 25% of every source recording after one-stride guard"
            ),
            "target_normalization": "first 2*maximum_budget windows per class, labels hidden",
            "target_calibration": (
                "nested non-overlapping windows from target normalization segment"
            ),
            "target_test": "all later target windows with no raw-sample overlap with calibration",
            "target_test_used_for_selection": False,
        },
        "budgets_per_class": args.budgets,
        "repeats": args.repeats,
        "source_training": source_training,
        "source_only_target_test": source_only,
        "target_test_windows": len(target_test[1]),
        "target_test_per_class": [int(np.sum(target_test[1] == label)) for label in range(4)],
        "nested_sampling_indices": nested_indices,
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"target={args.target_subject} source_only_bacc="
        f"{source_only['balanced_accuracy']:.4f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
