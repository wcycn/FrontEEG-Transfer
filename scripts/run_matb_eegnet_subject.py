#!/usr/bin/env python3
"""Train one strict-LOSO Fp1/Fp2 EEGNet baseline on raw MATB windows."""

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

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig

MODEL_CONFIG = EEGNetConfig(
    channels=2,
    samples=1_000,
    temporal_filters=8,
    depth_multiplier=2,
    separable_filters=16,
    temporal_kernel=125,
    separable_kernel=31,
    dropout=0.5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_subjects(root: Path) -> list[str]:
    suffix = "_ses-S1_matb_raw.npz"
    return sorted(path.name.removesuffix(suffix) for path in root.glob(f"*{suffix}"))


def normalize_windows(eeg: np.ndarray) -> np.ndarray:
    """Per-window, per-channel standardization without target-domain statistics."""
    values = np.asarray(eeg, dtype=np.float32)
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    return ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def load_session(root: Path, subject: str, session: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{subject}_ses-{session}_matb_raw.npz"
    with np.load(path, allow_pickle=False) as values:
        eeg = normalize_windows(values["eeg"])
        labels = np.asarray(values["label"], dtype=np.int64)
    return eeg, labels


def concatenate(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def make_loader(
    values: tuple[np.ndarray, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    eeg, labels = values
    dataset = TensorDataset(torch.from_numpy(eeg), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    classes: int = 3,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for eeg, target in loader:
            eeg = eeg.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(eeg)
                loss = functional.cross_entropy(logits, target)
            total_loss += float(loss) * len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    return {
        "loss": total_loss / len(labels),
        "accuracy": float(accuracy_score(label_array, prediction_array)),
        "balanced_accuracy": float(balanced_accuracy_score(label_array, prediction_array)),
        "macro_f1": float(f1_score(label_array, prediction_array, average="macro")),
        "confusion_matrix": confusion_matrix(
            label_array, prediction_array, labels=list(range(classes))
        ).tolist(),
    }


def train_source(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    model_config: EEGNetConfig = MODEL_CONFIG,
    classes: int = 3,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    seed_everything(seed)
    model = EEGNet(model_config, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    train_loader = make_loader(train, batch_size=batch_size, shuffle=True)
    validation_loader = make_loader(validation, batch_size=batch_size, shuffle=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for eeg, labels in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss = functional.cross_entropy(model(eeg), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        validation_metrics = evaluate(model, validation_loader, device, classes=classes)
        if validation_metrics["loss"] < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"source_epoch={epoch} validation_loss={validation_metrics['loss']:.4f} "
            f"validation_bacc={validation_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("EEGNet source training produced no checkpoint")
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    return best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "training_seconds": elapsed,
        "train_windows": len(train[1]),
        "validation_windows": len(validation[1]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": evaluate(model, validation_loader, device, classes=classes),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.raw_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    subjects = discover_subjects(root)
    if len(subjects) != 26 or args.target_subject not in subjects:
        raise ValueError("expected one target among 26 valid COG-BCI subjects")
    source_subjects = [subject for subject in subjects if subject != args.target_subject]
    source_train = concatenate(
        [
            load_session(root, subject, session)
            for subject in source_subjects
            for session in ("S1", "S2")
        ]
    )
    source_validation = concatenate(
        [load_session(root, subject, "S3") for subject in source_subjects]
    )
    target_test = load_session(root, args.target_subject, "S3")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    if args.reuse_checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model_state"]
        source_report = payload["source_training"]
    else:
        state, source_report = train_source(
            source_train,
            source_validation,
            epochs=args.source_epochs,
            patience=args.source_patience,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": state,
                "model_config": MODEL_CONFIG.to_dict(),
                "classes": 3,
                "target_subject": args.target_subject,
                "source_training": source_report,
            },
            checkpoint,
        )
    model = EEGNet(MODEL_CONFIG, classes=3).to(device)
    model.load_state_dict(state)
    test_loader = make_loader(target_test, batch_size=args.batch_size, shuffle=False)
    report = {
        "schema_version": 1,
        "experiment": "EEGNet raw-waveform strict LOSO source-only",
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "protocol": {
            "source_train": "25 subjects S1+S2 labels",
            "source_validation": "25 subjects S3 for early stopping",
            "target_test": "held-out target S3",
            "target_S1_used": False,
            "target_S2_used": False,
            "normalization": "per-window per-channel z-score",
        },
        "model_config": MODEL_CONFIG.to_dict(),
        "source_training": source_report,
        "target_test": evaluate(model, test_loader, device),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "target_test": report["target_test"]}, indent=2))


if __name__ == "__main__":
    main()
