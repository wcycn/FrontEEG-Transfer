#!/usr/bin/env python3
"""Run one strict participant-held-out FrontEEGNet experiment on ds007169."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_matb_eegnet_subject import (
    concatenate,
    evaluate,
    make_loader,
    normalize_windows,
    train_source,
)
from scipy.signal import resample_poly

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig
from fronteeg_transfer.external_ds007169 import (
    discover_recordings,
    source_temporal_partition,
    temporal_partition,
)

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
    parser.add_argument("--target-prefix-windows-per-class", type=int, default=8)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    return parser.parse_args()


def load_subject(
    recordings: dict[str, dict[int, Path]], subject: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    within_parts: list[np.ndarray] = []
    for level, path in sorted(recordings[subject].items()):
        with np.load(path, allow_pickle=False) as loaded:
            eeg = np.asarray(loaded["eeg"], dtype=np.float32)
            labels = np.asarray(loaded["label"], dtype=np.int64)
        if eeg.ndim != 3 or eeg.shape[1:] != (2, 800):
            raise ValueError(f"expected [N,2,800] in {path}, found {eeg.shape}")
        if not np.all(labels == level - 1):
            raise ValueError(f"label mismatch in {path}")
        resampled = resample_poly(eeg, up=5, down=4, axis=-1).astype(np.float32)
        if resampled.shape[1:] != (2, 1_000):
            raise ValueError(f"resampling failed for {path}: {resampled.shape}")
        eeg_parts.append(normalize_windows(resampled))
        label_parts.append(labels)
        within_parts.append(np.arange(len(labels), dtype=np.int64))
    return (
        np.concatenate(eeg_parts),
        np.concatenate(label_parts),
        np.concatenate(within_parts),
    )


def source_split(
    eeg: np.ndarray, labels: np.ndarray, within: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    split = source_temporal_partition(labels, within)
    return (
        (eeg[split["train"]], labels[split["train"]]),
        (eeg[split["validation"]], labels[split["validation"]]),
    )


def main() -> None:
    args = parse_args()
    if args.target_prefix_windows_per_class <= 0 or args.target_prefix_windows_per_class % 2:
        raise ValueError("target prefix must be a positive even number")
    root = Path(args.raw_root).expanduser().resolve()
    recordings = discover_recordings(root)
    subjects = sorted(recordings)
    if len(subjects) != 18 or args.target_subject not in recordings:
        raise ValueError("expected one target among 18 complete ds007169 participants")

    source_train: list[tuple[np.ndarray, np.ndarray]] = []
    source_validation: list[tuple[np.ndarray, np.ndarray]] = []
    for subject in subjects:
        if subject == args.target_subject:
            continue
        split = source_split(*load_subject(recordings, subject))
        source_train.append(split[0])
        source_validation.append(split[1])

    target_eeg, target_labels, target_within = load_subject(recordings, args.target_subject)
    maximum_budget = args.target_prefix_windows_per_class // 2
    target_indices = temporal_partition(
        target_labels, target_within, maximum_budget=maximum_budget
    )["test"]
    target_test = target_eeg[target_indices], target_labels[target_indices]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if args.reuse_checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_state = payload["model_state"]
        source_training = payload["source_training"]
    else:
        source_state, source_training = train_source(
            concatenate(source_train),
            concatenate(source_validation),
            epochs=args.source_epochs,
            patience=args.source_patience,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            model_config=MODEL_CONFIG,
            classes=4,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": source_state,
                "model_config": MODEL_CONFIG.to_dict(),
                "classes": 4,
                "target_subject": args.target_subject,
                "source_training": source_training,
            },
            checkpoint,
        )

    model = EEGNet(MODEL_CONFIG, classes=4).to(device)
    model.load_state_dict(source_state)
    target_metrics = evaluate(
        model,
        make_loader(target_test, batch_size=args.batch_size, shuffle=False),
        device,
        classes=4,
    )
    report = {
        "schema_version": 1,
        "experiment": "FrontEEGNet ds007169 strict participant-held-out source-only",
        "target_subject": args.target_subject,
        "source_subjects": [subject for subject in subjects if subject != args.target_subject],
        "classes": ["1-back", "2-back", "3-back", "4-back"],
        "protocol": {
            "source_train": "early temporal 75% of every source recording",
            "source_validation": "late temporal 25% after a one-stride guard",
            "target_test": (
                f"target windows after the first {args.target_prefix_windows_per_class} "
                "windows per class"
            ),
            "target_data_used_for_training_or_normalization": False,
            "resampling": "polyphase 200 Hz to 250 Hz",
            "normalization": "per-window per-channel z-score",
        },
        "model_config": MODEL_CONFIG.to_dict(),
        "source_training": source_training,
        "target_test_windows": int(len(target_test[1])),
        "target_test_per_class": [
            int(np.sum(target_test[1] == label)) for label in range(4)
        ],
        "target_test": target_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"target={args.target_subject} bacc={target_metrics['balanced_accuracy']:.4f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
