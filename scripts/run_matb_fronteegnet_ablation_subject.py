#!/usr/bin/env python3
"""Train one strict-LOSO FrontEEGNet architecture ablation on MATB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_matb_eegnet_subject import (
    MODEL_CONFIG,
    concatenate,
    discover_subjects,
    evaluate,
    make_loader,
    normalize_windows,
    train_source,
)

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig

VARIANTS = (
    "short_temporal",
    "fixed_common_difference",
    "fp1_only",
    "fp2_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    return parser.parse_args()


def variant_spec(name: str) -> tuple[EEGNetConfig, tuple[int, ...], str]:
    base = MODEL_CONFIG.to_dict()
    if name == "short_temporal":
        base.update(temporal_kernel=63, separable_kernel=15)
        return EEGNetConfig(**base), (0, 1), "shorter temporal kernels (63 and 15)"
    if name == "fixed_common_difference":
        base.update(spatial_mode="fixed_mean_difference")
        return (
            EEGNetConfig(**base),
            (0, 1),
            "fixed Fp1/Fp2 common and difference projection",
        )
    if name == "fp1_only":
        base.update(channels=1)
        return EEGNetConfig(**base), (0,), "Fp1 only"
    if name == "fp2_only":
        base.update(channels=1)
        return EEGNetConfig(**base), (1,), "Fp2 only"
    raise ValueError(f"unknown variant: {name}")


def load_session(
    root: Path, subject: str, session: str, channel_indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{subject}_ses-{session}_matb_raw.npz"
    with np.load(path, allow_pickle=False) as values:
        eeg = np.asarray(values["eeg"], dtype=np.float32)[:, channel_indices, :]
        labels = np.asarray(values["label"], dtype=np.int64)
    return normalize_windows(eeg), labels


def main() -> None:
    args = parse_args()
    root = Path(args.raw_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    subjects = discover_subjects(root)
    if len(subjects) != 26 or args.target_subject not in subjects:
        raise ValueError("expected one target among 26 valid COG-BCI participants")
    model_config, channel_indices, intervention = variant_spec(args.variant)
    source_subjects = [subject for subject in subjects if subject != args.target_subject]
    source_train = concatenate(
        [
            load_session(root, subject, session, channel_indices)
            for subject in source_subjects
            for session in ("S1", "S2")
        ]
    )
    source_validation = concatenate(
        [load_session(root, subject, "S3", channel_indices) for subject in source_subjects]
    )
    target_test = load_session(root, args.target_subject, "S3", channel_indices)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    if args.reuse_checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model_state"]
        source_training = payload["source_training"]
    else:
        state, source_training = train_source(
            source_train,
            source_validation,
            epochs=args.source_epochs,
            patience=args.source_patience,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            model_config=model_config,
            classes=3,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": state,
                "model_config": model_config.to_dict(),
                "classes": 3,
                "variant": args.variant,
                "target_subject": args.target_subject,
                "source_training": source_training,
            },
            checkpoint,
        )

    model = EEGNet(model_config, classes=3).to(device)
    model.load_state_dict(state)
    target_metrics = evaluate(
        model,
        make_loader(target_test, batch_size=args.batch_size, shuffle=False),
        device,
    )
    report = {
        "schema_version": 1,
        "experiment": "FrontEEGNet MATB strict-LOSO architecture ablation",
        "variant": args.variant,
        "intervention": intervention,
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "channel_indices": list(channel_indices),
        "protocol": {
            "source_train": "25 participants S1+S2 labels",
            "source_validation": "25 participants S3 for early stopping",
            "target_test": "held-out target participant S3",
            "target_data_used_for_training_or_normalization": False,
            "normalization": "per-window per-channel z-score",
        },
        "model_config": model_config.to_dict(),
        "source_training": source_training,
        "target_test": target_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"variant={args.variant} target={args.target_subject} "
        f"bacc={target_metrics['balanced_accuracy']:.4f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
