#!/usr/bin/env python3
"""Train the final all-public-participant FrontEEGNet deployment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_matb_eegnet_subject import (
    MODEL_CONFIG,
    concatenate,
    discover_subjects,
    load_session,
    train_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_json).expanduser().resolve()
    if (checkpoint.exists() or output.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite final FrontEEGNet artifacts")
    subjects = discover_subjects(raw_root)
    if len(subjects) != 26:
        raise ValueError(f"expected 26 valid COG-BCI participants, found {len(subjects)}")
    source_train = concatenate(
        [
            load_session(raw_root, subject, session)
            for subject in subjects
            for session in ("S1", "S2")
        ]
    )
    source_validation = concatenate(
        [load_session(raw_root, subject, "S3") for subject in subjects]
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    state, training = train_source(
        source_train,
        source_validation,
        epochs=args.source_epochs,
        patience=args.source_patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        model_config=MODEL_CONFIG,
        classes=3,
    )
    payload = {
        "schema_version": 1,
        "model_name": "FrontEEGNet",
        "model_state": state,
        "model_config": MODEL_CONFIG.to_dict(),
        "classes": 3,
        "class_names": ["Easy", "Medium", "Difficult"],
        "source_subjects": subjects,
        "source_training": training,
        "input": {
            "channels": ["Fp1", "Fp2"],
            "sampling_rate_hz": 250.0,
            "window_seconds": 4.0,
            "stride_seconds": 2.0,
            "filter_hz": [1.0, 40.0],
            "normalization": "per-window per-channel z-score",
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    report = {key: value for key, value in payload.items() if key != "model_state"}
    report["checkpoint"] = str(checkpoint)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"checkpoint={checkpoint} best_epoch={training['best_epoch']} "
        f"validation_bacc={training['validation']['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
