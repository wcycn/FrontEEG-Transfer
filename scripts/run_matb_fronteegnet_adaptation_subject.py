#!/usr/bin/env python3
"""Evaluate nested low-shot adaptation of one strict-LOSO FrontEEGNet fold."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from run_matb_eegnet_subject import evaluate, load_session, make_loader, seed_everything

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig

AdaptationMode = Literal["head", "partial", "full", "scratch"]
ADAPTATION_MODES: tuple[AdaptationMode, ...] = ("head", "partial", "full", "scratch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 4, 8, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--adapt-epochs", type=int, default=200)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def nested_nonoverlapping_indices(
    labels: np.ndarray, maximum: int, seed: int, classes: int = 3
) -> dict[int, list[int]]:
    """Return nested balanced prefixes after removing 50%-overlap neighbours."""
    rng = np.random.default_rng(seed)
    ordered: list[np.ndarray] = []
    for label in range(classes):
        candidates = np.flatnonzero(labels == label)[::2]
        rng.shuffle(candidates)
        if len(candidates) < maximum:
            raise ValueError(
                f"class {label} has {len(candidates)} non-overlapping windows, "
                f"fewer than k={maximum}"
            )
        ordered.append(candidates[:maximum])
    return {
        budget: sorted(
            int(index) for class_indices in ordered for index in class_indices[:budget]
        )
        for budget in range(1, maximum + 1)
    }


def configure_trainable_parameters(model: EEGNet, mode: AdaptationMode) -> int:
    """Configure adaptation scope and return the trainable parameter count."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        modules = (model.head,)
    elif mode == "partial":
        # Adapt the separable temporal convolutions and classifier while keeping all
        # batch-normalization statistics fixed for low-shot stability.
        modules = (model.features[7], model.features[8], model.head)
    elif mode in {"full", "scratch"}:
        modules = (model,)
    else:
        raise ValueError(f"unknown adaptation mode: {mode}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_adaptation_train_mode(model: EEGNet, mode: AdaptationMode) -> None:
    """Prevent frozen batch-normalization statistics from changing during adaptation."""
    if mode in {"full", "scratch"}:
        model.train()
        return
    model.eval()
    if mode == "partial":
        model.features[7].train()
        model.features[8].train()
    model.head.train()


def learning_rate(mode: AdaptationMode) -> float:
    return {
        "head": 1e-3,
        "partial": 3e-4,
        "full": 1e-4,
        "scratch": 1e-3,
    }[mode]


def adapt_model(
    calibration: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    source_state: dict[str, torch.Tensor],
    model_config: EEGNetConfig,
    mode: AdaptationMode,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    seed_everything(seed)
    model = EEGNet(model_config, classes=3).to(device)
    if mode != "scratch":
        model.load_state_dict(source_state)
    trainable_parameters = configure_trainable_parameters(model, mode)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate(mode),
        weight_decay=0.0,
    )
    loader = make_loader(calibration, batch_size=batch_size, shuffle=True)
    started = time.perf_counter()
    for _ in range(epochs):
        set_adaptation_train_mode(model, mode)
        for eeg, labels in loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(eeg), labels)
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = evaluate(
        model,
        make_loader(test, batch_size=batch_size, shuffle=False),
        device,
        classes=3,
    )
    return metrics, {
        "epochs": epochs,
        "optimizer_steps": epochs * math.ceil(len(calibration[1]) / batch_size),
        "labeled_windows": int(len(calibration[1])),
        "trainable_parameters": trainable_parameters,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate(mode),
        "training_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    if args.budgets != sorted(set(args.budgets)) or min(args.budgets) <= 0:
        raise ValueError("budgets must be positive, unique, and increasing")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    raw_root = Path(args.raw_root).expanduser().resolve()
    checkpoint_path = Path(args.source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("target_subject") != args.target_subject:
        raise ValueError("source checkpoint and requested target subject do not match")
    if int(checkpoint.get("classes", -1)) != 3:
        raise ValueError("expected a three-class source checkpoint")
    model_config = EEGNetConfig(**checkpoint["model_config"])
    source_state = checkpoint["model_state"]
    target_calibration = load_session(raw_root, args.target_subject, "S1")
    target_test = load_session(raw_root, args.target_subject, "S3")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    source_model = EEGNet(model_config, classes=3).to(device)
    source_model.load_state_dict(source_state)
    source_only = evaluate(
        source_model,
        make_loader(target_test, batch_size=args.batch_size, shuffle=False),
        device,
        classes=3,
    )
    records: list[dict[str, object]] = []
    sampled: dict[str, dict[str, list[int]]] = {}
    maximum = max(args.budgets)
    for repeat in range(args.repeats):
        choices = nested_nonoverlapping_indices(
            target_calibration[1], maximum, args.seed + 1_000 + repeat
        )
        sampled[str(repeat)] = {str(budget): choices[budget] for budget in args.budgets}
        for budget in args.budgets:
            indices = np.asarray(choices[budget], dtype=np.int64)
            calibration = (
                target_calibration[0][indices],
                target_calibration[1][indices],
            )
            for mode in ADAPTATION_MODES:
                epochs = args.scratch_epochs if mode == "scratch" else args.adapt_epochs
                metrics, training = adapt_model(
                    calibration,
                    target_test,
                    source_state=source_state,
                    model_config=model_config,
                    mode=mode,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    seed=args.seed + 100_000 * repeat + 1_000 * budget,
                    device=device,
                )
                records.append(
                    {
                        "repeat": repeat,
                        "budget_per_class": budget,
                        "method": mode,
                        "minimum_labeled_signal_seconds": 3 * budget * 4.0,
                        "training": training,
                        "target_test": metrics,
                    }
                )
                print(
                    f"target={args.target_subject} repeat={repeat} k={budget} "
                    f"method={mode} bacc={metrics['balanced_accuracy']:.4f}",
                    flush=True,
                )

    report = {
        "schema_version": 1,
        "experiment": "FrontEEGNet controlled nested low-shot target adaptation",
        "target_subject": args.target_subject,
        "protocol": {
            "source_checkpoint": str(checkpoint_path),
            "source_train": "25 source subjects S1+S2",
            "source_validation": "25 source subjects S3",
            "target_calibration": "held-out target S1 nested non-overlapping windows",
            "target_test": "held-out target S3, never used for optimization or selection",
            "target_S2_used": False,
            "normalization": "per-window per-channel z-score",
        },
        "model_config": model_config.to_dict(),
        "source_only_target_test": source_only,
        "budgets_per_class": args.budgets,
        "repeats": args.repeats,
        "adapt_epochs": args.adapt_epochs,
        "scratch_epochs": args.scratch_epochs,
        "nested_sampling_indices": sampled,
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output} source_only={source_only['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
