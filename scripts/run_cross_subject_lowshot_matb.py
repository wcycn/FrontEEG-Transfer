#!/usr/bin/env python3
"""Run a strict Fp1/Fp2 multi-source to low-shot target-subject MATB experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import run_matb_ot_gnn_paper as baseline
import torch
import torch.nn.functional as functional
from run_matb_ot_gnn_fp1_fp2 import fp1_fp2_graph, select_fp1_fp2
from torch_geometric.loader import DataLoader

MODEL_CONFIG = baseline.ModelConfig(
    graph_widths=(26, 13, 9),
    linear_widths=(53,),
    activation="relu",
    batch_size=256,
    learning_rate=6.056463457686898e-4,
    weight_decay=9.964185030586058e-4,
    dropout=0.10387442330843398,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--target-subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=60)
    parser.add_argument("--source-patience", type=int, default=10)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--max-source-subjects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def discover_subjects(root: Path) -> list[str]:
    suffix = "_ses-S1_matb_psd.npz"
    return sorted(path.name.removesuffix(suffix) for path in root.glob(f"*{suffix}"))


def load_subject(root: Path, subject: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    loaded = {
        session: select_fp1_fp2(baseline.load_session(root, subject, session))
        for session in ("S1", "S2", "S3")
    }
    channel_names = loaded["S1"][2]
    if channel_names != ["Fp1", "Fp2"]:
        raise ValueError(f"unexpected Fp1/Fp2 order for {subject}: {channel_names}")
    minimum = loaded["S1"][0].min(axis=0)
    span = loaded["S1"][0].max(axis=0) - minimum
    if np.any(span <= 0):
        raise ValueError(f"constant S1 channel-band feature for {subject}")
    return {
        session: (((values[0] - minimum) / span).astype(np.float32), values[1])
        for session, values in loaded.items()
    }


def concatenate(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def make_data(features: np.ndarray, labels: np.ndarray):
    edge_index, edge_weight = fp1_fp2_graph()
    return baseline.make_dataset(features, labels, edge_index, edge_weight)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def train_source(
    train_data,
    validation_data,
    *,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    baseline.seed_everything(seed)
    train_loader = DataLoader(train_data, batch_size=MODEL_CONFIG.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=MODEL_CONFIG.batch_size, shuffle=False
    )
    model = baseline.GcnClassifier(MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG.learning_rate,
        weight_decay=MODEL_CONFIG.weight_decay,
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
        validation = baseline.evaluate(model, validation_loader, device)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
        print(
            f"source_epoch={epoch} val_loss={validation['loss']:.4f} "
            f"val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("source training produced no checkpoint")
    synchronize(device)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    return best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "optimizer_steps": optimizer_steps,
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "training_seconds": training_seconds,
        "processed_windows_per_second": len(train_data) * epoch / training_seconds,
        "validation": baseline.evaluate(model, validation_loader, device),
    }


def nested_indices(labels: np.ndarray, maximum: int, seed: int) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    ordered = []
    for label in range(3):
        candidates = np.flatnonzero(labels == label)[::2]
        rng.shuffle(candidates)
        if len(candidates) < maximum:
            raise ValueError(f"label {label} has only {len(candidates)} non-overlapping candidates")
        ordered.append(candidates[:maximum])
    return {
        budget: sorted(int(index) for label_indices in ordered for index in label_indices[:budget])
        for budget in range(1, maximum + 1)
    }


def train_target(
    calibration_data,
    test_data,
    source_state: dict | None,
    method: str,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    baseline.seed_everything(seed)
    model = baseline.GcnClassifier(MODEL_CONFIG).to(device)
    if source_state is not None:
        model.load_state_dict(source_state)
    if method == "linear":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.linear_layers[-1].parameters():
            parameter.requires_grad = True
        learning_rate = 1e-3
    elif method == "full":
        learning_rate = 1e-4
    elif method == "scratch":
        learning_rate = MODEL_CONFIG.learning_rate
    else:
        raise ValueError(f"unknown target method {method}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=learning_rate,
        weight_decay=MODEL_CONFIG.weight_decay,
    )
    loader = DataLoader(
        calibration_data,
        batch_size=min(MODEL_CONFIG.batch_size, len(calibration_data)),
        shuffle=True,
    )
    optimizer_steps = 0
    synchronize(device)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    synchronize(device)
    training_seconds = time.perf_counter() - started
    test_loader = DataLoader(test_data, batch_size=MODEL_CONFIG.batch_size, shuffle=False)
    synchronize(device)
    started = time.perf_counter()
    metrics = baseline.evaluate(model, test_loader, device)
    synchronize(device)
    inference_seconds = time.perf_counter() - started
    return metrics, {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "batches_per_epoch": math.ceil(len(calibration_data) / MODEL_CONFIG.batch_size),
        "labeled_windows": len(calibration_data),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "total_parameters": parameter_count(model),
        "learning_rate": learning_rate,
        "training_seconds": training_seconds,
        "test_windows": len(test_data),
        "inference_seconds": inference_seconds,
        "test_windows_per_second": len(test_data) / inference_seconds,
    }


def aggregate(records: list[dict]) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["budget_per_class"])].append(record)
    summary = {}
    for (method, budget), records_for_cell in sorted(grouped.items()):
        key = f"{method}@{budget}"
        summary[key] = {
            "test": {
                name: {
                    "mean": float(
                        np.mean([record["test"][name] for record in records_for_cell])
                    ),
                    "std": float(
                        np.std([record["test"][name] for record in records_for_cell])
                    ),
                }
                for name in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            "training_seconds": {
                "mean": float(
                    np.mean(
                        [record["training"]["training_seconds"] for record in records_for_cell]
                    )
                ),
                "std": float(
                    np.std(
                        [record["training"]["training_seconds"] for record in records_for_cell]
                    )
                ),
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    all_subjects = discover_subjects(root)
    if args.target_subject not in all_subjects:
        raise ValueError(f"target {args.target_subject} not found under {root}")
    source_subjects = [subject for subject in all_subjects if subject != args.target_subject]
    if args.max_source_subjects > 0:
        source_subjects = source_subjects[: args.max_source_subjects]
    if not source_subjects:
        raise ValueError("at least one source subject is required")
    maximum_budget = max(args.budgets)
    if min(args.budgets) <= 0 or args.budgets != sorted(set(args.budgets)):
        raise ValueError("budgets must be positive, unique, and increasing")

    print(
        f"target={args.target_subject} source_subjects={len(source_subjects)}",
        flush=True,
    )
    source = {subject: load_subject(root, subject) for subject in source_subjects}
    target = load_subject(root, args.target_subject)
    source_train = concatenate(
        [source[subject][session] for subject in source_subjects for session in ("S1", "S2")]
    )
    source_validation = concatenate([source[subject]["S3"] for subject in source_subjects])
    source_train_data = make_data(*source_train)
    source_validation_data = make_data(*source_validation)
    target_test_data = make_data(*target["S3"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_state, source_report = train_source(
        source_train_data,
        source_validation_data,
        epochs=args.source_epochs,
        patience=args.source_patience,
        seed=args.seed,
        device=device,
    )
    source_model = baseline.GcnClassifier(MODEL_CONFIG).to(device)
    source_model.load_state_dict(source_state)
    target_test_loader = DataLoader(
        target_test_data, batch_size=MODEL_CONFIG.batch_size, shuffle=False
    )
    synchronize(device)
    started = time.perf_counter()
    source_only = baseline.evaluate(source_model, target_test_loader, device)
    synchronize(device)
    source_only_inference_seconds = time.perf_counter() - started
    print(f"source_only_test={source_only['accuracy']:.4f}", flush=True)

    records = []
    sampling = {}
    for repeat in range(args.repeats):
        choices = nested_indices(target["S1"][1], maximum_budget, args.seed + 1000 + repeat)
        sampling[str(repeat)] = {str(budget): choices[budget] for budget in args.budgets}
        for budget in args.budgets:
            indices = np.asarray(choices[budget], dtype=np.int64)
            calibration_data = make_data(target["S1"][0][indices], target["S1"][1][indices])
            for method in ("scratch", "linear", "full"):
                metrics, training = train_target(
                    calibration_data,
                    target_test_data,
                    None if method == "scratch" else source_state,
                    method,
                    epochs=args.adapt_epochs,
                    seed=args.seed + 10000 * repeat + budget,
                    device=device,
                )
                records.append(
                    {
                        "repeat": repeat,
                        "budget_per_class": budget,
                        "total_labeled_windows": 3 * budget,
                        "minimum_labeled_signal_seconds": 3 * budget * 4.0,
                        "method": method,
                        "training": training,
                        "test": metrics,
                    }
                )
                print(
                    f"repeat={repeat} budget={budget} method={method} "
                    f"test={metrics['accuracy']:.4f}",
                    flush=True,
                )

    report = {
        "schema_version": 2,
        "protocol": "multi-source subjects -> held-out target low-shot calibration",
        "task": "COG-BCI MATB Easy/Medium/Difficult",
        "channels": ["Fp1", "Fp2"],
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "source_train_sessions": ["S1", "S2"],
        "source_validation_session": "S3",
        "target_unlabeled_normalization_session": "S1 (all windows, labels hidden)",
        "target_labeled_calibration_session": "S1 (nested non-overlapping-window subset)",
        "target_test_session": "S3 (all labels hidden until scoring)",
        "target_session_S2_used": False,
        "model_config": {
            "graph_widths": MODEL_CONFIG.graph_widths,
            "linear_widths": MODEL_CONFIG.linear_widths,
            "activation": MODEL_CONFIG.activation,
            "dropout": MODEL_CONFIG.dropout,
        },
        "source_training": source_report,
        "source_only_target_test": source_only,
        "source_only_inference": {
            "test_windows": len(target_test_data),
            "inference_seconds": source_only_inference_seconds,
            "test_windows_per_second": len(target_test_data) / source_only_inference_seconds,
        },
        "budgets_per_class": args.budgets,
        "repeats": args.repeats,
        "adapt_epochs": args.adapt_epochs,
        "nested_sampling_indices": sampling,
        "records": records,
        "aggregate": aggregate(records),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
