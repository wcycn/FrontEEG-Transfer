#!/usr/bin/env python3
"""Run flat Linear and MLP baselines under the exact Fp1/Fp2 LOSO protocol."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import run_cross_subject_lowshot_matb as experiment
import run_matb_ot_gnn_paper as baseline
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ARCHITECTURES = ("linear", "mlp")


class FlatLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(8, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(values)


class FlatMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(8, 53)
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
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_model(architecture: str) -> nn.Module:
    if architecture == "linear":
        return FlatLinear()
    if architecture == "mlp":
        return FlatMlp()
    raise ValueError(f"unknown architecture {architecture}")


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features.reshape(len(features), -1), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=min(experiment.MODEL_CONFIG.batch_size, len(dataset)),
        shuffle=shuffle,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            logits = model(features)
            total_loss += functional.cross_entropy(logits, target).item() * len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": total_loss / len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def learning_rates(architecture: str) -> tuple[float, ...]:
    if architecture == "linear":
        return (1e-2, 3e-3, 1e-3)
    return (3e-3, 1e-3, 3e-4)


def train_source(
    architecture: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    train_loader = make_loader(train_features, train_labels, shuffle=True)
    validation_loader = make_loader(validation_features, validation_labels, shuffle=False)
    candidates = []
    for learning_rate in learning_rates(architecture):
        baseline.seed_everything(seed)
        model = make_model(architecture).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=experiment.MODEL_CONFIG.weight_decay,
        )
        best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for features, target in train_loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = functional.cross_entropy(model(features), target)
                loss.backward()
                optimizer.step()
            validation = evaluate(model, validation_loader, device)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("flat source training produced no checkpoint")
        candidates.append(
            {
                "learning_rate": learning_rate,
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
                "state": best_state,
            }
        )
    selected = min(candidates, key=lambda item: item["validation_loss"])
    model = make_model(architecture).to(device)
    model.load_state_dict(selected["state"])
    report = {
        "selected_learning_rate": selected["learning_rate"],
        "best_epoch": selected["best_epoch"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": evaluate(model, validation_loader, device),
        "candidate_validation_losses": {
            str(item["learning_rate"]): item["validation_loss"] for item in candidates
        },
    }
    return selected["state"], report


def train_target(
    architecture: str,
    calibration: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    source_state: dict | None,
    mode: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    baseline.seed_everything(seed)
    model = make_model(architecture).to(device)
    if source_state is not None:
        model.load_state_dict(source_state)
    if mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif mode not in {"scratch", "full"}:
        raise ValueError(f"unknown mode {mode}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=learning_rate, weight_decay=experiment.MODEL_CONFIG.weight_decay
    )
    calibration_loader = make_loader(*calibration, shuffle=True)
    test_loader = make_loader(*test, shuffle=False)
    experiment.synchronize(device)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for features, target in calibration_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(features), target)
            loss.backward()
            optimizer.step()
    experiment.synchronize(device)
    elapsed = time.perf_counter() - started
    return evaluate(model, test_loader, device), {
        "epochs": epochs,
        "training_seconds": elapsed,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    subjects = experiment.discover_subjects(root)
    if args.target_subject not in subjects:
        raise ValueError(f"target {args.target_subject} not found")
    source_subjects = [subject for subject in subjects if subject != args.target_subject]
    sources = {subject: experiment.load_subject(root, subject) for subject in source_subjects}
    target = experiment.load_subject(root, args.target_subject)
    source_train = experiment.concatenate(
        [sources[subject][session] for subject in source_subjects for session in ("S1", "S2")]
    )
    source_validation = experiment.concatenate(
        [sources[subject]["S3"] for subject in source_subjects]
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_states, source_reports, source_only = {}, {}, {}
    target_test_loader = make_loader(*target["S3"], shuffle=False)
    for architecture in ARCHITECTURES:
        state, report = train_source(
            architecture,
            *source_train,
            *source_validation,
            epochs=args.source_epochs,
            patience=args.source_patience,
            seed=args.seed,
            device=device,
        )
        model = make_model(architecture).to(device)
        model.load_state_dict(state)
        source_states[architecture] = state
        source_reports[architecture] = report
        source_only[architecture] = evaluate(model, target_test_loader, device)
        print(
            f"target={args.target_subject} architecture={architecture} "
            f"source_only={source_only[architecture]['accuracy']:.4f}",
            flush=True,
        )

    records, sampling = [], {}
    for repeat in range(args.repeats):
        choices = experiment.nested_indices(
            target["S1"][1], max(args.budgets), args.seed + 1000 + repeat
        )
        sampling[str(repeat)] = {str(budget): choices[budget] for budget in args.budgets}
        for budget in args.budgets:
            indices = np.asarray(choices[budget], dtype=np.int64)
            calibration = (target["S1"][0][indices], target["S1"][1][indices])
            for architecture in ARCHITECTURES:
                source_learning_rate = source_reports[architecture]["selected_learning_rate"]
                modes = ("scratch", "full") if architecture == "linear" else (
                    "scratch",
                    "head",
                    "full",
                )
                for mode in modes:
                    if mode == "scratch":
                        state = None
                        epochs = args.scratch_epochs
                        learning_rate = source_learning_rate
                        offset = 100
                    elif mode == "head":
                        state = source_states[architecture]
                        epochs = args.adapt_epochs
                        learning_rate = 1e-3
                        offset = 200
                    else:
                        state = source_states[architecture]
                        epochs = args.adapt_epochs
                        learning_rate = 1e-4
                        offset = 300
                    metrics, training = train_target(
                        architecture,
                        calibration,
                        target["S3"],
                        source_state=state,
                        mode=mode,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seed=args.seed + 10000 * repeat + offset,
                        device=device,
                    )
                    records.append(
                        {
                            "repeat": repeat,
                            "budget_per_class": budget,
                            "architecture": architecture,
                            "method": mode,
                            "test": metrics,
                            "training": training,
                        }
                    )
    report = {
        "schema_version": 1,
        "experiment": "flat architecture baselines under strict LOSO",
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "features": "flattened Fp1/Fp2 x four PSD bands (8 dimensions)",
        "architectures": list(ARCHITECTURES),
        "source_training": source_reports,
        "source_only_target_test": source_only,
        "budgets_per_class": args.budgets,
        "repeats": args.repeats,
        "adapt_epochs": args.adapt_epochs,
        "scratch_epochs": args.scratch_epochs,
        "target_session_S2_used": False,
        "nested_sampling_indices": sampling,
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
