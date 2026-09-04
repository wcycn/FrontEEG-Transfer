#!/usr/bin/env python3
"""Paper-aligned COG-BCI cross-session GNN plus backward Sinkhorn OT.

This is an engineering reimplementation of the public COG_BCIg5 notebook. It
keeps the official protocol and parameter ranges while making the number of
search trials configurable so a reproduction can be validated before running
the paper's 150-trial search.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import ot
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_max_pool


@dataclass(frozen=True)
class ModelConfig:
    graph_widths: tuple[int, ...]
    linear_widths: tuple[int, ...]
    activation: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--minimum-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_configs(count: int, seed: int) -> list[ModelConfig]:
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(count):
        graph_layers = int(rng.integers(1, 5))
        linear_layers = int(rng.integers(2, 6))
        configs.append(
            ModelConfig(
                graph_widths=tuple(int(value) for value in rng.integers(4, 33, size=graph_layers)),
                linear_widths=tuple(
                    int(value) for value in rng.integers(8, 65, size=linear_layers - 1)
                ),
                activation=str(rng.choice(("gelu", "relu", "tanh"))),
                batch_size=int(rng.integers(16, 65)),
                learning_rate=float(10 ** rng.uniform(-4.0, -2.0)),
                weight_decay=float(10 ** rng.uniform(-4.0, -1.0)),
                dropout=float(10 ** rng.uniform(np.log10(1e-2), np.log10(0.5))),
            )
        )
    return configs


def load_session(root: Path, subject: str, session: str):
    path = root / f"{subject}_ses-{session}_matb_psd.npz"
    with np.load(path, allow_pickle=False) as values:
        return (
            np.asarray(values["features"], dtype=np.float64),
            np.asarray(values["label"], dtype=np.int64),
            [str(value) for value in values["channel_names"]],
            str(values["preprocessing"]),
        )


def minmax_from_train(sessions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    minimum = sessions["S1"].min(axis=0)
    span = sessions["S1"].max(axis=0) - minimum
    if np.any(span <= 0):
        raise ValueError("constant S1 channel-band feature cannot be MinMax scaled")
    return {
        name: ((values - minimum) / span).astype(np.float32) for name, values in sessions.items()
    }


def backward_sinkhorn(source: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float]:
    source_flat = source.reshape(len(source), -1)
    train_flat = train.reshape(len(train), -1)
    regularization = 1e-3
    while regularization <= 16.384:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                transport = ot.da.SinkhornTransport(reg_e=regularization, norm="max")
                transport.fit(Xs=source_flat, Xt=train_flat)
                transformed = transport.transform(Xs=source_flat)
            if np.isfinite(transformed).all():
                return transformed.reshape(source.shape).astype(np.float32), regularization
        except (FloatingPointError, RuntimeWarning, UserWarning):
            pass
        regularization *= 2.0
    raise RuntimeError("Sinkhorn transport did not converge")


def graph(channel_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    montage = mne.channels.make_standard_montage("standard_1020")
    canonical = {
        name.casefold(): value for name, value in montage.get_positions()["ch_pos"].items()
    }
    coordinates = np.stack([canonical[name.casefold()] for name in channel_names])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    weights = np.divide(1.0, distances, out=np.zeros_like(distances), where=distances > 0)
    threshold = np.quantile(weights, 0.94)
    sources, targets = np.where(weights > threshold)
    edge_weights = weights[sources, targets]
    edge_weights /= edge_weights.max()
    if len(np.unique(np.concatenate((sources, targets)))) != len(channel_names):
        raise ValueError("official 6% graph threshold produced isolated channels")
    return (
        torch.tensor(np.stack((sources, targets)), dtype=torch.long),
        torch.tensor(edge_weights, dtype=torch.float32),
    )


def make_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> list[Data]:
    return [
        Data(
            x=torch.tensor(values, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor(label, dtype=torch.long),
        )
        for values, label in zip(features, labels, strict=True)
    ]


class GcnClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        graph_dimensions = (4, *config.graph_widths)
        self.graph_layers = nn.ModuleList(
            GraphConv(input_width, output_width)
            for input_width, output_width in zip(
                graph_dimensions[:-1], graph_dimensions[1:], strict=True
            )
        )
        linear_dimensions = (graph_dimensions[-1], *config.linear_widths, 3)
        self.linear_layers = nn.ModuleList(
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(
                linear_dimensions[:-1], linear_dimensions[1:], strict=True
            )
        )
        self.activation = config.activation
        self.dropout = config.dropout

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return functional.relu(values)
        if self.activation == "tanh":
            return torch.tanh(values)
        return functional.gelu(values)

    def forward(self, batch: Data) -> torch.Tensor:
        values = batch.x
        for layer in self.graph_layers:
            values = self.activate(layer(values, batch.edge_index, batch.edge_weight))
        values = global_max_pool(values, batch.batch)
        values = functional.dropout(values, p=self.dropout, training=self.training)
        for layer in self.linear_layers[:-1]:
            values = self.activate(layer(values))
        return self.linear_layers[-1](values)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            total_loss += functional.cross_entropy(logits, batch.y).item() * batch.num_graphs
            labels.extend(batch.y.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": float(total_loss / len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def train_one(
    split_data: dict[str, list[Data]],
    config: ModelConfig,
    *,
    epochs: int,
    minimum_checkpoint_epoch: int,
    seed: int,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    train_loader = DataLoader(split_data["S1"], batch_size=config.batch_size, shuffle=True)
    loaders = {
        session: DataLoader(values, batch_size=config.batch_size, shuffle=False)
        for session, values in split_data.items()
    }
    model = GcnClassifier(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.99, patience=10, min_lr=1e-5
    )
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, loaders["S1"], device)
        validation_metrics = evaluate(model, loaders["S2"], device)
        scheduler.step(validation_metrics["loss"])
        gap = validation_metrics["loss"] - train_metrics["loss"]
        score = 0.75 * validation_metrics["loss"] + 0.25 * (gap if gap >= 0 else 1e6)
        if epoch > minimum_checkpoint_epoch and score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no checkpoint: epochs must exceed minimum-checkpoint-epoch")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "heuristic_loss": float(best_score),
        "train": evaluate(model, loaders["S1"], device),
        "validation": evaluate(model, loaders["S2"], device),
        "test": evaluate(model, loaders["S3"], device),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    loaded = {session: load_session(root, args.subject, session) for session in ("S1", "S2", "S3")}
    channel_names = loaded["S1"][2]
    if len(channel_names) != 61:
        raise ValueError(f"paper-aligned input requires 61 channels, found {len(channel_names)}")
    if any(values[2] != channel_names for values in loaded.values()):
        raise ValueError("channel order differs across sessions")
    labels = {session: loaded[session][1] for session in loaded}
    features = minmax_from_train({session: loaded[session][0] for session in loaded})
    edge_index, edge_weight = graph(channel_names)
    validation_ot, validation_reg = backward_sinkhorn(features["S2"], features["S1"])
    test_ot, test_reg = backward_sinkhorn(features["S3"], features["S1"])
    variants = {
        "plain": features,
        "sinkhorn_ot": {"S1": features["S1"], "S2": validation_ot, "S3": test_ot},
    }
    configs = sample_configs(args.trials, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for variant_name, variant_features in variants.items():
        print(f"variant={variant_name}", flush=True)
        split_data = {
            session: make_dataset(values, labels[session], edge_index, edge_weight)
            for session, values in variant_features.items()
        }
        trials = []
        for index, config in enumerate(configs):
            trial = train_one(
                split_data,
                config,
                epochs=args.epochs,
                minimum_checkpoint_epoch=args.minimum_checkpoint_epoch,
                seed=args.seed + index,
                device=device,
            )
            trial["config"] = asdict(config)
            trials.append(trial)
            print(
                f"  trial={index} epoch={trial['best_epoch']} "
                f"heuristic={trial['heuristic_loss']:.4f} "
                f"val={trial['validation']['accuracy']:.4f} "
                f"test={trial['test']['accuracy']:.4f}",
                flush=True,
            )
        selected = min(trials, key=lambda trial: trial["heuristic_loss"])
        results[variant_name] = {"selected": selected, "trials": trials}

    summary = {
        "subject": args.subject,
        "protocol": "within-subject cross-session S1 train / S2 validation / S3 test",
        "target_labels_used_for_adaptation": False,
        "target_signal_scope": "entire S2/S3 session (transductive backward Sinkhorn)",
        "paper_alignment": {
            "channels": 61,
            "node_features": 4,
            "feature_dimensions": 244,
            "global_pooling": "max",
            "self_loops": False,
            "strongest_edges_percent": 6,
            "sinkhorn_initial_reg_e": 1e-3,
            "sinkhorn_selected_reg_e": {"S2": validation_reg, "S3": test_reg},
            "search_trials": args.trials,
            "paper_search_trials": 150,
            "known_difference": "no EEGLAB Picard/ICLabel ICA rejection",
        },
        "preprocessing": loaded["S1"][3],
        "results": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "plain_test": results["plain"]["selected"]["test"]["accuracy"],
                "ot_test": results["sinkhorn_ot"]["selected"]["test"]["accuracy"],
                "ot_gain": results["sinkhorn_ot"]["selected"]["test"]["accuracy"]
                - results["plain"]["selected"]["test"]["accuracy"],
                "selected_reg_e": {"S2": validation_reg, "S3": test_reg},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
