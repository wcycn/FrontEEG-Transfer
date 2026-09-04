#!/usr/bin/env python3
"""Evaluate the verified MATB GNN+OT baseline after restricting it to Fp1/Fp2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import run_matb_ot_gnn_paper as baseline
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--minimum-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def select_fp1_fp2(loaded: tuple) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    features, labels, channel_names, preprocessing = loaded
    lookup = {name.casefold(): index for index, name in enumerate(channel_names)}
    missing = [name for name in ("fp1", "fp2") if name not in lookup]
    if missing:
        raise ValueError(f"missing required channels: {missing}")
    indices = [lookup["fp1"], lookup["fp2"]]
    return (
        features[:, indices, :],
        labels,
        [channel_names[index] for index in indices],
        preprocessing,
    )


def minmax_float64(sessions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    minimum = sessions["S1"].min(axis=0)
    span = sessions["S1"].max(axis=0) - minimum
    if np.any(span <= 0):
        raise ValueError("constant S1 channel-band feature cannot be MinMax scaled")
    return {name: (values - minimum) / span for name, values in sessions.items()}


def fp1_fp2_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Use the only non-self spatial connection available for two nodes."""
    return (
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.ones(2, dtype=torch.float32),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    loaded = {
        session: select_fp1_fp2(baseline.load_session(root, args.subject, session))
        for session in ("S1", "S2", "S3")
    }
    channel_names = loaded["S1"][2]
    if any(values[2] != channel_names for values in loaded.values()):
        raise ValueError("channel order differs across sessions")
    labels = {session: loaded[session][1] for session in loaded}
    features = minmax_float64({session: loaded[session][0] for session in loaded})
    validation_ot, validation_reg = baseline.backward_sinkhorn(features["S2"], features["S1"])
    test_ot, test_reg = baseline.backward_sinkhorn(features["S3"], features["S1"])
    variants = {
        "plain": features,
        "sinkhorn_ot": {"S1": features["S1"], "S2": validation_ot, "S3": test_ot},
    }
    edge_index, edge_weight = fp1_fp2_graph()
    configs = baseline.sample_configs(args.trials, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for variant_name, variant_features in variants.items():
        print(f"variant={variant_name}", flush=True)
        split_data = {
            session: baseline.make_dataset(values, labels[session], edge_index, edge_weight)
            for session, values in variant_features.items()
        }
        trials = []
        for index, config in enumerate(configs):
            trial = baseline.train_one(
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
        results[variant_name] = {
            "selected": min(trials, key=lambda trial: trial["heuristic_loss"]),
            "trials": trials,
        }

    plain_test = results["plain"]["selected"]["test"]["accuracy"]
    ot_test = results["sinkhorn_ot"]["selected"]["test"]["accuracy"]
    summary = {
        "subject": args.subject,
        "experiment": "Fp1/Fp2 ablation of the verified full-channel GNN+OT baseline",
        "protocol": "within-subject cross-session S1 train / S2 validation / S3 test",
        "target_labels_used_for_adaptation": False,
        "target_signal_scope": "entire S2/S3 session (transductive backward Sinkhorn)",
        "channels": channel_names,
        "node_features": 4,
        "feature_dimensions": 8,
        "graph": "bidirectional Fp1 <-> Fp2, no explicit self-loops",
        "sinkhorn_initial_reg_e": 1e-3,
        "sinkhorn_selected_reg_e": {"S2": validation_reg, "S3": test_reg},
        "search_trials": args.trials,
        "epochs": args.epochs,
        "known_preprocessing_difference": "no EEGLAB Picard/ICLabel ICA rejection",
        "results": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "plain_test": plain_test,
                "ot_test": ot_test,
                "ot_gain": ot_test - plain_test,
                "selected_reg_e": {"S2": validation_reg, "S3": test_reg},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
