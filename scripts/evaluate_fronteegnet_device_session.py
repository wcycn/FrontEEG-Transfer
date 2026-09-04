#!/usr/bin/env python3
"""Evaluate a frozen FrontEEGNet checkpoint on a complete or interrupted device session."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_matb_eegnet_subject import normalize_windows
from scipy.signal import butter, sosfiltfilt

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig
from fronteeg_transfer.device_session import (
    CONDITION_TO_LABEL,
    bandpass_rms,
    discover_blocks,
    extract_block_windows,
    prediction_summary,
    read_events,
    robust_rms_threshold,
    signal_quality_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--source-checkpoint", default="checkpoints/final_fronteegnet_source.pt"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include an interrupted final block only as a labelled diagnostic segment.",
    )
    return parser.parse_args()


def filter_windows(windows: np.ndarray, sampling_rate: float) -> np.ndarray:
    if not len(windows):
        return np.asarray(windows, dtype=np.float32)
    sos = butter(4, (1.0, 40.0), btype="bandpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, windows, axis=-1).astype(np.float32)


def predict(
    model: EEGNet,
    windows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            values = torch.from_numpy(windows[start : start + batch_size]).to(device)
            logits = model(values)
            parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts)


def per_block_summary(
    records: list[dict[str, Any]], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    block_ids = sorted({str(record["block_id"]) for record in records})
    for block_id in block_ids:
        indices = np.asarray(
            [index for index, record in enumerate(records) if record["block_id"] == block_id]
        )
        block = next(record for record in records if record["block_id"] == block_id)
        result[block_id] = {
            "condition": block["condition"],
            "block_status": block["block_status"],
            **prediction_summary(labels[indices], probabilities[indices]),
        }
    return result


def main() -> None:
    args = parse_args()
    session = Path(args.session).expanduser().resolve()
    metadata = json.loads((session / "session_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("simulated_eeg"):
        raise ValueError("this evaluator requires a real device recording")
    with np.load(session / "raw_eeg.npz", allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=np.float64)
        local_timestamps = np.asarray(archive["local_timestamps"], dtype=np.float64)
        sampling_rate = float(np.asarray(archive["sampling_rate"]).item())
    if metadata.get("input_unit") == "volts":
        samples *= 1e6
    elif metadata.get("input_unit") != "microvolts":
        raise ValueError(f"unknown configured input unit: {metadata.get('input_unit')}")
    if len(samples) < 2 or not np.all(np.diff(local_timestamps) > 0):
        raise ValueError("device timestamps must be strictly increasing")

    checkpoint_path = Path(args.source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("classes", -1)) != 3:
        raise ValueError("FrontEEGNet checkpoint must have three output classes")
    if checkpoint.get("class_names") != ["Easy", "Medium", "Difficult"]:
        raise ValueError("FrontEEGNet checkpoint has incompatible class names")
    model_config = EEGNetConfig(**checkpoint["model_config"])
    expected_rate = float(checkpoint["input"]["sampling_rate_hz"])
    if abs(sampling_rate - expected_rate) > 1e-6:
        raise ValueError(
            f"device sampling rate {sampling_rate} Hz differs from model rate {expected_rate} Hz"
        )
    window_seconds = float(checkpoint["input"]["window_seconds"])
    stride_seconds = float(checkpoint["input"]["stride_seconds"])
    events = read_events(session / "events.tsv")
    phase_blocks = {
        phase: discover_blocks(
            events,
            phase=phase,
            recording_end=float(local_timestamps[-1]),
            include_partial=args.include_partial,
        )
        for phase in ("calibration", "test")
    }
    phase_data = {
        phase: extract_block_windows(
            samples,
            local_timestamps,
            blocks,
            sampling_rate=sampling_rate,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        for phase, blocks in phase_blocks.items()
    }
    calibration_windows = phase_data["calibration"][0]
    test_windows, test_labels, test_records = phase_data["test"]
    if not len(test_windows):
        raise ValueError("no complete or explicitly included partial test windows")

    calibration_rms = bandpass_rms(calibration_windows, sampling_rate)
    test_rms = bandpass_rms(test_windows, sampling_rate)
    quality_rule = robust_rms_threshold(calibration_rms)
    threshold = np.asarray(quality_rule["threshold_input_scale"])
    quality_accepted = ~np.any(test_rms > threshold, axis=1)
    model_windows = normalize_windows(filter_windows(test_windows, sampling_rate))
    if model_windows.shape[1:] != (model_config.channels, model_config.samples):
        raise ValueError(
            f"device windows {model_windows.shape[1:]} do not match model "
            f"{(model_config.channels, model_config.samples)}"
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EEGNet(model_config, classes=3).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities = predict(model, model_windows, device=device)

    complete_conditions = {
        block["condition"] for block in phase_blocks["test"] if block["status"] == "complete"
    }
    primary_valid = (
        metadata.get("status") in {"completed", "aborted_after_workload"}
        and bool(metadata.get("workload_completed", metadata.get("status") == "completed"))
        and complete_conditions == set(CONDITION_TO_LABEL)
    )
    label_to_condition = {label: name for name, label in CONDITION_TO_LABEL.items()}
    report = {
        "schema_version": 1,
        "experiment": "frozen FrontEEGNet real-device evaluation",
        "session_id": metadata.get("session_id"),
        "participant_pseudonym": metadata.get("participant_pseudonym"),
        "session_status": metadata.get("status"),
        "checkpoint": str(checkpoint_path),
        "model_config": model_config.to_dict(),
        "preprocessing": {
            "timestamp_interpolation": "uniform 250 Hz grid inside each task block",
            "bandpass_filter": "fourth-order zero-phase Butterworth, 1-40 Hz",
            "normalization": "per-window per-channel z-score",
            "target_data_used_to_update_model": False,
            "calibration_used_only_for_quality_threshold": True,
        },
        "recording": signal_quality_summary(samples, local_timestamps, sampling_rate),
        "test_blocks": phase_blocks["test"],
        "test_window_class_counts": dict(
            Counter(label_to_condition[int(label)] for label in test_labels)
        ),
        "quality_rule": quality_rule,
        "quality_rejected_test_windows": int(np.sum(~quality_accepted)),
        "all_recorded_test_windows": prediction_summary(test_labels, probabilities),
        "quality_accepted_test_windows": (
            prediction_summary(
                test_labels[quality_accepted], probabilities[quality_accepted]
            )
            if np.any(quality_accepted)
            else None
        ),
        "per_block": per_block_summary(test_records, test_labels, probabilities),
        "primary_three_class_device_metric_valid": primary_valid,
        "primary_metric_exclusion_reason": (
            None
            if primary_valid
            else "The test phase lacks complete blocks for all three workload classes."
        ),
        "interpretation_boundary": (
            "Metrics from incomplete test phases are diagnostic and are not the planned "
            "three-class real-device result. The device paradigm is a lightweight MATB "
            "implementation and differs from the public laboratory protocol."
        ),
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else session / "fronteegnet_evaluation.json"
    )
    write_json(output, report)
    print(
        f"output={output} windows={len(test_labels)} "
        f"primary_valid={primary_valid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
