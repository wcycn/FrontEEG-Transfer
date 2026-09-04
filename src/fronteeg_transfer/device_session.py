"""Utilities for auditing saved two-channel wearable EEG sessions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

CONDITION_TO_LABEL = {"Easy": 0, "Medium": 1, "Difficult": 2}


def read_events(path: str | Path) -> list[dict[str, str]]:
    """Read the incrementally saved event table."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: str | Path, values: Any) -> None:
    """Write a human-readable JSON report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_phase_windows(
    samples: np.ndarray,
    timestamps: np.ndarray,
    events: list[dict[str, str]],
    *,
    sampling_rate: float,
    phase: str,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    maximum_gap_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Interpolate fixed windows that remain inside complete task blocks."""
    values = np.asarray(samples, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected samples [N, 2], found {values.shape}")
    if times.ndim != 1 or len(times) != len(values):
        raise ValueError("timestamps must have one value per sample")
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("timestamps must be strictly increasing")
    if sampling_rate <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("sampling rate, window and stride must be positive")

    starts: dict[str, dict[str, str]] = {}
    ends: dict[str, dict[str, str]] = {}
    for event in events:
        if event["phase"] != phase:
            continue
        if event["event_type"] == "block_start":
            starts[event["block_id"]] = event
        elif event["event_type"] == "block_end":
            ends[event["block_id"]] = event
    if not starts:
        raise ValueError(f"no block_start events found for phase={phase}")
    if set(starts) != set(ends):
        raise ValueError(f"incomplete block boundaries for phase={phase}")

    sample_count = int(round(window_seconds * sampling_rate))
    windows: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    for block_id, start_event in starts.items():
        condition = start_event["condition"]
        if condition not in CONDITION_TO_LABEL:
            raise ValueError(f"unknown condition in {block_id}: {condition}")
        block_start = float(start_event["lsl_timestamp"])
        block_end = float(ends[block_id]["lsl_timestamp"])
        candidate_start = block_start
        window_index = 0
        while candidate_start + window_seconds <= block_end + 1e-9:
            grid = candidate_start + np.arange(sample_count, dtype=np.float64) / sampling_rate
            left = int(np.searchsorted(times, grid[0], side="right") - 1)
            right = int(np.searchsorted(times, grid[-1], side="left") + 1)
            if left >= 0 and right < len(times):
                local_times = times[left : right + 1]
                if np.diff(local_times).max(initial=0.0) <= maximum_gap_seconds:
                    window = np.stack(
                        [np.interp(grid, times, values[:, channel]) for channel in range(2)]
                    )
                    windows.append(window.astype(np.float32))
                    labels.append(CONDITION_TO_LABEL[condition])
                    records.append(
                        {
                            "phase": phase,
                            "block_id": block_id,
                            "condition": condition,
                            "window_index": window_index,
                            "start_lsl_timestamp": candidate_start,
                            "end_lsl_timestamp": candidate_start + window_seconds,
                        }
                    )
            candidate_start += stride_seconds
            window_index += 1
    if not windows:
        raise ValueError(f"no valid windows extracted for phase={phase}")
    return np.stack(windows), np.asarray(labels, dtype=np.int64), records


def signal_quality_summary(
    samples: np.ndarray, timestamps: np.ndarray, sampling_rate: float
) -> dict[str, Any]:
    """Summarize continuity and amplitude in the configured input scale."""
    values = np.asarray(samples, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    gaps = np.diff(times)
    expected_interval = 1.0 / sampling_rate
    return {
        "samples": int(len(values)),
        "duration_seconds": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "estimated_sampling_rate_hz": (
            float(1.0 / np.median(gaps)) if len(gaps) and np.median(gaps) > 0 else math.nan
        ),
        "gaps_over_2x_expected": int(np.sum(gaps > 2.0 * expected_interval)),
        "maximum_gap_seconds": float(gaps.max(initial=0.0)),
        "channel_mean_input_scale": values.mean(axis=0).tolist(),
        "channel_std_input_scale": values.std(axis=0).tolist(),
        "channel_peak_to_peak_input_scale": np.ptp(values, axis=0).tolist(),
        "non_finite_values": int(np.size(values) - np.isfinite(values).sum()),
    }


def discover_blocks(
    events: list[dict[str, str]],
    *,
    phase: str,
    recording_end: float,
    include_partial: bool,
) -> list[dict[str, Any]]:
    """Return time-ordered complete blocks and optionally one interrupted block."""
    starts: dict[str, dict[str, str]] = {}
    ends: dict[str, dict[str, str]] = {}
    abort_times: list[float] = []
    for event in events:
        if event["event_type"] == "abort":
            abort_times.append(float(event["lsl_timestamp"]))
        if event["phase"] != phase:
            continue
        if event["event_type"] == "block_start":
            starts[event["block_id"]] = event
        elif event["event_type"] == "block_end":
            ends[event["block_id"]] = event

    interruption_time = min([recording_end, *abort_times]) if abort_times else recording_end
    blocks: list[dict[str, Any]] = []
    for block_id, start_event in starts.items():
        start = float(start_event["lsl_timestamp"])
        if block_id in ends:
            end = float(ends[block_id]["lsl_timestamp"])
            status = "complete"
        elif include_partial and interruption_time > start:
            end = interruption_time
            status = "partial"
        else:
            continue
        blocks.append(
            {
                "phase": phase,
                "block_id": block_id,
                "condition": start_event["condition"],
                "start_timestamp": start,
                "end_timestamp": end,
                "duration_seconds": end - start,
                "status": status,
            }
        )
    return sorted(blocks, key=lambda block: block["start_timestamp"])


def extract_block_windows(
    samples: np.ndarray,
    timestamps: np.ndarray,
    blocks: list[dict[str, Any]],
    *,
    sampling_rate: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract windows independently within each complete or declared partial block."""
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for block in blocks:
        synthetic_events = [
            {
                "lsl_timestamp": str(block["start_timestamp"]),
                "phase": block["phase"],
                "block_id": block["block_id"],
                "condition": block["condition"],
                "event_type": "block_start",
            },
            {
                "lsl_timestamp": str(block["end_timestamp"]),
                "phase": block["phase"],
                "block_id": block["block_id"],
                "condition": block["condition"],
                "event_type": "block_end",
            },
        ]
        try:
            block_windows, block_labels, block_records = extract_phase_windows(
                samples,
                timestamps,
                synthetic_events,
                sampling_rate=sampling_rate,
                phase=block["phase"],
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )
        except ValueError as error:
            if "no valid windows extracted" in str(error):
                continue
            raise
        for record in block_records:
            record["block_status"] = block["status"]
        windows.append(block_windows)
        labels.append(block_labels)
        records.extend(block_records)
    if not windows:
        sample_count = int(round(window_seconds * sampling_rate))
        return (
            np.empty((0, 2, sample_count), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            [],
        )
    return np.concatenate(windows), np.concatenate(labels), records


def bandpass_rms(windows: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Return per-window, per-channel 1--40 Hz RMS in the configured input scale."""
    if not len(windows):
        return np.empty((0, 2), dtype=np.float64)
    sos = butter(4, (1.0, 40.0), btype="bandpass", fs=sampling_rate, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(windows, dtype=np.float64), axis=-1)
    return np.sqrt(np.mean(np.square(filtered), axis=-1))


def robust_rms_threshold(calibration_rms: np.ndarray) -> dict[str, Any]:
    """Fit a label-independent high-amplitude rule on calibration windows only."""
    values = np.asarray(calibration_rms, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("calibration_rms must have shape [N, 2]")
    log_values = np.log(np.maximum(values, np.finfo(np.float64).tiny))
    median = np.median(log_values, axis=0)
    mad = np.median(np.abs(log_values - median), axis=0)
    threshold = np.exp(median + 6.0 * 1.4826 * mad)
    return {
        "method": "calibration log-RMS median + 6 scaled MAD",
        "threshold_input_scale": threshold.tolist(),
        "calibration_median_rms_input_scale": np.median(values, axis=0).tolist(),
        "calibration_windows": int(len(values)),
    }


def prediction_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Summarize only the classes actually present in a complete or partial recording."""
    predictions = probabilities.argmax(axis=1)
    present = sorted(int(label) for label in np.unique(labels))
    label_to_condition = {label: condition for condition, label in CONDITION_TO_LABEL.items()}
    recalls = {
        label_to_condition[label]: float(np.mean(predictions[labels == label] == label))
        for label in present
    }
    return {
        "examples": int(len(labels)),
        "true_class_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
        "predicted_class_counts": np.bincount(predictions, minlength=3).astype(int).tolist(),
        "observed_class_accuracy": float(np.mean(predictions == labels)),
        "observed_class_balanced_accuracy": float(np.mean(list(recalls.values()))),
        "class_recall": recalls,
        "mean_probability": probabilities.mean(axis=0).tolist(),
    }
