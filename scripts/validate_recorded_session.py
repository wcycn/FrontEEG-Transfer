#!/usr/bin/env python3
"""Validate one recorded device session before disconnecting the EEG hardware."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--maximum-gap-seconds", type=float, default=0.1)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=1.0)
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_blocks(
    metadata: dict[str, Any], events: list[dict[str, str]], tolerance: float
) -> tuple[float, float]:
    starts: dict[str, list[dict[str, str]]] = {}
    ends: dict[str, list[dict[str, str]]] = {}
    for event in events:
        target = None
        if event["event_type"] == "block_start":
            target = starts
        elif event["event_type"] == "block_end":
            target = ends
        if target is not None:
            target.setdefault(event["block_id"], []).append(event)

    workload_starts: list[float] = []
    workload_ends: list[float] = []
    for block in metadata["schedule"]:
        block_id = block["block_id"]
        require(len(starts.get(block_id, [])) == 1, f"missing or duplicate start: {block_id}")
        require(len(ends.get(block_id, [])) == 1, f"missing or duplicate end: {block_id}")
        start = float(starts[block_id][0]["lsl_timestamp"])
        end = float(ends[block_id][0]["lsl_timestamp"])
        actual = end - start
        planned = float(block["duration_seconds"])
        require(
            abs(actual - planned) <= tolerance,
            f"block {block_id} duration {actual:.3f}s differs from {planned:.3f}s",
        )
        workload_starts.append(start)
        workload_ends.append(end)

    if metadata.get("artifact_protocol_enabled") and metadata["status"] == "completed":
        for block_id in ("eyes_open", "paced_blink", "head_motion"):
            require(len(starts.get(block_id, [])) == 1, f"missing artifact start: {block_id}")
            require(len(ends.get(block_id, [])) == 1, f"missing artifact end: {block_id}")
    return min(workload_starts), max(workload_ends)


def main() -> None:
    args = parse_args()
    session = Path(args.session).expanduser().resolve()
    required_files = (
        "session_plan.json",
        "session_metadata.json",
        "events.tsv",
        "raw_eeg.npz",
        "raw_eeg_recovery.csv",
    )
    for name in required_files:
        require((session / name).is_file(), f"missing required file: {name}")

    metadata = json.loads((session / "session_metadata.json").read_text(encoding="utf-8"))
    require(bool(metadata.get("workload_completed")), "workload task was not completed")
    require(
        metadata.get("status") in {"completed", "aborted_after_workload"},
        "invalid session status: " + str(metadata.get("status")),
    )
    events = load_events(session / "events.tsv")
    first_block, last_block = validate_blocks(
        metadata, events, args.duration_tolerance_seconds
    )

    with np.load(session / "raw_eeg.npz", allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=np.float64)
        all_samples = np.asarray(archive["all_stream_samples"], dtype=np.float64)
        timestamps = np.asarray(archive["local_timestamps"], dtype=np.float64)
        source_timestamps = np.asarray(archive["lsl_timestamps"], dtype=np.float64)
        sampling_rate = float(np.asarray(archive["sampling_rate"]).item())

    require(samples.ndim == 2 and samples.shape[1] == 2, f"bad selected shape: {samples.shape}")
    require(all_samples.ndim == 2, f"bad complete stream shape: {all_samples.shape}")
    require(
        len(samples) == len(all_samples) == len(timestamps) == len(source_timestamps),
        "raw arrays have inconsistent lengths",
    )
    require(len(samples) > 10, "too few EEG samples")
    require(np.isfinite(samples).all() and np.isfinite(all_samples).all(), "non-finite EEG values")
    gaps = np.diff(timestamps)
    require(np.all(gaps > 0), "local timestamps are not strictly increasing")
    require(float(gaps.max()) <= args.maximum_gap_seconds, f"maximum EEG gap is {gaps.max():.6f}s")
    observed_rate = 1.0 / float(np.median(gaps))
    require(
        abs(observed_rate - sampling_rate) / sampling_rate <= 0.20,
        f"observed rate {observed_rate:.2f} differs from declared {sampling_rate:.2f}",
    )
    require(timestamps[0] <= first_block, "EEG begins after the first workload block")
    require(timestamps[-1] >= last_block, "EEG ends before the last workload block")

    with (session / "raw_eeg_recovery.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        recovery_rows = sum(1 for _ in csv.reader(handle)) - 1
    require(
        recovery_rows == len(samples),
        f"recovery CSV has {recovery_rows} rows but NPZ has {len(samples)} samples",
    )

    report = {
        "session": str(session),
        "status": metadata["status"],
        "samples": len(samples),
        "all_stream_channels": all_samples.shape[1],
        "declared_sampling_rate_hz": sampling_rate,
        "observed_sampling_rate_hz": observed_rate,
        "maximum_gap_seconds": float(gaps.max()),
        "selected_channel_std": samples.std(axis=0).tolist(),
        "selected_channel_peak_to_peak": np.ptp(samples, axis=0).tolist(),
        "workload_blocks": len(metadata["schedule"]),
        "artifact_complete": metadata["status"] == "completed",
        "result": "PASS",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
