from __future__ import annotations

import numpy as np

from fronteeg_transfer.device_session import (
    discover_blocks,
    extract_phase_windows,
    robust_rms_threshold,
)


def test_marks_interrupted_block_as_partial_only_when_requested() -> None:
    events = [
        {
            "lsl_timestamp": "10.0",
            "phase": "test",
            "block_id": "test-01",
            "condition": "Easy",
            "event_type": "block_start",
        },
        {
            "lsl_timestamp": "14.0",
            "phase": "system",
            "block_id": "",
            "condition": "",
            "event_type": "abort",
        },
    ]

    assert discover_blocks(
        events, phase="test", recording_end=14.2, include_partial=False
    ) == []
    blocks = discover_blocks(
        events, phase="test", recording_end=14.2, include_partial=True
    )
    assert len(blocks) == 1
    assert blocks[0]["status"] == "partial"
    assert blocks[0]["duration_seconds"] == 4.0


def test_rms_threshold_rejects_extreme_window() -> None:
    calibration = np.asarray([[10.0, 12.0], [11.0, 13.0], [9.0, 11.0]])
    result = robust_rms_threshold(calibration)
    threshold = np.asarray(result["threshold_input_scale"])

    assert np.all(threshold > calibration.max(axis=0))
    assert np.any(np.asarray([1000.0, 1000.0]) > threshold)


def test_extract_windows_stays_inside_complete_blocks() -> None:
    sampling_rate = 100.0
    timestamps = np.arange(0.0, 24.0, 1.0 / sampling_rate)
    samples = np.stack((timestamps, -timestamps), axis=1)
    events = [
        {
            "lsl_timestamp": "1.0",
            "phase": "calibration",
            "block_id": "calibration-01",
            "condition": "Easy",
            "event_type": "block_start",
        },
        {
            "lsl_timestamp": "11.0",
            "phase": "calibration",
            "block_id": "calibration-01",
            "condition": "Easy",
            "event_type": "block_end",
        },
    ]
    windows, labels, records = extract_phase_windows(
        samples,
        timestamps,
        events,
        sampling_rate=sampling_rate,
        phase="calibration",
    )

    assert windows.shape == (4, 2, 400)
    assert labels.tolist() == [0, 0, 0, 0]
    assert [record["start_lsl_timestamp"] for record in records] == [1.0, 3.0, 5.0, 7.0]
    assert np.isclose(windows[-1, 0, -1], 10.99)
