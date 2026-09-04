from __future__ import annotations

import numpy as np

from fronteeg_transfer.device_experiment import (
    CONDITIONS,
    ProtocolConfig,
    balanced_condition_order,
    build_schedule,
    signal_quality_summary,
    validate_participant_id,
)


def test_balanced_schedule_separates_calibration_and_test() -> None:
    config = ProtocolConfig(
        calibration_repeats_per_condition=2,
        test_repeats_per_condition=3,
        calibration_block_duration_seconds=60.0,
        test_block_duration_seconds=300.0,
    )
    schedule = build_schedule(config)
    practice = [block for block in schedule if block.phase == "practice"]
    calibration = [block for block in schedule if block.phase == "calibration"]
    test = [block for block in schedule if block.phase == "test"]

    assert [block.condition for block in practice] == list(CONDITIONS)
    assert len(calibration) == 6
    assert len(test) == 9
    counts = {
        condition: [block.condition for block in calibration].count(condition)
        for condition in CONDITIONS
    }
    assert counts == {condition: 2 for condition in CONDITIONS}
    assert {block.duration_seconds for block in calibration} == {60.0}
    assert {block.duration_seconds for block in test} == {300.0}
    assert all(
        left.condition != right.condition
        for left, right in zip(calibration, calibration[1:], strict=False)
    )
    assert {block.block_id for block in calibration}.isdisjoint(
        {block.block_id for block in test}
    )

def test_participant_id_is_pseudonymous_and_path_safe() -> None:
    assert validate_participant_id("sub-self_01") == "sub-self_01"
    for invalid in ("", "non_ascii-测试", "../private", "name with spaces"):
        try:
            validate_participant_id(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid participant ID accepted: {invalid}")


def test_balanced_condition_order_is_reproducible() -> None:
    assert balanced_condition_order(5, 123) == balanced_condition_order(5, 123)


def test_signal_quality_summary_detects_sampling_gaps() -> None:
    timestamps = np.asarray([0.0, 0.004, 0.008, 0.020])
    samples = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    summary = signal_quality_summary(samples, timestamps, sampling_rate=250.0)

    assert summary["samples"] == 4
    assert summary["gaps_over_2x_expected"] == 1
    assert summary["non_finite_values"] == 0


def test_display_font_selection_is_cross_platform() -> None:
    from run_psychopy_device_experiment import select_display_font

    assert select_display_font() == "Arial"
    assert select_display_font("Custom Font") == "Custom Font"


def test_live_event_log_survives_before_final_export(tmp_path) -> None:
    from fronteeg_transfer.device_experiment import EventLog

    destination = tmp_path / "events.tsv"
    events = EventLog(destination)
    events.append(
        lsl_timestamp="1.000000000",
        phase="practice",
        block_id="practice-01",
        condition="Easy",
        event_type="block_start",
    )

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "practice-01" in lines[1]


def test_warmup_validation_rejects_duplicate_channels() -> None:
    from run_psychopy_device_experiment import validate_warmup_signal

    timestamps = np.arange(20, dtype=np.float64) / 250.0
    duplicate = np.column_stack((np.arange(20), np.arange(20)))
    try:
        validate_warmup_signal(duplicate, timestamps, 250.0)
    except RuntimeError as error:
        assert "exactly identical" in str(error)
    else:
        raise AssertionError("duplicate channels were accepted")


def test_formal_protocol_has_practice_split_test_and_nonoverlap_capacity() -> None:
    from fronteeg_transfer.config import load_yaml

    values = load_yaml("configs/device_experiment.yaml")
    config = ProtocolConfig.from_mapping(values["protocol"])
    schedule = build_schedule(config)
    practice = [block for block in schedule if block.phase == "practice"]
    calibration = [block for block in schedule if block.phase == "calibration"]
    test = [block for block in schedule if block.phase == "test"]

    assert len(practice) == 3
    assert len(calibration) == 3
    assert len(test) == 3
    assert {block.duration_seconds for block in calibration} == {64.0}
    assert {block.duration_seconds for block in test} == {300.0}
    assert all(
        calibration[index].condition != test[index].condition for index in range(3)
    )
    windows_per_calibration_block = int((64.0 - 4.0) / 2.0) + 1
    assert (windows_per_calibration_block + 1) // 2 >= 16


def test_smoke_protocol_exercises_artifact_prompts() -> None:
    from fronteeg_transfer.config import load_yaml

    values = load_yaml("configs/device_experiment_smoke.yaml")
    artifact = values["artifact"]
    assert artifact["enabled"] is True
    assert artifact["eyes_open_seconds"] > 0
    assert artifact["paced_blink_seconds"] > 0
    assert artifact["head_motion_seconds"] > 0
    assert 0 < artifact["cue_display_seconds"] < artifact["cue_interval_seconds"]


def test_valid_lsl_timestamps_are_not_modified() -> None:
    from fronteeg_transfer.lsl_device import normalize_lsl_chunk_timestamps

    raw = np.asarray([10.0, 10.004, 10.008])
    local, repaired = normalize_lsl_chunk_timestamps(
        raw,
        time_correction_seconds=0.25,
        sampling_rate=250.0,
        previous_local_timestamp=10.20,
    )

    np.testing.assert_allclose(local, raw + 0.25)
    assert repaired == 0


def test_duplicate_lsl_timestamps_are_reconstructed_but_preserved_by_caller() -> None:
    from fronteeg_transfer.lsl_device import normalize_lsl_chunk_timestamps

    raw = np.asarray([10.0, 10.0, 10.0])
    raw_copy = raw.copy()
    local, repaired = normalize_lsl_chunk_timestamps(
        raw,
        time_correction_seconds=0.5,
        sampling_rate=250.0,
        previous_local_timestamp=None,
    )

    np.testing.assert_array_equal(raw, raw_copy)
    np.testing.assert_allclose(np.diff(local), np.asarray([0.004, 0.004]))
    assert np.isclose(local[-1], 10.5)
    assert repaired == 3


def test_reconstructed_lsl_chunk_stays_after_previous_chunk() -> None:
    from fronteeg_transfer.lsl_device import normalize_lsl_chunk_timestamps

    local, repaired = normalize_lsl_chunk_timestamps(
        [10.0, 10.0],
        time_correction_seconds=0.5,
        sampling_rate=250.0,
        previous_local_timestamp=10.499,
    )

    assert np.isclose(local[0], 10.503)
    assert np.all(np.diff(local) > 0)
    assert repaired == 2


def test_synthetic_recorder_preserves_recovery_and_full_stream(tmp_path) -> None:
    import time

    from fronteeg_transfer.lsl_device import SyntheticEegRecorder

    recovery = tmp_path / "raw_eeg_recovery.csv"
    recorder = SyntheticEegRecorder(250.0, seed=7, recovery_path=recovery)
    recorder.start()
    time.sleep(0.05)
    recorder.stop()
    samples, timestamps, source_timestamps = recorder.snapshot()

    assert len(samples) >= 10
    assert recorder.full_snapshot().shape == samples.shape
    np.testing.assert_array_equal(timestamps, source_timestamps)
    assert len(recovery.read_text(encoding="utf-8").splitlines()) == len(samples) + 1


def test_recorded_session_validator_accepts_complete_raw_bundle(tmp_path, monkeypatch) -> None:
    import csv
    import json
    import sys

    from validate_recorded_session import main as validate_session

    session = tmp_path / "session"
    session.mkdir()
    metadata = {
        "status": "completed",
        "workload_completed": True,
        "artifact_protocol_enabled": False,
        "schedule": [
            {
                "block_id": "test-01",
                "phase": "test",
                "condition": "Easy",
                "duration_seconds": 4.0,
            }
        ],
    }
    (session / "session_plan.json").write_text("{}", encoding="utf-8")
    (session / "session_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (session / "events.tsv").write_text(
        "lsl_timestamp\tphase\tblock_id\tcondition\tevent_type\n"
        "0.500000000\ttest\ttest-01\tEasy\tblock_start\n"
        "4.500000000\ttest\ttest-01\tEasy\tblock_end\n",
        encoding="utf-8",
    )
    timestamps = np.arange(0.0, 5.004, 0.004)
    samples = np.column_stack((np.sin(timestamps), np.cos(timestamps))).astype(np.float32)
    np.savez_compressed(
        session / "raw_eeg.npz",
        samples=samples,
        all_stream_samples=samples,
        local_timestamps=timestamps,
        lsl_timestamps=timestamps,
        sampling_rate=np.asarray(250.0),
    )
    with (session / "raw_eeg_recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_timestamp", "lsl_timestamp", "channel_0", "channel_1"])
        writer.writerows(
            [timestamp, timestamp, *sample]
            for timestamp, sample in zip(timestamps, samples, strict=True)
        )
    monkeypatch.setattr(sys, "argv", ["validate_recorded_session.py", "--session", str(session)])

    validate_session()
