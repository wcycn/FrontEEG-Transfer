from __future__ import annotations

import numpy as np

from fronteeg_transfer.artifact_stress import add_blink_artifacts, add_motion_artifacts


def test_artifact_perturbations_are_deterministic_and_non_mutating() -> None:
    eeg = np.zeros((3, 2, 1000), dtype=np.float32)
    blink_a = add_blink_artifacts(eeg, sampling_rate=250.0, amplitude_uv=100.0, seed=7)
    blink_b = add_blink_artifacts(eeg, sampling_rate=250.0, amplitude_uv=100.0, seed=7)
    motion = add_motion_artifacts(eeg, sampling_rate=250.0, amplitude_uv=50.0, seed=7)
    np.testing.assert_array_equal(eeg, 0.0)
    np.testing.assert_array_equal(blink_a, blink_b)
    assert np.max(np.abs(blink_a)) > 80.0
    assert np.max(np.abs(motion)) > 40.0
    assert not np.array_equal(blink_a, motion)
