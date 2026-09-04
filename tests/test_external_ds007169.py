from __future__ import annotations

import numpy as np

from fronteeg_transfer.external_ds007169 import (
    apply_minmax,
    extract_psd_features,
    minmax_statistics,
    source_temporal_partition,
    temporal_partition,
)


def test_extract_psd_features_identifies_tone_band() -> None:
    sampling_rate = 200.0
    time = np.arange(800) / sampling_rate
    eeg = np.stack(
        [
            np.stack([np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 22 * time)]),
            np.stack([np.sin(2 * np.pi * 6 * time), np.sin(2 * np.pi * 15 * time)]),
        ]
    ).astype(np.float32)
    features = extract_psd_features(eeg, sampling_rate)
    assert features.shape == (2, 2, 4)
    assert int(np.argmax(features[0, 0])) == 1
    assert int(np.argmax(features[0, 1])) == 3
    assert int(np.argmax(features[1, 0])) == 0
    assert int(np.argmax(features[1, 1])) == 2


def test_temporal_partitions_have_no_overlap() -> None:
    labels = np.repeat(np.arange(4), 20)
    within = np.tile(np.arange(20), 4)
    target = temporal_partition(labels, within, maximum_budget=4)
    assert len(target["calibration_candidates"]) == 16
    assert len(target["test"]) == 48
    for label in range(4):
        calibration = target["calibration_candidates"][
            labels[target["calibration_candidates"]] == label
        ]
        test = target["test"][labels[target["test"]] == label]
        assert within[calibration].tolist() == [0, 2, 4, 6]
        assert within[test].min() == 8
    source = source_temporal_partition(labels, within)
    for label in range(4):
        train = source["train"][labels[source["train"]] == label]
        validation = source["validation"][labels[source["validation"]] == label]
        assert within[validation].min() - within[train].max() == 2


def test_minmax_uses_supplied_reference_only() -> None:
    reference = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    minimum, span = minmax_statistics(reference)
    normalized = apply_minmax(reference, minimum, span)
    np.testing.assert_allclose(normalized.min(axis=0), 0.0)
    np.testing.assert_allclose(normalized.max(axis=0), 1.0)
