from __future__ import annotations

import numpy as np
from pyriemann.geometry.mean import mean_riemann

from fronteeg_transfer.riemann_transfer import (
    CovarianceDomain,
    align_by_group,
    align_to_reference,
    estimate_covariances,
    fit_source_classifier,
    predict_target,
    recenter_by_group,
    recenter_covariances,
)


def synthetic_eeg(seed: int, scale: float, count: int = 18) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 3
    eeg = rng.normal(size=(count, 2, 128))
    eeg[:, 1] += (0.15 + 0.2 * labels[:, None]) * eeg[:, 0]
    return (eeg * scale).astype(np.float32), labels


def test_recentered_domain_mean_is_identity() -> None:
    eeg, _ = synthetic_eeg(1, 3.0)
    centered = recenter_covariances(estimate_covariances(eeg))
    np.testing.assert_allclose(mean_riemann(centered), np.eye(2), atol=1e-6)


def test_groups_are_recentered_independently() -> None:
    first, _ = synthetic_eeg(2, 1.0)
    second, _ = synthetic_eeg(3, 8.0)
    covariances = estimate_covariances(np.concatenate((first, second)))
    groups = np.array(["a"] * len(first) + ["b"] * len(second))
    centered = recenter_by_group(covariances, groups)
    np.testing.assert_allclose(mean_riemann(centered[groups == "a"]), np.eye(2), atol=1e-6)
    np.testing.assert_allclose(mean_riemann(centered[groups == "b"]), np.eye(2), atol=1e-6)


def test_separate_reference_alignment_does_not_consume_test_distribution() -> None:
    reference_eeg, _ = synthetic_eeg(20, 4.0)
    test_eeg, _ = synthetic_eeg(21, 9.0)
    reference = estimate_covariances(reference_eeg)
    test = estimate_covariances(test_eeg)

    aligned = align_to_reference(test, reference, mean="euclidean")
    assert aligned.shape == test.shape
    assert not np.allclose(aligned, test)


def test_euclidean_group_alignment_centers_arithmetic_means() -> None:
    first, _ = synthetic_eeg(22, 1.0)
    second, _ = synthetic_eeg(23, 7.0)
    covariances = estimate_covariances(np.concatenate((first, second)))
    groups = np.array(["a"] * len(first) + ["b"] * len(second))
    aligned = align_by_group(covariances, groups, mean="euclidean")

    np.testing.assert_allclose(aligned[groups == "a"].mean(axis=0), np.eye(2), atol=1e-6)
    np.testing.assert_allclose(aligned[groups == "b"].mean(axis=0), np.eye(2), atol=1e-6)


def test_zero_shot_prediction_does_not_require_target_labels() -> None:
    source_a, labels_a = synthetic_eeg(4, 1.0)
    source_b, labels_b = synthetic_eeg(5, 4.0)
    source_eeg = np.concatenate((source_a, source_b))
    source_labels = np.concatenate((labels_a, labels_b))
    groups = np.array(["a"] * len(source_a) + ["b"] * len(source_b))
    source = CovarianceDomain(estimate_covariances(source_eeg), source_labels, groups)
    target_eeg, _ = synthetic_eeg(6, 10.0)
    target = CovarianceDomain(estimate_covariances(target_eeg))
    classifier = fit_source_classifier(source, recenter=True)
    probabilities = predict_target(classifier, target, recenter=True)
    assert probabilities.shape == (len(target_eeg), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
