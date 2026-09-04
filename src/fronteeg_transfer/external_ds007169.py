"""Utilities for the participant-disjoint OpenNeuro ds007169 experiment."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.signal import welch

BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
FILE_PATTERN = re.compile(r"(?P<subject>sub-\d+)_nback-(?P<level>[1-4])_(?:train|val|test)\.npz")


def discover_recordings(root: Path) -> dict[str, dict[int, Path]]:
    """Return the four n-back recordings for every available participant."""
    recordings: dict[str, dict[int, Path]] = {}
    for path in sorted(root.glob("sub-*_nback-*.npz")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        subject = match.group("subject")
        level = int(match.group("level"))
        if level in recordings.setdefault(subject, {}):
            raise ValueError(f"duplicate {subject} n-back level {level}")
        recordings[subject][level] = path
    incomplete = {
        subject: sorted(levels) for subject, levels in recordings.items() if len(levels) != 4
    }
    if incomplete:
        raise ValueError(f"participants without four n-back levels: {incomplete}")
    return recordings


def extract_psd_features(eeg: np.ndarray, sampling_rate: float = 200.0) -> np.ndarray:
    """Extract the same four log-PSD bands used by the MATB feature pipeline."""
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 3 or eeg.shape[1] != 2:
        raise ValueError(f"expected [windows, 2, samples], got {eeg.shape}")
    segment = int(round(sampling_rate))
    frequencies, power = welch(
        eeg,
        fs=sampling_rate,
        window="hamming",
        nperseg=segment,
        noverlap=segment // 2,
        nfft=256,
        axis=-1,
    )
    spectrum_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    features = []
    for lower, upper in BANDS:
        mask = (frequencies >= lower) & (frequencies < upper)
        if not np.any(mask):
            raise ValueError(f"frequency grid contains no bins for {lower}-{upper} Hz")
        features.append(spectrum_db[..., mask].mean(axis=-1))
    result = np.stack(features, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite PSD feature")
    return result


def temporal_partition(
    labels: np.ndarray,
    within_class_index: np.ndarray,
    *,
    maximum_budget: int,
) -> dict[str, np.ndarray]:
    """Create non-overlapping calibration candidates and a later test segment.

    Windows have a 4 s duration and 2 s stride. Candidate indices 0, 2, ... are
    non-overlapping. Test windows begin after ``2 * maximum_budget`` so no raw
    samples are shared with the final calibration candidate.
    """
    labels = np.asarray(labels)
    within_class_index = np.asarray(within_class_index)
    calibration, test = [], []
    boundary = 2 * maximum_budget
    for label in sorted(np.unique(labels).tolist()):
        label_indices = np.flatnonzero(labels == label)
        ordered = label_indices[np.argsort(within_class_index[label_indices])]
        candidates = ordered[:boundary:2]
        later = ordered[boundary:]
        if len(candidates) < maximum_budget or len(later) == 0:
            raise ValueError(
                f"class {label} cannot support {maximum_budget}-shot plus a later test set"
            )
        calibration.extend(candidates.tolist())
        test.extend(later.tolist())
    return {
        "calibration_candidates": np.asarray(sorted(calibration), dtype=np.int64),
        "test": np.asarray(sorted(test), dtype=np.int64),
    }


def source_temporal_partition(
    labels: np.ndarray,
    within_class_index: np.ndarray,
    *,
    validation_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Split each source recording in time with a one-stride leakage guard."""
    labels = np.asarray(labels)
    within_class_index = np.asarray(within_class_index)
    train, validation = [], []
    for label in sorted(np.unique(labels).tolist()):
        label_indices = np.flatnonzero(labels == label)
        ordered = label_indices[np.argsort(within_class_index[label_indices])]
        split = int(np.floor(len(ordered) * (1.0 - validation_fraction)))
        if split < 3 or len(ordered) - split < 2:
            raise ValueError(f"class {label} is too short for temporal source split")
        train.extend(ordered[: split - 1].tolist())
        validation.extend(ordered[split:].tolist())
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "validation": np.asarray(sorted(validation), dtype=np.int64),
    }


def minmax_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(features).min(axis=0)
    span = np.asarray(features).max(axis=0) - minimum
    if np.any(span <= 0):
        raise ValueError("constant channel-band feature in normalization reference")
    return minimum, span


def apply_minmax(features: np.ndarray, minimum: np.ndarray, span: np.ndarray) -> np.ndarray:
    return ((np.asarray(features) - minimum) / span).astype(np.float32)
