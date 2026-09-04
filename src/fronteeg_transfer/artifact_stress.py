"""Deterministic synthetic artifact perturbations for offline EEG stress tests."""

from __future__ import annotations

import numpy as np


def add_blink_artifacts(
    eeg: np.ndarray,
    *,
    sampling_rate: float,
    amplitude_uv: float,
    seed: int,
) -> np.ndarray:
    """Add one frontal Gaussian blink pulse to each window."""
    values = np.asarray(eeg, dtype=np.float32).copy()
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"expected [windows, 2, samples], got {values.shape}")
    rng = np.random.default_rng(seed)
    samples = values.shape[-1]
    time = np.arange(samples, dtype=np.float32) / sampling_rate
    centers = rng.uniform(0.7, samples / sampling_rate - 0.7, size=len(values))
    widths = rng.uniform(0.07, 0.13, size=len(values))
    polarities = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
    pulse = np.exp(-0.5 * ((time[None, :] - centers[:, None]) / widths[:, None]) ** 2)
    pulse *= (amplitude_uv * polarities)[:, None]
    values[:, 0] += pulse.astype(np.float32)
    values[:, 1] += (0.85 * pulse).astype(np.float32)
    return values


def add_motion_artifacts(
    eeg: np.ndarray,
    *,
    sampling_rate: float,
    amplitude_uv: float,
    seed: int,
) -> np.ndarray:
    """Add channel-asymmetric electrode steps with exponential recovery."""
    values = np.asarray(eeg, dtype=np.float32).copy()
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"expected [windows, 2, samples], got {values.shape}")
    rng = np.random.default_rng(seed)
    samples = values.shape[-1]
    onset = rng.integers(
        int(round(0.7 * sampling_rate)),
        int(round((samples / sampling_rate - 0.7) * sampling_rate)),
        size=len(values),
    )
    decay = rng.uniform(0.35, 0.8, size=len(values))
    channel = rng.integers(0, 2, size=len(values))
    polarity = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
    for index in range(len(values)):
        offset = np.arange(samples - onset[index], dtype=np.float32) / sampling_rate
        transient = amplitude_uv * polarity[index] * np.exp(-offset / decay[index])
        values[index, channel[index], onset[index] :] += transient.astype(np.float32)
        values[index, 1 - channel[index], onset[index] :] += (0.2 * transient).astype(np.float32)
    return values
