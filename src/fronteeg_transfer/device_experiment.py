"""Protocol and data utilities for the real-device workload experiment."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("Easy", "Medium", "Difficult")
EVENT_FIELDS = (
    "lsl_timestamp",
    "phase",
    "block_id",
    "condition",
    "event_type",
    "trial_index",
    "stimulus",
    "is_target",
    "response",
    "correct",
    "reaction_time_seconds",
    "detail",
)


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    phase: str
    condition: str
    duration_seconds: float
    seed: int


@dataclass(frozen=True)
class ProtocolConfig:
    practice_block_duration_seconds: float = 30.0
    calibration_repeats_per_condition: int = 1
    test_repeats_per_condition: int = 1
    calibration_block_duration_seconds: float = 60.0
    test_block_duration_seconds: float = 300.0
    break_seconds: float = 30.0
    countdown_seconds: int = 3
    seed: int = 20260902

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ProtocolConfig:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown protocol fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.practice_block_duration_seconds < 4.0:
            raise ValueError("practice_block_duration_seconds must be at least 4 seconds")
        if self.calibration_repeats_per_condition <= 0:
            raise ValueError("calibration_repeats_per_condition must be positive")
        if self.test_repeats_per_condition <= 0:
            raise ValueError("test_repeats_per_condition must be positive")
        if self.calibration_block_duration_seconds < 4.0:
            raise ValueError("calibration_block_duration_seconds must be at least 4 seconds")
        if self.test_block_duration_seconds < 4.0:
            raise ValueError("test_block_duration_seconds must be at least 4 seconds")
        if self.break_seconds < 0 or self.countdown_seconds < 0:
            raise ValueError("break and countdown durations cannot be negative")


def validate_participant_id(value: str) -> str:
    """Return a filesystem-safe pseudonym without collecting identifying information."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", value):
        raise ValueError("participant ID must be 1-32 ASCII letters, digits, '_' or '-'")
    return value


def balanced_condition_order(repeats: int, seed: int) -> list[str]:
    """Randomize balanced mini-rounds while preventing identical adjacent conditions."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    names = list(CONDITIONS)
    order: list[str] = []
    for _ in range(repeats):
        candidates = names.copy()
        rng.shuffle(candidates)
        if order and candidates[0] == order[-1]:
            swap = next(index for index, name in enumerate(candidates) if name != order[-1])
            candidates[0], candidates[swap] = candidates[swap], candidates[0]
        order.extend(candidates)
    return order


def build_schedule(config: ProtocolConfig) -> list[BlockSpec]:
    schedule = [
        BlockSpec(
            block_id=f"practice-{index:02d}",
            phase="practice",
            condition=condition,
            duration_seconds=config.practice_block_duration_seconds,
            seed=(config.seed + 50) * 100 + index,
        )
        for index, condition in enumerate(CONDITIONS, start=1)
    ]
    calibration_seed = config.seed + 101
    test_seed = config.seed + 202
    calibration_order = balanced_condition_order(
        config.calibration_repeats_per_condition, calibration_seed
    )
    test_order = balanced_condition_order(config.test_repeats_per_condition, test_seed)
    if len(calibration_order) == len(test_order) == len(CONDITIONS):
        offset = 1 + random.Random(test_seed).randrange(len(CONDITIONS) - 1)
        test_order = calibration_order[offset:] + calibration_order[:offset]
    phases = (
        (
            "calibration",
            calibration_order,
            config.calibration_block_duration_seconds,
            calibration_seed,
        ),
        ("test", test_order, config.test_block_duration_seconds, test_seed),
    )
    for phase, order, duration, seed in phases:
        for index, condition in enumerate(order, start=1):
            schedule.append(
                BlockSpec(
                    block_id=f"{phase}-{index:02d}",
                    phase=phase,
                    condition=condition,
                    duration_seconds=duration,
                    seed=seed * 100 + index,
                )
            )
    return schedule


class EventLog:
    """Thread-safe event table with optional crash-resilient incremental storage."""

    def __init__(self, live_path: str | Path | None = None) -> None:
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._live_path = Path(live_path) if live_path is not None else None
        if self._live_path is not None:
            self._live_path.parent.mkdir(parents=True, exist_ok=True)
            with self._live_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=EVENT_FIELDS, delimiter="\t").writeheader()

    def append(self, **values: Any) -> None:
        unknown = set(values) - set(EVENT_FIELDS)
        if unknown:
            raise ValueError(f"unknown event fields: {sorted(unknown)}")
        row = {field: values.get(field, "") for field in EVENT_FIELDS}
        with self._lock:
            self._rows.append(row)
            if self._live_path is not None:
                with self._live_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, delimiter="\t")
                    writer.writerow(row)
                    handle.flush()

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def write_tsv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows())
        temporary.replace(destination)


def write_json(path: str | Path, values: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def schedule_as_dicts(schedule: list[BlockSpec]) -> list[dict[str, Any]]:
    return [asdict(block) for block in schedule]


def signal_quality_summary(
    samples_microvolts: np.ndarray, timestamps: np.ndarray, sampling_rate: float
) -> dict[str, Any]:
    values = np.asarray(samples_microvolts, dtype=np.float64)
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
        "channel_mean_microvolts": values.mean(axis=0).tolist(),
        "channel_std_microvolts": values.std(axis=0).tolist(),
        "channel_peak_to_peak_microvolts": np.ptp(values, axis=0).tolist(),
        "non_finite_values": int(np.size(values) - np.isfinite(values).sum()),
    }
