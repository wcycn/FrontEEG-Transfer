"""Minimal LSL acquisition layer for the confidential two-channel device."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StreamMetadata:
    name: str
    stream_type: str
    channel_count: int
    nominal_sampling_rate: float
    source_id: str
    time_correction_seconds: float
    channel_labels: tuple[str, ...]
    channel_units: tuple[str, ...]
    stream_xml: str


def read_stream_channel_metadata(stream: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read optional LSL channel labels/units without trusting vendor metadata."""
    labels: list[str] = []
    units: list[str] = []
    try:
        channel = stream.desc().child("channels").child("channel")
        for index in range(int(stream.channel_count())):
            if channel.empty():
                labels.extend(f"channel_{item}" for item in range(index, stream.channel_count()))
                units.extend("" for _ in range(index, stream.channel_count()))
                break
            labels.append(channel.child_value("label") or f"channel_{index}")
            units.append(channel.child_value("unit") or "")
            channel = channel.next_sibling()
    except Exception:
        labels = [f"channel_{index}" for index in range(int(stream.channel_count()))]
        units = ["" for _ in labels]
    return tuple(labels), tuple(units)


def normalize_lsl_chunk_timestamps(
    source_timestamps: list[float] | np.ndarray,
    *,
    time_correction_seconds: float,
    sampling_rate: float,
    previous_local_timestamp: float | None,
) -> tuple[np.ndarray, int]:
    """Map raw LSL timestamps to a monotonic local timeline without altering the raw copy."""
    source = np.asarray(source_timestamps, dtype=np.float64)
    if source.ndim != 1 or not len(source):
        raise ValueError("source_timestamps must be a non-empty one-dimensional sequence")
    if not np.isfinite(source).all() or sampling_rate <= 0:
        raise ValueError("LSL timestamps must be finite and sampling_rate must be positive")
    local = source + float(time_correction_seconds)
    internal_ok = len(local) == 1 or bool(np.all(np.diff(local) > 0))
    boundary_ok = previous_local_timestamp is None or local[0] > previous_local_timestamp
    if internal_ok and boundary_ok:
        return local, 0

    step = 1.0 / float(sampling_rate)
    anchor = float(local[-1])
    repaired = anchor - step * np.arange(len(local) - 1, -1, -1, dtype=np.float64)
    if previous_local_timestamp is not None and repaired[0] <= previous_local_timestamp:
        repaired += previous_local_timestamp + step - repaired[0]
    return repaired, len(repaired)


class LslEegRecorder:
    """Receive selected EEG channels in a background thread, preserving LSL timestamps."""

    def __init__(
        self,
        *,
        stream_name: str | None,
        channel_indices: tuple[int, int],
        resolve_timeout_seconds: float = 8.0,
        fallback_sampling_rate: float | None = None,
        recovery_path: str | Path | None = None,
    ) -> None:
        try:
            from pylsl import StreamInlet, local_clock, resolve_streams
        except ImportError as error:
            raise RuntimeError("pylsl is required for device acquisition") from error
        streams = resolve_streams(wait_time=resolve_timeout_seconds)
        eeg_streams = [stream for stream in streams if stream.type().strip().upper() == "EEG"]
        if stream_name is not None:
            eeg_streams = [stream for stream in eeg_streams if stream.name() == stream_name]
        if not eeg_streams:
            available = [(stream.name(), stream.type()) for stream in streams]
            raise RuntimeError(f"no matching EEG LSL stream; available={available}")
        if len(eeg_streams) > 1:
            names = [stream.name() for stream in eeg_streams]
            raise RuntimeError(f"multiple EEG streams found; choose --stream-name from {names}")
        stream = eeg_streams[0]
        if min(channel_indices) < 0 or max(channel_indices) >= stream.channel_count():
            raise ValueError(
                f"channel indices {channel_indices} exceed stream count {stream.channel_count()}"
            )
        nominal_rate = float(stream.nominal_srate())
        if nominal_rate <= 0:
            if fallback_sampling_rate is None or fallback_sampling_rate <= 0:
                raise ValueError("irregular stream requires a positive fallback sampling rate")
            nominal_rate = float(fallback_sampling_rate)
        self._inlet = StreamInlet(stream, max_buflen=3600)
        self._time_correction = float(
            self._inlet.time_correction(timeout=resolve_timeout_seconds)
        )
        full_stream = self._inlet.info(timeout=resolve_timeout_seconds)
        channel_labels, channel_units = read_stream_channel_metadata(full_stream)
        try:
            stream_xml = full_stream.as_xml()
        except Exception:
            stream_xml = ""
        self.metadata = StreamMetadata(
            name=full_stream.name(),
            stream_type=full_stream.type(),
            channel_count=int(full_stream.channel_count()),
            nominal_sampling_rate=nominal_rate,
            source_id=full_stream.source_id(),
            time_correction_seconds=self._time_correction,
            channel_labels=channel_labels,
            channel_units=channel_units,
            stream_xml=stream_xml,
        )
        self._local_clock = local_clock
        self._indices = channel_indices
        self._recovery_path = Path(recovery_path) if recovery_path is not None else None
        self._samples: list[list[float]] = []
        self._all_sample_chunks: list[np.ndarray] = []
        self._source_timestamps: list[float] = []
        self._timestamps: list[float] = []
        self._lock = threading.Lock()
        self._timestamp_repaired_samples = 0
        self._timestamp_repaired_chunks = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def clock(self) -> float:
        return float(self._local_clock())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        self._thread = threading.Thread(target=self._run, name="lsl-eeg-recorder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        recovery_handle = None
        try:
            recovery_writer = None
            if self._recovery_path is not None:
                self._recovery_path.parent.mkdir(parents=True, exist_ok=True)
                recovery_handle = self._recovery_path.open("w", encoding="utf-8", newline="")
                recovery_writer = csv.writer(recovery_handle)
                recovery_writer.writerow(
                    [
                        "local_timestamp",
                        "lsl_timestamp",
                        "timestamp_repaired",
                        *(f"channel_{index}" for index in range(self.metadata.channel_count)),
                    ]
                )
                recovery_handle.flush()
            while not self._stop.is_set():
                samples, timestamps = self._inlet.pull_chunk(timeout=0.2, max_samples=1024)
                if not timestamps:
                    continue
                all_samples = [[float(value) for value in sample] for sample in samples]
                selected = [
                    [sample[self._indices[0]], sample[self._indices[1]]]
                    for sample in all_samples
                ]
                source_times = [float(timestamp) for timestamp in timestamps]
                with self._lock:
                    previous_timestamp = self._timestamps[-1] if self._timestamps else None
                    local_array, repaired_samples = normalize_lsl_chunk_timestamps(
                        source_times,
                        time_correction_seconds=self._time_correction,
                        sampling_rate=self.metadata.nominal_sampling_rate,
                        previous_local_timestamp=previous_timestamp,
                    )
                    local_times = local_array.tolist()
                    self._samples.extend(selected)
                    self._all_sample_chunks.append(np.asarray(all_samples, dtype=np.float32))
                    self._source_timestamps.extend(source_times)
                    self._timestamps.extend(local_times)
                    self._timestamp_repaired_samples += repaired_samples
                    self._timestamp_repaired_chunks += int(repaired_samples > 0)
                if recovery_writer is not None:
                    recovery_writer.writerows(
                        [local, source, int(repaired_samples > 0), *sample]
                        for local, source, sample in zip(
                            local_times, source_times, all_samples, strict=True
                        )
                    )
                    recovery_handle.flush()
        except Exception as error:  # pragma: no cover - hardware failure path
            self.error = error
            self._stop.set()
        finally:
            if recovery_handle is not None:
                recovery_handle.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive() and self.error is None:
                self.error = RuntimeError("EEG recorder thread did not stop cleanly")

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            samples = np.asarray(self._samples, dtype=np.float32)
            source_timestamps = np.asarray(self._source_timestamps, dtype=np.float64)
            timestamps = np.asarray(self._timestamps, dtype=np.float64)
        if samples.size == 0:
            samples = np.empty((0, 2), dtype=np.float32)
        return samples, timestamps, source_timestamps

    def full_snapshot(self) -> np.ndarray:
        with self._lock:
            chunks = [chunk.copy() for chunk in self._all_sample_chunks]
        if not chunks:
            return np.empty((0, self.metadata.channel_count), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    @property
    def timestamp_diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {
                "repaired_samples": self._timestamp_repaired_samples,
                "repaired_chunks": self._timestamp_repaired_chunks,
            }


class SyntheticEegRecorder:
    """LSL-free signal source for validating the full task and file pipeline."""

    def __init__(
        self,
        sampling_rate: float = 250.0,
        seed: int = 20260902,
        recovery_path: str | Path | None = None,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        self.metadata = StreamMetadata(
            name="synthetic",
            stream_type="EEG",
            channel_count=2,
            nominal_sampling_rate=sampling_rate,
            source_id="synthetic",
            time_correction_seconds=0.0,
            channel_labels=("Fp1", "Fp2"),
            channel_units=("microvolts", "microvolts"),
            stream_xml="",
        )
        self._sampling_rate = sampling_rate
        self._rng = np.random.default_rng(seed)
        self._recovery_path = Path(recovery_path) if recovery_path is not None else None
        self._started_at = 0.0
        self._samples: list[list[float]] = []
        self._timestamps: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def clock(self) -> float:
        return time.monotonic()

    def start(self) -> None:
        self._started_at = self.clock()
        self._thread = threading.Thread(target=self._run, name="synthetic-eeg", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        recovery_handle = None
        try:
            recovery_writer = None
            if self._recovery_path is not None:
                self._recovery_path.parent.mkdir(parents=True, exist_ok=True)
                recovery_handle = self._recovery_path.open("w", encoding="utf-8", newline="")
                recovery_writer = csv.writer(recovery_handle)
                recovery_writer.writerow(
                    ["local_timestamp", "lsl_timestamp", "channel_0", "channel_1"]
                )
                recovery_handle.flush()
            index = 0
            while not self._stop.is_set():
                target_index = int((self.clock() - self._started_at) * self._sampling_rate)
                rows = []
                while index <= target_index:
                    timestamp = self._started_at + index / self._sampling_rate
                    theta = np.sin(2 * np.pi * 6.0 * index / self._sampling_rate)
                    alpha = np.sin(2 * np.pi * 10.0 * index / self._sampling_rate)
                    noise = self._rng.normal(0.0, 1.0, size=2)
                    sample = [10.0 * theta + noise[0], 8.0 * alpha + noise[1]]
                    with self._lock:
                        self._samples.append(sample)
                        self._timestamps.append(timestamp)
                    rows.append([timestamp, timestamp, *sample])
                    index += 1
                if recovery_writer is not None and rows:
                    recovery_writer.writerows(rows)
                    recovery_handle.flush()
                time.sleep(0.002)
        except Exception as error:  # pragma: no cover - synthetic I/O failure path
            self.error = error
            self._stop.set()
        finally:
            if recovery_handle is not None:
                recovery_handle.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            samples = np.asarray(self._samples, dtype=np.float32).reshape(-1, 2)
            timestamps = np.asarray(self._timestamps, dtype=np.float64)
        return samples, timestamps, timestamps.copy()

    def full_snapshot(self) -> np.ndarray:
        return self.snapshot()[0]

    @property
    def timestamp_diagnostics(self) -> dict[str, int]:
        return {
            "repaired_samples": 0,
            "repaired_chunks": 0,
        }


class LslMarkerOutlet:
    """Optional marker stream for third-party recording software."""

    def __init__(self, session_id: str) -> None:
        try:
            from pylsl import StreamInfo, StreamOutlet
        except ImportError as error:
            raise RuntimeError("pylsl is required for marker output") from error
        info = StreamInfo(
            name=f"FrontEEGMarkers-{session_id}",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=f"fronteeg-transfer-{session_id}",
        )
        self._outlet: Any = StreamOutlet(info)

    def push(self, value: str, timestamp: float) -> None:
        self._outlet.push_sample([value], timestamp=timestamp)
