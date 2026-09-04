#!/usr/bin/env python3
"""List local LSL streams and verify that an EEG stream is producing samples."""

from __future__ import annotations

import argparse
import time

import numpy as np

from fronteeg_transfer.lsl_device import read_stream_channel_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--stream-name", default=None)
    parser.add_argument(
        "--channel-indices", nargs=2, type=int, default=(0, 1), help="Fp1 Fp2 indices"
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=3.0,
        help="Continue receiving the selected EEG stream and report its observed rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.watch_seconds < 0:
        raise ValueError("--watch-seconds must be non-negative")
    try:
        from pylsl import StreamInlet, resolve_streams
    except ImportError as error:
        raise RuntimeError("install pylsl before inspecting LSL streams") from error
    streams = resolve_streams(wait_time=args.timeout)
    if not streams:
        raise RuntimeError("no LSL streams found")
    print("name\ttype\tchannels\tnominal_rate\tsource_id")
    for stream in streams:
        print(
            f"{stream.name()}\t{stream.type()}\t{stream.channel_count()}\t"
            f"{stream.nominal_srate()}\t{stream.source_id()}"
        )
    eeg = [stream for stream in streams if stream.type().strip().upper() == "EEG"]
    if args.stream_name is not None:
        eeg = [stream for stream in eeg if stream.name() == args.stream_name]
    if len(eeg) != 1:
        raise RuntimeError(
            f"expected exactly one selected EEG stream, found {[stream.name() for stream in eeg]}"
        )
    inlet = StreamInlet(eeg[0], max_buflen=30)
    info = inlet.info(timeout=args.timeout)
    labels, units = read_stream_channel_metadata(info)
    print(f"channel_labels={list(labels)}")
    print(f"channel_units={list(units)}")
    indices = tuple(args.channel_indices)
    if len(set(indices)) != 2 or min(indices) < 0 or max(indices) >= info.channel_count():
        raise ValueError(
            "--channel-indices must select two distinct channels from "
            f"0..{info.channel_count() - 1}"
        )
    sample, timestamp = inlet.pull_sample(timeout=args.timeout)
    if sample is None:
        raise RuntimeError("EEG stream exists but no sample arrived")
    print(f"first_sample_timestamp={timestamp:.9f} values={sample}")
    started = time.monotonic()
    received_samples = [sample]
    received_timestamps = [float(timestamp)]
    while time.monotonic() - started < args.watch_seconds:
        remaining = args.watch_seconds - (time.monotonic() - started)
        samples, timestamps = inlet.pull_chunk(
            timeout=min(0.5, max(remaining, 0.0)), max_samples=1024
        )
        received_samples.extend(samples)
        received_timestamps.extend(float(value) for value in timestamps)
    elapsed = time.monotonic() - started
    values = np.asarray(received_samples, dtype=np.float64)
    times = np.asarray(received_timestamps, dtype=np.float64)
    observed_rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    selected = values[:, indices]
    print(
        f"monitor_seconds={elapsed:.3f} received_samples={len(values)} "
        f"observed_rate_hz={observed_rate:.3f}"
    )
    print(
        f"selected_indices={list(indices)} "
        f"selected_labels={[labels[index] for index in indices]} "
        f"finite={bool(np.isfinite(selected).all())} "
        f"mean={selected.mean(axis=0).tolist()} "
        f"std={selected.std(axis=0).tolist()} "
        f"peak_to_peak={np.ptp(selected, axis=0).tolist()}"
    )


if __name__ == "__main__":
    main()
