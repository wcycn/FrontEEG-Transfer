#!/usr/bin/env python3
"""Prepare one COG-BCI subject as paper-aligned Fp1/Fp2 raw MATB windows."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from prepare_cog_matb_psd_subject import extract_pair, matb_label, subject_session

from fronteeg_transfer.cog_bci import canonical_channel_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fp1_fp2_indices(channel_names: list[str]) -> list[int]:
    canonical = [canonical_channel_name(name) for name in channel_names]
    return [canonical.index("fp1"), canonical.index("fp2")]


def sliding_windows(data: np.ndarray, window: int = 1_000, step: int = 500) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must have shape [channels, samples]")
    if window <= 0 or step <= 0 or values.shape[1] < window:
        raise ValueError("invalid sliding-window dimensions")
    starts = range(0, values.shape[1] - window + 1, step)
    return np.stack([values[:, start : start + window] for start in starts])


def extract_epochs(set_path: Path) -> tuple[np.ndarray, list[str]]:
    import mne

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    canonical = [canonical_channel_name(name) for name in raw.ch_names]
    if "tp10" not in canonical:
        raise ValueError(f"TP10 reference channel missing in {set_path}")
    tp10 = raw.ch_names[canonical.index("tp10")]
    raw.set_eeg_reference(ref_channels=[tp10], projection=False, verbose="ERROR")
    selection = fp1_fp2_indices(raw.ch_names)
    names = [raw.ch_names[index] for index in selection]
    raw.pick(selection)
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    raw.resample(250.0, verbose="ERROR")
    epochs = sliding_windows(raw.get_data(units="uV"))
    return epochs.astype(np.float32), names


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"not a complete ZIP archive: {archive_path}")

    reports = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name.casefold() in {"matbeasy.set", "matbmed.set", "matbdiff.set"}
        ]
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for member in members:
            grouped[subject_session(member)].append(member)
        with tempfile.TemporaryDirectory(prefix="cog-matb-raw-") as temporary:
            temporary_path = Path(temporary)
            for (subject, session), session_members in sorted(grouped.items()):
                destination = output_root / f"{subject}_{session}_matb_raw.npz"
                if destination.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite {destination}")
                epoch_parts: list[np.ndarray] = []
                label_parts: list[np.ndarray] = []
                channel_names: list[str] | None = None
                for member in sorted(session_members, key=matb_label):
                    epochs, current_names = extract_epochs(
                        extract_pair(archive, member, temporary_path)
                    )
                    if channel_names is None:
                        channel_names = current_names
                    elif channel_names != current_names:
                        raise ValueError(f"channel order mismatch in {subject}/{session}")
                    epoch_parts.append(epochs)
                    label_parts.append(
                        np.full(len(epochs), matb_label(member), dtype=np.int64)
                    )
                if len(epoch_parts) != 3 or channel_names is None:
                    raise ValueError(f"expected three MATB conditions for {subject}/{session}")
                eeg = np.concatenate(epoch_parts)
                labels = np.concatenate(label_parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    eeg=eeg,
                    label=labels,
                    channel_names=np.asarray(channel_names),
                    subject=np.asarray(subject),
                    session=np.asarray(session.removeprefix("ses-")),
                    task=np.asarray("matb"),
                    sampling_rate=np.asarray(250.0, dtype=np.float32),
                    epoch_seconds=np.asarray(4.0, dtype=np.float32),
                    overlap=np.asarray(0.5, dtype=np.float32),
                    input_unit=np.asarray("microvolts"),
                    preprocessing=np.asarray(
                        "TP10-reference, Fp1/Fp2, 1-40Hz, 250Hz, no ICA"
                    ),
                )
                reports.append(
                    {
                        "path": str(destination),
                        "subject": subject,
                        "session": session.removeprefix("ses-"),
                        "shape": list(eeg.shape),
                        "per_class": [int((labels == label).sum()) for label in range(3)],
                        "finite": bool(np.isfinite(eeg).all()),
                    }
                )
    print(json.dumps({"archive": str(archive_path), "sessions": reports}, indent=2))


if __name__ == "__main__":
    main()
