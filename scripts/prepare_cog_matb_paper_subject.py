#!/usr/bin/env python3
"""Prepare one COG-BCI MATB subject with the paper's 61-channel PSD layout.

The paper used EEGLAB/Picard/ICLabel before feature extraction. This local
reproduction implements the deterministic signal/feature steps, but does not
claim to reproduce automatic ICA component rejection.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from prepare_cog_matb_psd_subject import extract_pair, matb_label, subject_session
from scipy.signal import welch

from fronteeg_transfer.cog_bci import canonical_channel_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paper_channel_indices(channel_names: list[str]) -> list[int]:
    """Keep the paper's common 61 scalp channels after TP10 re-reference."""
    excluded = {"cz", "tp10"}
    selected = [
        index
        for index, name in enumerate(channel_names)
        if not canonical_channel_name(name).startswith("ecg")
        and canonical_channel_name(name) not in excluded
    ]
    if len(selected) != 61:
        raise ValueError(f"expected 61 common EEG channels, found {len(selected)}")
    return selected


def extract_features(set_path: Path) -> tuple[np.ndarray, list[str]]:
    import mne

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    if "TP10" not in raw.ch_names:
        raise ValueError(f"TP10 reference channel missing in {set_path}")
    raw.set_eeg_reference(ref_channels=["TP10"], projection=False, verbose="ERROR")
    selection = paper_channel_indices(raw.ch_names)
    channel_names = [raw.ch_names[index] for index in selection]
    raw.pick(selection)
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    raw.resample(250.0, verbose="ERROR")
    data = raw.get_data(units="uV")

    window_samples = 1_000
    step_samples = 500
    starts = range(0, data.shape[1] - window_samples + 1, step_samples)
    epochs = np.stack([data[:, start : start + window_samples] for start in starts])
    frequencies, power = welch(
        epochs,
        fs=250.0,
        window="hamming",
        nperseg=250,
        noverlap=125,
        nfft=256,
        axis=-1,
    )
    spectrum_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
    features = []
    for lower, upper in bands:
        mask = (frequencies >= lower) & (frequencies < upper)
        features.append(spectrum_db[..., mask].mean(axis=-1))
    return np.stack(features, axis=-1).astype(np.float32), channel_names


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
        with tempfile.TemporaryDirectory(prefix="cog-matb-paper-") as temporary:
            temporary_path = Path(temporary)
            for (subject, session), session_members in sorted(grouped.items()):
                destination = output_root / f"{subject}_{session}_matb_psd.npz"
                if destination.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite {destination}")
                feature_parts, label_parts = [], []
                channel_names: list[str] | None = None
                for member in sorted(session_members, key=matb_label):
                    values, current_names = extract_features(
                        extract_pair(archive, member, temporary_path)
                    )
                    if channel_names is None:
                        channel_names = current_names
                    elif channel_names != current_names:
                        raise ValueError(f"channel order mismatch in {subject}/{session}")
                    feature_parts.append(values)
                    label_parts.append(np.full(len(values), matb_label(member), dtype=np.int64))
                if len(feature_parts) != 3 or channel_names is None:
                    raise ValueError(f"expected three MATB conditions for {subject}/{session}")
                features = np.concatenate(feature_parts)
                labels = np.concatenate(label_parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    features=features,
                    label=labels,
                    channel_names=np.asarray(channel_names),
                    subject=np.asarray(subject),
                    session=np.asarray(session.removeprefix("ses-")),
                    task=np.asarray("matb"),
                    channel_set=np.asarray("paper_61ch"),
                    sampling_rate=np.asarray(250.0, dtype=np.float32),
                    epoch_seconds=np.asarray(4.0, dtype=np.float32),
                    overlap=np.asarray(0.5, dtype=np.float32),
                    bands=np.asarray(((4, 8), (8, 13), (13, 20), (20, 30))),
                    preprocessing=np.asarray(
                        "TP10-reference, remove ECG/Cz/TP10, 1-40Hz, no ICA, "
                        "Welch Hamming 1s 50%, mean spectral dB"
                    ),
                )
                reports.append(
                    {
                        "path": str(destination),
                        "subject": subject,
                        "session": session.removeprefix("ses-"),
                        "shape": list(features.shape),
                        "per_class": [int((labels == label).sum()) for label in range(3)],
                        "finite": bool(np.isfinite(features).all()),
                        "range_db": [float(features.min()), float(features.max())],
                    }
                )
    print(json.dumps({"archive": str(archive_path), "sessions": reports}, indent=2))


if __name__ == "__main__":
    main()
