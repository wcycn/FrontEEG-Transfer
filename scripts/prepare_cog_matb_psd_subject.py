#!/usr/bin/env python3
"""Prepare one COG-BCI MATB subject as paper-matched PSD node features."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch

from fronteeg_transfer.cog_bci import cog_channel_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--channel-set", choices=("all_eeg", "fp1_fp2"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subject_session(member: str) -> tuple[str, str]:
    match = re.search(r"(?:^|/)(sub-[^/]+)/(ses-[^/]+)/", member, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"expected sub-*/ses-* path, got {member}")
    return match.group(1), match.group(2)


def matb_label(member: str) -> int:
    stem = Path(member).stem.casefold()
    if stem == "matbeasy":
        return 0
    if stem == "matbmed":
        return 1
    if stem == "matbdiff":
        return 2
    raise ValueError(f"unknown MATB condition in {member}")


def extract_pair(archive: zipfile.ZipFile, set_name: str, destination: Path) -> Path:
    fdt_name = str(Path(set_name).with_suffix(".fdt"))
    if fdt_name not in set(archive.namelist()):
        raise FileNotFoundError(f"missing {fdt_name}")
    archive.extract(set_name, destination)
    archive.extract(fdt_name, destination)
    return destination / set_name


def extract_features(set_path: Path, channel_set: str) -> tuple[np.ndarray, list[str]]:
    import mne

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    if "TP10" not in raw.ch_names:
        raise ValueError(f"TP10 reference channel missing in {set_path}")
    raw.set_eeg_reference(ref_channels=["TP10"], projection=False, verbose="ERROR")
    selection = cog_channel_indices(raw.ch_names, channel_set)
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
        nperseg=250,
        noverlap=125,
        axis=-1,
    )
    bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
    features = []
    for lower, upper in bands:
        mask = (frequencies >= lower) & (frequencies < upper)
        features.append(power[..., mask].mean(axis=-1))
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
        with tempfile.TemporaryDirectory(prefix="cog-matb-psd-") as temporary:
            temporary_path = Path(temporary)
            for (subject, session), session_members in sorted(grouped.items()):
                destination = (
                    output_root / args.channel_set / f"{subject}_{session}_matb_psd.npz"
                )
                if destination.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite {destination}")
                features, labels = [], []
                channel_names: list[str] | None = None
                for member in sorted(session_members, key=matb_label):
                    values, current_names = extract_features(
                        extract_pair(archive, member, temporary_path),
                        args.channel_set,
                    )
                    if channel_names is None:
                        channel_names = current_names
                    elif channel_names != current_names:
                        raise ValueError(f"channel order mismatch in {subject}/{session}")
                    label = matb_label(member)
                    features.append(values)
                    labels.append(np.full(len(values), label, dtype=np.int64))
                if len(features) != 3 or channel_names is None:
                    raise ValueError(f"expected three MATB conditions for {subject}/{session}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    features=np.concatenate(features),
                    label=np.concatenate(labels),
                    channel_names=np.asarray(channel_names),
                    subject=np.asarray(subject),
                    session=np.asarray(session.removeprefix("ses-")),
                    task=np.asarray("matb"),
                    channel_set=np.asarray(args.channel_set),
                    sampling_rate=np.asarray(250.0, dtype=np.float32),
                    epoch_seconds=np.asarray(4.0, dtype=np.float32),
                    overlap=np.asarray(0.5, dtype=np.float32),
                    bands=np.asarray(((4, 8), (8, 13), (13, 20), (20, 30))),
                    preprocessing=np.asarray("TP10-reference, 1-40Hz, no ICA"),
                )
                reports.append(
                    {
                        "path": str(destination),
                        "subject": subject,
                        "session": session.removeprefix("ses-"),
                        "shape": list(np.concatenate(features).shape),
                        "per_class": [len(values) for values in features],
                    }
                )
    print(json.dumps({"archive": str(archive_path), "sessions": reports}, indent=2))


if __name__ == "__main__":
    main()
