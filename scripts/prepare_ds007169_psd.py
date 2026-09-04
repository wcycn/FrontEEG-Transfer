#!/usr/bin/env python3
"""Convert preprocessed ds007169 Fp1/Fp2 windows to four-band PSD features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fronteeg_transfer.external_ds007169 import (
    BANDS,
    discover_recordings,
    extract_psd_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_directory).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    recordings = discover_recordings(input_root)
    reports = []
    for subject, levels in sorted(recordings.items()):
        output = output_root / f"{subject}_fp1_fp2_psd.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output}")
        feature_parts, label_parts, index_parts, start_parts = [], [], [], []
        for level, path in sorted(levels.items()):
            with np.load(path, allow_pickle=False) as loaded:
                eeg = loaded["eeg"]
                labels = loaded["label"].astype(np.int64)
                starts = loaded["window_start_sec"].astype(np.float64)
            expected_label = level - 1
            if not np.all(labels == expected_label):
                raise ValueError(f"label mismatch in {path}")
            feature_parts.append(extract_psd_features(eeg))
            label_parts.append(labels)
            index_parts.append(np.arange(len(labels), dtype=np.int64))
            start_parts.append(starts)
        features = np.concatenate(feature_parts)
        labels = np.concatenate(label_parts)
        within_class_index = np.concatenate(index_parts)
        starts = np.concatenate(start_parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            features=features,
            label=labels,
            within_class_index=within_class_index,
            window_start_sec=starts,
            channel_names=np.asarray(["Fp1", "Fp2"]),
            bands=np.asarray(BANDS, dtype=np.float32),
            sampling_rate=np.asarray(200.0, dtype=np.float32),
            epoch_seconds=np.asarray(4.0, dtype=np.float32),
            stride_seconds=np.asarray(2.0, dtype=np.float32),
            subject=np.asarray(subject),
        )
        reports.append(
            {
                "subject": subject,
                "output": str(output),
                "shape": list(features.shape),
                "per_class": [int(np.sum(labels == label)) for label in range(4)],
            }
        )
    print(json.dumps({"subjects": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
