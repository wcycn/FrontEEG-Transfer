#!/usr/bin/env python3
"""Run all participant-held-out ds007169 folds across local GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--gpus", nargs="+", default=("0", "1"))
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def run_fold(
    subject: str,
    gpu: str,
    args: argparse.Namespace,
    feature_root: Path,
    output_root: Path,
) -> str:
    output = output_root / "subjects" / f"{subject}.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_ds007169_subject.py")),
        "--feature-root",
        str(feature_root),
        "--target-subject",
        subject,
        "--output",
        str(output),
        "--device",
        f"cuda:{gpu}",
        "--budgets",
        *[str(value) for value in args.budgets],
        "--repeats",
        str(args.repeats),
        "--source-epochs",
        str(args.source_epochs),
        "--source-patience",
        str(args.source_patience),
        "--adapt-epochs",
        str(args.adapt_epochs),
        "--scratch-epochs",
        str(args.scratch_epochs),
        "--seed",
        str(args.seed),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subjects = sorted(
        path.name.removesuffix("_fp1_fp2_psd.npz")
        for path in feature_root.glob("sub-*_fp1_fp2_psd.npz")
    )
    if len(subjects) != 18:
        raise ValueError(f"expected 18 subjects, found {len(subjects)}")
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(
                run_fold,
                subject,
                args.gpus[index % len(args.gpus)],
                args,
                feature_root,
                output_root,
            ): subject
            for index, subject in enumerate(subjects)
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append({"subject": subject, "error": repr(error)})
                print(f"failed target={subject}: {error!r}", flush=True)
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
