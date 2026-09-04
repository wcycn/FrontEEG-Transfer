#!/usr/bin/env python3
"""Run all strict-LOSO EEGNet targets over a configurable set of local GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_target(args: argparse.Namespace, target: str, gpu: int) -> str:
    output_root = Path(args.output_directory).expanduser().resolve()
    result = output_root / "subjects" / f"{target}.json"
    checkpoint = output_root / "checkpoints" / f"{target}.pt"
    if result.exists() and not args.overwrite:
        return f"skip target={target} existing={result}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_matb_eegnet_subject.py")),
        "--raw-root",
        str(Path(args.raw_root).expanduser().resolve()),
        "--target-subject",
        target,
        "--output",
        str(result),
        "--checkpoint",
        str(checkpoint),
        "--device",
        f"cuda:{gpu}",
        "--source-epochs",
        str(args.source_epochs),
        "--source-patience",
        str(args.source_patience),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--reuse-checkpoint",
    ]
    log = output_root / "logs" / f"{target}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return f"complete target={target} gpu={gpu} result={result}"


def main() -> None:
    args = parse_args()
    root = Path(args.raw_root).expanduser().resolve()
    suffix = "_ses-S1_matb_raw.npz"
    subjects = sorted(path.name.removesuffix(suffix) for path in root.glob(f"*{suffix}"))
    if len(subjects) != 26:
        raise ValueError(f"expected 26 subjects, found {len(subjects)}")
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        pending = {
            executor.submit(run_target, args, subject, args.gpus[index % len(args.gpus)]): subject
            for index, subject in enumerate(subjects)
        }
        for future in as_completed(pending):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
