#!/usr/bin/env python3
"""Run controlled FrontEEGNet low-shot adaptation for all MATB targets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_matb_eegnet_subject import discover_subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 4, 8, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--adapt-epochs", type=int, default=200)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_target(args: argparse.Namespace, target: str, gpu: int) -> str:
    output_root = Path(args.output_directory).expanduser().resolve()
    output = output_root / "subjects" / f"{target}.json"
    if output.exists() and not args.overwrite:
        return f"skip target={target}"
    checkpoint = (
        Path(args.source_directory).expanduser().resolve()
        / "checkpoints"
        / f"{target}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_matb_fronteegnet_adaptation_subject.py")),
        "--raw-root",
        str(Path(args.raw_root).expanduser().resolve()),
        "--source-checkpoint",
        str(checkpoint),
        "--target-subject",
        target,
        "--output",
        str(output),
        "--budgets",
        *(str(value) for value in args.budgets),
        "--repeats",
        str(args.repeats),
        "--adapt-epochs",
        str(args.adapt_epochs),
        "--scratch-epochs",
        str(args.scratch_epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        f"cuda:{gpu}",
    ]
    log = output_root / "logs" / f"{target}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return f"complete target={target} gpu={gpu}"


def main() -> None:
    args = parse_args()
    subjects = discover_subjects(Path(args.raw_root).expanduser().resolve())
    if len(subjects) != 26:
        raise ValueError(f"expected 26 valid subjects, found {len(subjects)}")
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        pending = {
            executor.submit(run_target, args, target, args.gpus[index % len(args.gpus)]): target
            for index, target in enumerate(subjects)
        }
        for future in as_completed(pending):
            target = pending[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(f"failed target={target}: {error!r}") from error


if __name__ == "__main__":
    main()
