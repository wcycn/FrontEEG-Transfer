#!/usr/bin/env python3
"""Run flat architecture baselines across all held-out target subjects."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_cross_subject_lowshot_matb import discover_subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--log-directory", required=True)
    parser.add_argument("--gpus", nargs="+", default=("0", "1"))
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=40)
    parser.add_argument("--source-patience", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def complete(path: Path, target: str, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = args.repeats * len(args.budgets) * 5
    return (
        report.get("target_subject") == target
        and report.get("budgets_per_class") == args.budgets
        and report.get("repeats") == args.repeats
        and len(report.get("records", [])) == expected
    )


def worker(
    gpu: str,
    targets: list[str],
    *,
    script: Path,
    feature_root: Path,
    output_root: Path,
    log_root: Path,
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONWARNINGS"] = "ignore"
    statuses = []
    for target in targets:
        output = output_root / f"{target}_flat_baselines.json"
        if complete(output, target, args):
            print(f"gpu={gpu} target={target} status=skip-complete", flush=True)
            statuses.append((target, "skip-complete"))
            continue
        command = [
            sys.executable,
            str(script),
            "--feature-root",
            str(feature_root),
            "--target-subject",
            target,
            "--output",
            str(output),
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
            "--device",
            "cuda:0",
        ]
        log = log_root / f"{target}.log"
        print(f"gpu={gpu} target={target} status=running", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        status = "complete" if completed.returncode == 0 else f"failed:{completed.returncode}"
        print(f"gpu={gpu} target={target} status={status}", flush=True)
        statuses.append((target, status))
        if completed.returncode != 0:
            raise RuntimeError(f"{target} failed on GPU {gpu}; inspect {log}")
    return statuses


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    log_root = Path(args.log_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_flat_architecture_baselines_matb.py")
    targets = discover_subjects(feature_root)
    assignments = [targets[index :: len(args.gpus)] for index in range(len(args.gpus))]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                worker,
                gpu,
                assigned,
                script=script,
                feature_root=feature_root,
                output_root=output_root,
                log_root=log_root,
                args=args,
            )
            for gpu, assigned in zip(args.gpus, assignments, strict=True)
        ]
        statuses = [status for future in futures for status in future.result()]
    failed = [status for status in statuses if not status[1].startswith(("complete", "skip"))]
    print(
        f"subjects={len(statuses)} completed={len(statuses) - len(failed)} "
        f"failed={len(failed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
