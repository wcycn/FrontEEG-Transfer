#!/usr/bin/env python3
"""Prepare paper-aligned MATB features for all valid COG-BCI subjects."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Raw-archive audit: sub-27 duplicates all MATB sessions of sub-17; sub-25
# and sub-28 share identical S1 MATB files but diverge in S2/S3, leaving
# participant identity ambiguous. Exclude all three before any fold is built.
EXCLUDED_SUBJECTS = {"sub-25", "sub-27", "sub-28"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_one(script: Path, archive: Path, output: Path, overwrite: bool) -> dict:
    command = [
        sys.executable,
        str(script),
        "--archive",
        str(archive),
        "--output-directory",
        str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    return {
        "subject": archive.stem,
        "sessions": len(report["sessions"]),
        "shapes": [session["shape"] for session in report["sessions"]],
    }


def main() -> None:
    args = parse_args()
    archive_root = Path(args.archive_directory).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    script = Path(__file__).with_name("prepare_cog_matb_paper_subject.py")
    archives = [
        archive
        for archive in sorted(archive_root.glob("sub-*.zip"))
        if archive.stem not in EXCLUDED_SUBJECTS
    ]
    if not archives:
        raise FileNotFoundError(f"no valid sub-*.zip archives under {archive_root}")
    reports = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {
            executor.submit(prepare_one, script, archive, output_root, args.overwrite): archive
            for archive in archives
        }
        for future in as_completed(pending):
            archive = pending[future]
            report = future.result()
            reports.append(report)
            print(
                f"prepared={archive.stem} sessions={report['sessions']} "
                f"shape={report['shapes'][0]}",
                flush=True,
            )
    reports.sort(key=lambda report: report["subject"])
    print(
        json.dumps(
            {
                "subjects": len(reports),
                "excluded": sorted(EXCLUDED_SUBJECTS),
                "reports": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
