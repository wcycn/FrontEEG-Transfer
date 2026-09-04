#!/usr/bin/env bash
set -euo pipefail

# Official COG-BCI record: 29 subject archives, 31.7 GB in total.
# curl --continue-at makes this safe to interrupt and resume.
destination="${1:-data/cog_bci_raw}"
workers="${COG_BCI_WORKERS:-4}"
mkdir -p "$destination"

download_subject() {
    local subject="$1"
    local archive="sub-${subject}.zip"
    local destination_path="$destination/$archive"
    if python3 -c 'import sys, zipfile; raise SystemExit(not zipfile.is_zipfile(sys.argv[1]))' \
        "$destination_path" 2>/dev/null; then
        echo "skip complete $archive"
        return
    fi
    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 10 \
        --retry-all-errors \
        --output "$destination_path" \
        "https://zenodo.org/records/7413650/files/$archive?download=1"
}

export -f download_subject
export destination
seq -w 1 29 | xargs --max-args=1 --max-procs="$workers" -r bash -c 'download_subject "$1"' _

curl \
    --fail \
    --location \
    --continue-at - \
    --retry 10 \
    --retry-all-errors \
    --output "$destination/COG-BCI_info.pdf" \
    "https://zenodo.org/records/7413650/files/COG-BCI_info.pdf?download=1"
