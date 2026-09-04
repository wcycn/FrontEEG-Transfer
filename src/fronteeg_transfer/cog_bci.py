from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

NBACK_BLOCK_MARKERS = {
    0: (6011, 6012),
    1: (6111, 6112),
    2: (6211, 6212),
}

_MISSING_FOR_SOME_SUBJECTS = {"cz"}


@dataclass(frozen=True)
class NBackBlock:
    label: int
    start_sample: int
    stop_sample: int


@dataclass(frozen=True)
class EEGWindow:
    label: int
    block_index: int
    start_sample: int
    stop_sample: int


def canonical_channel_name(name: str) -> str:
    """Normalize channel spellings without changing their documented identity."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def cog_channel_indices(channel_names: Iterable[str], channel_set: str) -> list[int]:
    """Select a fixed COG-BCI channel setting from documented channel names.

    ``all_eeg`` removes ECG and Cz. Cz was not recorded for participants 1--9,
    so dropping it globally gives every subject the same sensor geometry. This
    is a collection-level rule from the dataset documentation, not a channel
    search on held-out EEG.
    """
    names = list(channel_names)
    canonical = [canonical_channel_name(name) for name in names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate canonical COG-BCI channel names")
    lookup = {name: index for index, name in enumerate(canonical)}
    if channel_set == "fp1_fp2":
        missing = [name for name in ("fp1", "fp2") if name not in lookup]
        if missing:
            raise ValueError(f"Fp1/Fp2 setting is missing channels: {missing}")
        return [lookup["fp1"], lookup["fp2"]]
    if channel_set == "all_eeg":
        selected = [
            index
            for index, name in enumerate(canonical)
            if not name.startswith("ecg") and name not in _MISSING_FOR_SOME_SUBJECTS
        ]
        if not selected:
            raise ValueError("No scalp EEG channels remain after documented exclusions")
        return selected
    raise ValueError(f"Unknown COG-BCI channel_set={channel_set!r}")


def event_code(description: str | int) -> int | None:
    """Extract one official numeric trigger code from an EEGLAB annotation."""
    if isinstance(description, int):
        return description
    values = re.findall(r"\d+", str(description))
    if len(values) != 1:
        return None
    return int(values[0])


def nback_blocks(
    events: Iterable[tuple[int, str | int]],
    *,
    repeated_start: str = "error",
) -> list[NBackBlock]:
    """Recover labelled N-back block intervals from official start/end triggers.

    COG-BCI zero-back files contain an extra start trigger just before each of
    the three block starts. ``repeated_start='replace'`` uses the last start
    before its end marker, retaining three non-overlapping labelled blocks.
    Strict mode remains the default to surface this issue in other datasets.
    """
    if repeated_start not in {"error", "replace"}:
        raise ValueError("repeated_start must be 'error' or 'replace'")
    starts = {start: label for label, (start, _) in NBACK_BLOCK_MARKERS.items()}
    ends = {end: label for label, (_, end) in NBACK_BLOCK_MARKERS.items()}
    open_blocks: dict[int, int] = {}
    recovered: list[NBackBlock] = []
    previous_sample = -1
    for sample, description in events:
        if sample < previous_sample:
            raise ValueError("Events must be sorted by sample")
        previous_sample = sample
        code = event_code(description)
        if code in starts:
            label = starts[code]
            if label in open_blocks and repeated_start == "error":
                raise ValueError(f"Nested or duplicate N-back start for label {label}")
            open_blocks[label] = sample
        elif code in ends:
            label = ends[code]
            if label not in open_blocks:
                raise ValueError(f"N-back end without start for label {label}")
            start_sample = open_blocks.pop(label)
            if sample <= start_sample:
                raise ValueError(f"Non-positive N-back block for label {label}")
            recovered.append(NBackBlock(label, start_sample, sample))
    if open_blocks:
        raise ValueError(f"Unclosed N-back blocks: {sorted(open_blocks)}")
    return sorted(recovered, key=lambda block: block.start_sample)


def nonoverlapping_windows(
    blocks: Iterable[NBackBlock],
    *,
    sample_rate: float,
    window_seconds: float = 5.0,
) -> list[EEGWindow]:
    """Create fixed windows strictly inside labelled blocks, never across boundaries."""
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate and window_seconds must be positive")
    window_samples = round(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError("window has no samples")
    windows: list[EEGWindow] = []
    for block_index, block in enumerate(blocks):
        for start_sample in range(
            block.start_sample,
            block.stop_sample - window_samples + 1,
            window_samples,
        ):
            windows.append(
                EEGWindow(
                    label=block.label,
                    block_index=block_index,
                    start_sample=start_sample,
                    stop_sample=start_sample + window_samples,
                )
            )
    return windows
