import pytest

from fronteeg_transfer.cog_bci import (
    NBackBlock,
    cog_channel_indices,
    event_code,
    nback_blocks,
    nonoverlapping_windows,
)


def test_cog_channel_selection_uses_documented_global_exclusions() -> None:
    channels = ["Fp2", "ECG1", "Cz", "F3", "FP1"]
    assert cog_channel_indices(channels, "fp1_fp2") == [4, 0]
    assert cog_channel_indices(channels, "all_eeg") == [0, 3, 4]


def test_event_code_accepts_numeric_annotations_only() -> None:
    assert event_code(6011) == 6011
    assert event_code("6012") == 6012
    assert event_code("S 6211") == 6211
    assert event_code("comment 6011 6012") is None


def test_nback_blocks_follow_official_markers() -> None:
    blocks = nback_blocks(
        [
            (100, "6011"),
            (500, "6021"),
            (1100, "6012"),
            (1200, "6111"),
            (2200, "6112"),
            (2300, "6211"),
            (3300, "6212"),
        ]
    )
    assert [(block.label, block.start_sample, block.stop_sample) for block in blocks] == [
        (0, 100, 1100),
        (1, 1200, 2200),
        (2, 2300, 3300),
    ]
    windows = nonoverlapping_windows(blocks, sample_rate=100, window_seconds=5)
    assert [(window.label, window.block_index) for window in windows] == [
        (0, 0),
        (0, 0),
        (1, 1),
        (1, 1),
        (2, 2),
        (2, 2),
    ]


def test_nback_blocks_reject_unclosed_or_reordered_events() -> None:
    with pytest.raises(ValueError, match="Unclosed"):
        nback_blocks([(100, "6011")])
    with pytest.raises(ValueError, match="sorted"):
        nback_blocks([(200, "6011"), (100, "6012")])


def test_nback_blocks_can_recover_documented_repeated_zero_back_starts() -> None:
    events = [(100, "6011"), (200, "6011"), (1_000, "6012")]
    with pytest.raises(ValueError, match="duplicate"):
        nback_blocks(events)
    assert nback_blocks(events, repeated_start="replace") == [NBackBlock(0, 200, 1_000)]
