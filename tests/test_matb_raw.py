from __future__ import annotations

import numpy as np
from prepare_cog_matb_raw_subject import fp1_fp2_indices, sliding_windows


def test_fp1_fp2_indices_preserve_requested_order() -> None:
    assert fp1_fp2_indices(["Cz", "FP2", "Fp1", "TP10"]) == [2, 1]


def test_sliding_windows_match_four_second_half_overlap_geometry() -> None:
    data = np.arange(2 * 2_000).reshape(2, 2_000)
    windows = sliding_windows(data, window=1_000, step=500)

    assert windows.shape == (3, 2, 1_000)
    np.testing.assert_array_equal(windows[0], data[:, :1_000])
    np.testing.assert_array_equal(windows[1], data[:, 500:1_500])
    np.testing.assert_array_equal(windows[2], data[:, 1_000:])
