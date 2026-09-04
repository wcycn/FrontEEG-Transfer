from __future__ import annotations

from run_matb_feature_ablation_subject import feature_indices


def test_feature_masks_match_two_channel_four_band_layout() -> None:
    assert feature_indices("full") == list(range(8))
    assert feature_indices("only_theta") == [0, 4]
    assert feature_indices("only_high_beta") == [3, 7]
    assert feature_indices("drop_alpha") == [0, 2, 3, 4, 6, 7]
    assert feature_indices("only_fp1") == [0, 1, 2, 3]
    assert feature_indices("only_fp2") == [4, 5, 6, 7]
