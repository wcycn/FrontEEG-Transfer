import numpy as np
import pytest
import torch
from evaluate_fronteegnet_device_session import filter_windows
from run_matb_fronteegnet_ablation_subject import VARIANTS, variant_spec

from fronteeg_transfer.baselines import EEGNet


@pytest.mark.parametrize("variant", VARIANTS)
def test_matb_ablation_variant_input_shape(variant: str) -> None:
    config, channels, _ = variant_spec(variant)
    model = EEGNet(config, classes=3)
    assert model(torch.randn(2, len(channels), 1_000)).shape == (2, 3)


def test_fixed_projection_has_expected_common_and_difference_components() -> None:
    config, _, _ = variant_spec("fixed_common_difference")
    model = EEGNet(config, classes=3)
    spatial = model.features[2]
    values = torch.tensor([[[[3.0], [1.0]]]])
    projected = spatial(values)
    assert projected.shape == (1, 2, 1, 1)
    assert torch.allclose(projected[:, 0], torch.tensor([[[2.0]]]))
    assert torch.allclose(projected[:, 1], torch.tensor([[[1.0]]]))


def test_device_filter_preserves_shape_and_removes_dc() -> None:
    sampling_rate = 250.0
    time = np.arange(1_000) / sampling_rate
    windows = np.stack(
        [
            np.stack(
                [
                    100.0 + np.sin(2 * np.pi * 10 * time),
                    -50.0 + np.sin(2 * np.pi * 12 * time),
                ]
            )
        ]
    ).astype(np.float32)
    filtered = filter_windows(windows, sampling_rate)
    assert filtered.shape == windows.shape
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(filtered.mean(axis=-1))) < 0.1
