import torch

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig


def test_eegnet_four_class_shape() -> None:
    model = EEGNet(EEGNetConfig(samples=200), classes=4)
    assert model(torch.randn(3, 2, 200)).shape == (3, 4)


def test_eegnet_three_class_matb_shape() -> None:
    model = EEGNet(EEGNetConfig(samples=1_000, temporal_kernel=125), classes=3)
    assert model(torch.randn(2, 2, 1_000)).shape == (2, 3)


def test_eegnet_fixed_spatial_shape_and_parameter_reduction() -> None:
    learned = EEGNet(EEGNetConfig(samples=1_000, temporal_kernel=125), classes=3)
    fixed = EEGNet(
        EEGNetConfig(
            samples=1_000,
            temporal_kernel=125,
            spatial_mode="fixed_mean_difference",
        ),
        classes=3,
    )
    assert fixed(torch.randn(2, 2, 1_000)).shape == (2, 3)
    assert sum(parameter.numel() for parameter in fixed.parameters()) < sum(
        parameter.numel() for parameter in learned.parameters()
    )
