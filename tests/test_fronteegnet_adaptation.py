import numpy as np
from run_matb_fronteegnet_adaptation_subject import (
    configure_trainable_parameters,
    nested_nonoverlapping_indices,
    set_adaptation_train_mode,
)

from fronteeg_transfer.baselines import EEGNet, EEGNetConfig


def test_nested_nonoverlapping_indices_are_balanced_and_nested() -> None:
    labels = np.repeat(np.arange(3), 20)
    selections = nested_nonoverlapping_indices(labels, maximum=4, seed=7)
    assert set(selections[1]).issubset(selections[2])
    assert set(selections[2]).issubset(selections[4])
    for budget, indices in selections.items():
        selected_labels = labels[indices]
        assert np.bincount(selected_labels, minlength=3).tolist() == [budget] * 3
        for label in range(3):
            within = sorted(index for index in indices if labels[index] == label)
            assert all(
                right - left > 1
                for left, right in zip(within, within[1:], strict=False)
            )


def test_adaptation_scopes_have_expected_parameter_counts() -> None:
    config = EEGNetConfig(channels=2, samples=1000, temporal_kernel=125, separable_kernel=31)
    expected = {"head": 51, "partial": 803, "full": 1915, "scratch": 1915}
    for mode, count in expected.items():
        model = EEGNet(config, classes=3)
        assert configure_trainable_parameters(model, mode) == count


def test_head_and_partial_keep_frozen_batch_norm_in_evaluation_mode() -> None:
    config = EEGNetConfig(channels=2, samples=1000, temporal_kernel=125, separable_kernel=31)
    for mode in ("head", "partial"):
        model = EEGNet(config, classes=3)
        configure_trainable_parameters(model, mode)
        set_adaptation_train_mode(model, mode)
        assert not model.features[1].training
        assert not model.features[3].training
        assert not model.features[9].training
