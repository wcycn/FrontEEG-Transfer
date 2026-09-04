from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


class FixedMeanDifferenceSpatialProjection(nn.Module):
    """Project Fp1/Fp2 to fixed common and differential components."""

    def forward(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 4 or eeg.shape[2] != 2:
            raise ValueError(f"Expected [B,F,2,T], got {tuple(eeg.shape)}")
        common = 0.5 * (eeg[:, :, :1] + eeg[:, :, 1:2])
        difference = 0.5 * (eeg[:, :, :1] - eeg[:, :, 1:2])
        return torch.stack((common, difference), dim=2).flatten(1, 2)


@dataclass(frozen=True)
class EEGNetConfig:
    channels: int = 2
    samples: int = 800
    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16
    temporal_kernel: int = 63
    separable_kernel: int = 15
    dropout: float = 0.5
    spatial_mode: Literal["learned", "fixed_mean_difference"] = "learned"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class EEGNet(nn.Module):
    """Compact EEGNet-style model for Fp1/Fp2 windows."""

    def __init__(self, config: EEGNetConfig, classes: int = 4) -> None:
        super().__init__()
        self.config = config
        spatial_filters = config.temporal_filters * config.depth_multiplier
        if config.spatial_mode == "learned":
            spatial_projection: nn.Module = nn.Conv2d(
                config.temporal_filters,
                spatial_filters,
                kernel_size=(config.channels, 1),
                groups=config.temporal_filters,
                bias=False,
            )
        elif config.spatial_mode == "fixed_mean_difference":
            if config.channels != 2 or config.depth_multiplier != 2:
                raise ValueError(
                    "fixed_mean_difference requires channels=2 and depth_multiplier=2"
                )
            spatial_projection = FixedMeanDifferenceSpatialProjection()
        else:
            raise ValueError(f"Unknown spatial_mode: {config.spatial_mode}")
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                config.temporal_filters,
                kernel_size=(1, config.temporal_kernel),
                padding=(0, config.temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(config.temporal_filters),
            spatial_projection,
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(config.dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, config.separable_kernel),
                padding=(0, config.separable_kernel // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, config.separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.separable_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(config.separable_filters, classes)

    def forward_features(self, eeg: Tensor) -> Tensor:
        expected = (self.config.channels, self.config.samples)
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                f"Expected EEG [B,{expected[0]},{expected[1]}], got {tuple(eeg.shape)}"
            )
        return self.features(eeg.unsqueeze(1)).flatten(1)

    def forward(self, eeg: Tensor) -> Tensor:
        return self.head(self.forward_features(eeg))
