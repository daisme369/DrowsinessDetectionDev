from __future__ import annotations

import torch
from torch import nn


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class SmallEyeStateCNN(nn.Module):
    """Compact CNN baseline for eye/upper-face state classification."""

    def __init__(self, num_classes: int = 2, input_channels: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(input_channels, 16, stride=2),
            DepthwiseSeparableConv(16, 24, stride=2),
            DepthwiseSeparableConv(24, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
