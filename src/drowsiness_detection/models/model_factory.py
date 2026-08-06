from __future__ import annotations

from typing import Any

import torch
from torch import nn

from drowsiness_detection.models.eye_classifier import SmallEyeStateCNN
from drowsiness_detection.utils.config import BaselineBModelConfig


def build_eye_classifier(config: BaselineBModelConfig) -> nn.Module:
    architecture = config.architecture.lower()
    input_channels = 1 if config.grayscale else 3
    if architecture in {"small_cnn", "small", "custom_cnn"}:
        return SmallEyeStateCNN(
            num_classes=config.num_classes,
            input_channels=input_channels,
            dropout=config.dropout,
        )
    if architecture == "mobilenet_v3_small":
        return _build_mobilenet_v3_small(config, input_channels)
    if architecture in {"shufflenet_v2_x0_5", "shufflenet_v2"}:
        return _build_shufflenet_v2_x0_5(config, input_channels)
    raise ValueError(f"Unsupported eye classifier architecture: {config.architecture}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint_state(model: nn.Module, checkpoint_path: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state_dict}


def _build_mobilenet_v3_small(config: BaselineBModelConfig, input_channels: int) -> nn.Module:
    from torchvision import models

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    _replace_first_conv_if_needed(model.features[0][0], input_channels)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config.num_classes)
    return model


def _build_shufflenet_v2_x0_5(config: BaselineBModelConfig, input_channels: int) -> nn.Module:
    from torchvision import models

    weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if config.pretrained else None
    model = models.shufflenet_v2_x0_5(weights=weights)
    _replace_first_conv_if_needed(model.conv1[0], input_channels)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    return model


def _replace_first_conv_if_needed(conv: nn.Conv2d, input_channels: int) -> None:
    if conv.in_channels == input_channels:
        return
    replacement = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3 and input_channels == 1:
            replacement.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out")
        if replacement.bias is not None:
            replacement.bias.zero_()
    conv.weight = replacement.weight
    conv.in_channels = input_channels
    if conv.bias is not None:
        conv.bias = replacement.bias
