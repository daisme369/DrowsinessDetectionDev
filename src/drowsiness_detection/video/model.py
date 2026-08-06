from __future__ import annotations

from typing import Any


def build_baseline_d_model(config: dict[str, Any]):
    import torch
    from torch import nn

    experiment = config.get("experiment", {})
    use_visual = bool(experiment.get("use_visual", True))
    use_behavior = bool(experiment.get("use_behavior", True))
    temporal_model = str(experiment.get("temporal_model", "lstm"))

    if temporal_model != "lstm":
        raise ValueError(f"Unsupported Baseline D temporal model: {temporal_model}")
    if not use_visual and not use_behavior:
        raise ValueError("At least one feature stream must be enabled")

    visual_encoder = ResNet18FrameEncoder(config) if use_visual else None
    visual_dim = int(config["model"]["cnn"].get("output_dimension", 512)) if use_visual else 0
    behavior_dim = int(config["model"]["fusion"].get("behavior_dimension", 2)) if use_behavior else 0
    return HybridResNetLSTMClassifier(
        visual_encoder=visual_encoder,
        visual_dim=visual_dim,
        behavior_dim=behavior_dim,
        hidden_size=int(config["model"]["lstm"]["hidden_size"]),
        num_layers=int(config["model"]["lstm"].get("num_layers", 1)),
        bidirectional=bool(config["model"]["lstm"].get("bidirectional", False)),
        dropout=float(config["model"]["classifier"]["dropout"]),
        num_classes=int(config["model"]["classifier"]["number_of_classes"]),
    )


class ResNet18FrameEncoder:
    def __new__(cls, config: dict[str, Any]):
        import torch
        from torch import nn
        from torchvision import models

        cnn_config = config["model"]["cnn"]
        pretrained = bool(cnn_config.get("pretrained", True))
        strict_pretrained = bool(cnn_config.get("strict_pretrained", False))
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = "DEFAULT"
        try:
            resnet = models.resnet18(weights=weights)
        except Exception:
            if pretrained and strict_pretrained:
                raise
            resnet = models.resnet18(weights=None)
        _ = torch
        layers = [(name, module) for name, module in resnet.named_children() if name != "fc"]
        layers.append(("flatten", nn.Flatten()))
        from collections import OrderedDict

        return nn.Sequential(OrderedDict(layers))


class HybridResNetLSTMClassifier:
    def __new__(
        cls,
        *,
        visual_encoder,
        visual_dim: int,
        behavior_dim: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int,
    ):
        import torch
        from torch import nn

        class _HybridResNetLSTMClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual_encoder = visual_encoder
                self.visual_dim = int(visual_dim)
                self.behavior_dim = int(behavior_dim)
                self.fused_dim = self.visual_dim + self.behavior_dim
                self.lstm = nn.LSTM(
                    input_size=self.fused_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                output_dim = hidden_size * (2 if bidirectional else 1)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(output_dim, num_classes)

            def forward(self, face_frames, behavior_features, valid_mask, *, return_debug: bool = False):
                features = []
                frame_embeddings = None
                if self.visual_encoder is not None:
                    batch_size, time_steps, channels, height, width = face_frames.shape
                    flat_frames = face_frames.reshape(batch_size * time_steps, channels, height, width)
                    frame_embeddings = self.visual_encoder(flat_frames).reshape(batch_size, time_steps, self.visual_dim)
                    features.append(frame_embeddings)
                if self.behavior_dim:
                    features.append(behavior_features[..., : self.behavior_dim])
                fused = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
                hidden_states, _ = self.lstm(fused)
                clip_embedding = masked_temporal_mean(hidden_states, valid_mask)
                logits = self.classifier(self.dropout(clip_embedding))
                if return_debug:
                    return {
                        "logits": logits,
                        "frame_embeddings": frame_embeddings,
                        "fused_sequence": fused,
                        "temporal_hidden_states": hidden_states,
                        "clip_embedding": clip_embedding,
                    }
                return logits

        return _HybridResNetLSTMClassifier()


def masked_temporal_mean(sequence, valid_mask):
    import torch

    mask = valid_mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (sequence * mask).sum(dim=1) / denominator


def freeze_visual_encoder(model, freeze: bool = True) -> None:
    encoder = getattr(model, "visual_encoder", None)
    if encoder is None:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze


def unfreeze_visual_stage(model, stage_name: str) -> None:
    encoder = getattr(model, "visual_encoder", None)
    if encoder is None:
        return
    for name, parameter in encoder.named_parameters():
        if name.startswith(stage_name):
            parameter.requires_grad = True
