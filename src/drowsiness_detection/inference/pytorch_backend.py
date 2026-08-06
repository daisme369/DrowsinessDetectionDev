from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from drowsiness_detection.inference.backend import ClassificationPrediction
from drowsiness_detection.models import build_eye_classifier
from drowsiness_detection.models.model_factory import load_checkpoint_state
from drowsiness_detection.training.datasets import preprocess_roi_for_torch
from drowsiness_detection.utils.config import BaselineBModelConfig


class PyTorchEyeClassifierBackend:
    def __init__(
        self,
        model_config: BaselineBModelConfig,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_eye_classifier(model_config)
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Baseline B checkpoint does not exist: {checkpoint}. "
                "Train it with `python scripts/train_baseline_b.py`."
            )
        load_checkpoint_state(self.model, str(checkpoint), map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, roi_bgr: Any) -> ClassificationPrediction:
        started = time.perf_counter()
        tensor = preprocess_roi_for_torch(
            roi_bgr,
            input_size=self.model_config.input_size,
            grayscale=self.model_config.grayscale,
            normalize_mean=self.model_config.normalize_mean,
            normalize_std=self.model_config.normalize_std,
        )
        batch = tensor.unsqueeze(0).to(self.device)
        logits = self.model(batch)
        probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
        probabilities = [float(value) for value in probabilities_tensor.tolist()]
        class_index = int(probabilities_tensor.argmax().item())
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ClassificationPrediction(
            class_index=class_index,
            confidence=float(probabilities[class_index]),
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        return None
