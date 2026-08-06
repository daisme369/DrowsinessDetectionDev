from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from drowsiness_detection.inference.backend import ClassificationPrediction
from drowsiness_detection.training.datasets import preprocess_roi_to_numpy
from drowsiness_detection.utils.config import BaselineBModelConfig


class ONNXRuntimeEyeClassifierBackend:
    def __init__(
        self,
        model_config: BaselineBModelConfig,
        onnx_path: str | Path,
        *,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is required for Baseline B ONNX inference. "
                "Install it with `python -m pip install onnxruntime`."
            ) from error

        model_path = Path(onnx_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Baseline B ONNX model does not exist: {model_path}. "
                "Export it with `python scripts/export_baseline_b_onnx.py`."
            )
        self.model_config = model_config
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, roi_bgr: Any) -> ClassificationPrediction:
        started = time.perf_counter()
        array = preprocess_roi_to_numpy(
            roi_bgr,
            input_size=self.model_config.input_size,
            grayscale=self.model_config.grayscale,
            normalize_mean=self.model_config.normalize_mean,
            normalize_std=self.model_config.normalize_std,
        )
        logits = self.session.run(None, {self.input_name: array[None, ...]})[0][0]
        probabilities = _softmax(logits)
        class_index = int(np.argmax(probabilities))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ClassificationPrediction(
            class_index=class_index,
            confidence=float(probabilities[class_index]),
            probabilities=[float(value) for value in probabilities.tolist()],
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.maximum(exp.sum(), 1e-12)
