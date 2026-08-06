from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    class_index: int
    confidence: float
    probabilities: list[float]
    latency_ms: float

    def probability_for(self, class_index: int) -> float:
        if 0 <= class_index < len(self.probabilities):
            return self.probabilities[class_index]
        return 0.0


class InferenceBackend(Protocol):
    def predict(self, roi_bgr: Any) -> ClassificationPrediction:
        ...

    def close(self) -> None:
        ...
