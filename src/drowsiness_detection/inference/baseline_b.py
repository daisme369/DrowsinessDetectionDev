from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from drowsiness_detection.inference.backend import ClassificationPrediction
from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet, FaceLandmarker, MediaPipeFaceMeshLandmarker
from drowsiness_detection.landmarks.geometric_features import LEFT_EYE_INDICES, MOUTH_INDICES, RIGHT_EYE_INDICES
from drowsiness_detection.models import load_geometric_classifier
from drowsiness_detection.preprocessing.roi_extraction import PixelBox, box_from_landmark_points
from drowsiness_detection.temporal.state_machine import DriverState
from drowsiness_detection.training import build_frame_feature_dict, feature_vector_from_dict
from drowsiness_detection.utils.config import BaselineBConfig


@dataclass(slots=True)
class BaselineBResult:
    timestamp_seconds: float
    face_detected: bool
    features: dict[str, float] | None
    prediction: ClassificationPrediction | None
    state: DriverState
    inference_latency_ms: float
    landmarks: FaceLandmarkSet | None = None
    left_eye_box: PixelBox | None = None
    right_eye_box: PixelBox | None = None
    mouth_box: PixelBox | None = None


class GeometricFeatureClassifierBackend:
    def __init__(self, config: BaselineBConfig) -> None:
        self.config = config
        self.model, metadata = load_geometric_classifier(config.inference.model_path)
        self.feature_names = list(metadata.get("feature_names") or config.model.feature_names)

    def predict(self, feature_dict: dict[str, float]) -> ClassificationPrediction:
        started = time.perf_counter()
        vector = feature_vector_from_dict(feature_dict, self.feature_names).reshape(1, -1)
        class_index = int(self.model.predict(vector)[0])
        probabilities = self._predict_probabilities(vector, class_index)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ClassificationPrediction(
            class_index=class_index,
            confidence=float(probabilities[class_index]),
            probabilities=[float(value) for value in probabilities],
            latency_ms=latency_ms,
        )

    def _predict_probabilities(self, vector: np.ndarray, class_index: int) -> list[float]:
        if hasattr(self.model, "predict_proba"):
            values = self.model.predict_proba(vector)[0]
            return [float(value) for value in values.tolist()]
        probabilities = [0.0] * self.config.model.num_classes
        probabilities[class_index] = 1.0
        return probabilities

    def close(self) -> None:
        return None


class BaselineBPipeline:
    """Frame-level geometric feature classifier baseline."""

    def __init__(
        self,
        config: BaselineBConfig,
        landmarker: FaceLandmarker | None = None,
        backend: GeometricFeatureClassifierBackend | None = None,
    ) -> None:
        self.config = config
        self.landmarker = landmarker or self._build_landmarker(config)
        self.backend = backend or GeometricFeatureClassifierBackend(config)

    @staticmethod
    def _build_landmarker(config: BaselineBConfig) -> FaceLandmarker:
        detector = config.landmarks.detector.lower()
        if detector != "mediapipe":
            raise ValueError(f"Unsupported landmark detector '{config.landmarks.detector}'")
        return MediaPipeFaceMeshLandmarker(config.landmarks)

    def process_frame(self, frame: Any, timestamp_seconds: float | None = None) -> BaselineBResult:
        timestamp = timestamp_seconds if timestamp_seconds is not None else time.monotonic()
        started = time.perf_counter()
        faces = self.landmarker.detect(frame)

        if not faces:
            return self._result(
                timestamp=timestamp,
                face_detected=False,
                features=None,
                prediction=None,
                state=DriverState.NO_FACE,
                started=started,
                landmarks=None,
            )

        landmarks = faces[0]
        try:
            feature_dict = build_frame_feature_dict(landmarks)
        except ValueError:
            return self._result(
                timestamp=timestamp,
                face_detected=True,
                features=None,
                prediction=None,
                state=DriverState.NO_FACE,
                started=started,
                landmarks=landmarks,
            )

        prediction = self.backend.predict(feature_dict)
        state = self._state_from_prediction(prediction)
        return self._result(
            timestamp=timestamp,
            face_detected=True,
            features=feature_dict,
            prediction=prediction,
            state=state,
            started=started,
            landmarks=landmarks,
        )

    def _state_from_prediction(self, prediction: ClassificationPrediction) -> DriverState:
        if prediction.confidence < self.config.inference.confidence_threshold:
            return DriverState.POSSIBLY_DROWSY
        if prediction.class_index == self.config.inference.closed_class_index:
            return DriverState.DROWSY
        return DriverState.ALERT

    def _result(
        self,
        *,
        timestamp: float,
        face_detected: bool,
        features: dict[str, float] | None,
        prediction: ClassificationPrediction | None,
        state: DriverState,
        started: float,
        landmarks: FaceLandmarkSet | None,
    ) -> BaselineBResult:
        left_eye_box = _box_for_indices(landmarks, LEFT_EYE_INDICES, padding=0.25) if landmarks is not None else None
        right_eye_box = _box_for_indices(landmarks, RIGHT_EYE_INDICES, padding=0.25) if landmarks is not None else None
        mouth_box = _box_for_indices(landmarks, MOUTH_INDICES, padding=0.20) if landmarks is not None else None
        latency_ms = (time.perf_counter() - started) * 1000.0
        return BaselineBResult(
            timestamp_seconds=timestamp,
            face_detected=face_detected,
            features=features,
            prediction=prediction,
            state=state,
            inference_latency_ms=latency_ms,
            landmarks=landmarks,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
            mouth_box=mouth_box,
        )

    def close(self) -> None:
        self.landmarker.close()
        self.backend.close()

    def __enter__(self) -> "BaselineBPipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _box_for_indices(
    landmarks: FaceLandmarkSet,
    indices: tuple[int, ...],
    *,
    padding: float,
) -> PixelBox | None:
    if landmarks.pixel.shape[0] <= max(indices):
        return None
    points = landmarks.pixel[list(indices)]
    return box_from_landmark_points(points, landmarks.image_width, landmarks.image_height, padding=padding)
