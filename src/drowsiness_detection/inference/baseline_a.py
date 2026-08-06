from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from drowsiness_detection.landmarks.face_landmarker import FaceLandmarker, MediaPipeFaceMeshLandmarker
from drowsiness_detection.landmarks.geometric_features import FaceGeometryFeatures, compute_face_geometry
from drowsiness_detection.temporal.state_machine import DriverState, StateEvidence, TemporalStateMachine
from drowsiness_detection.utils.config import BaselineAConfig


@dataclass(slots=True)
class PipelineResult:
    timestamp_seconds: float
    face_detected: bool
    features: FaceGeometryFeatures | None
    state: DriverState
    evidence: StateEvidence
    inference_latency_ms: float
    landmarks: Any | None = None


class BaselineAPipeline:
    """Landmark-only drowsiness baseline with temporal aggregation."""

    def __init__(self, config: BaselineAConfig, landmarker: FaceLandmarker | None = None) -> None:
        self.config = config
        self.landmarker = landmarker or self._build_landmarker(config)
        self.temporal = TemporalStateMachine(config.thresholds)

    @staticmethod
    def _build_landmarker(config: BaselineAConfig) -> FaceLandmarker:
        detector = config.landmarks.detector.lower()
        if detector != "mediapipe":
            raise ValueError(f"Unsupported landmark detector '{config.landmarks.detector}'")
        return MediaPipeFaceMeshLandmarker(config.landmarks)

    def process_frame(self, frame: Any, timestamp_seconds: float | None = None) -> PipelineResult:
        timestamp = timestamp_seconds if timestamp_seconds is not None else time.monotonic()
        started = time.perf_counter()
        faces = self.landmarker.detect(frame)

        if not faces:
            evidence = self.temporal.update_no_face(timestamp)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return PipelineResult(
                timestamp_seconds=timestamp,
                face_detected=False,
                features=None,
                state=self.temporal.state,
                evidence=evidence,
                inference_latency_ms=latency_ms,
                landmarks=None,
            )

        landmarks = faces[0]
        features = compute_face_geometry(landmarks)
        evidence = self.temporal.update(timestamp, features)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PipelineResult(
            timestamp_seconds=timestamp,
            face_detected=True,
            features=features,
            state=self.temporal.state,
            evidence=evidence,
            inference_latency_ms=latency_ms,
            landmarks=landmarks,
        )

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "BaselineAPipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

