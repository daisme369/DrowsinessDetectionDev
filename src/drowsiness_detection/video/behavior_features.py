from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drowsiness_detection.landmarks.geometric_features import eye_aspect_ratio
from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet


@dataclass(frozen=True, slots=True)
class BehaviorFeatureResult:
    ear: float | None
    mar: float | None
    is_valid: bool
    failure_reason: str | None = None


def compute_behavior_features(landmarks: FaceLandmarkSet, config: dict[str, Any]) -> BehaviorFeatureResult:
    landmark_config = config["landmarks"]
    feature_config = config["behavior_features"]
    try:
        right_eye = _points(landmarks.pixel, landmark_config["right_eye"])
        left_eye = _points(landmarks.pixel, landmark_config["left_eye"])
        ear = float(np.mean([eye_aspect_ratio(right_eye), eye_aspect_ratio(left_eye)]))
        mar = _four_point_mar(
            landmarks.pixel,
            left_corner=int(landmark_config["mouth_left_corner"]),
            right_corner=int(landmark_config["mouth_right_corner"]),
            upper_lip=int(landmark_config["upper_inner_lip"]),
            lower_lip=int(landmark_config["lower_inner_lip"]),
            epsilon=float(feature_config["epsilon"]),
        )
    except (IndexError, ValueError) as error:
        return BehaviorFeatureResult(None, None, False, f"missing_required_landmarks:{error}")

    ear_min, ear_max = [float(value) for value in feature_config["ear_valid_range"]]
    mar_min, mar_max = [float(value) for value in feature_config["mar_valid_range"]]
    if not ear_min <= ear <= ear_max:
        return BehaviorFeatureResult(ear, mar, False, f"ear_out_of_range:{ear:.6f}")
    if not mar_min <= mar <= mar_max:
        return BehaviorFeatureResult(ear, mar, False, f"mar_out_of_range:{mar:.6f}")
    return BehaviorFeatureResult(ear, mar, True, None)


def _points(points: np.ndarray, indices: list[int]) -> np.ndarray:
    if points.shape[0] <= max(indices):
        raise IndexError(f"landmark_count={points.shape[0]} max_required={max(indices)}")
    return points[indices]


def _four_point_mar(
    points: np.ndarray,
    *,
    left_corner: int,
    right_corner: int,
    upper_lip: int,
    lower_lip: int,
    epsilon: float,
) -> float:
    required = [left_corner, right_corner, upper_lip, lower_lip]
    if points.shape[0] <= max(required):
        raise IndexError(f"landmark_count={points.shape[0]} max_required={max(required)}")
    horizontal = np.linalg.norm(points[left_corner, :2] - points[right_corner, :2])
    vertical = np.linalg.norm(points[upper_lip, :2] - points[lower_lip, :2])
    return float(vertical / max(horizontal, epsilon))
