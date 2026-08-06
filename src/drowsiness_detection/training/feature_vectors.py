from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet
from drowsiness_detection.landmarks.geometric_features import FaceGeometryFeatures, compute_face_geometry


DEFAULT_GEOMETRIC_FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "mean_ear",
    "ear_diff",
    "mar",
    "mar_to_ear",
    "pitch_degrees",
    "yaw_degrees",
    "roll_degrees",
]


def build_frame_feature_dict(landmarks: FaceLandmarkSet) -> dict[str, float]:
    features = compute_face_geometry(landmarks)
    if not has_required_frame_features(features):
        raise ValueError("Required EAR/MAR landmarks are missing")

    left_ear = _required(features.left_ear)
    right_ear = _required(features.right_ear)
    mean_ear = _required(features.ear)
    mar = _required(features.mar)
    pose = features.head_pose
    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mean_ear": mean_ear,
        "ear_diff": abs(left_ear - right_ear),
        "mar": mar,
        "mar_to_ear": mar / max(mean_ear, 1e-6),
        "pitch_degrees": 0.0 if pose is None else float(pose.pitch_degrees),
        "yaw_degrees": 0.0 if pose is None else float(pose.yaw_degrees),
        "roll_degrees": 0.0 if pose is None else float(pose.roll_degrees),
    }


def has_required_frame_features(features: FaceGeometryFeatures) -> bool:
    return features.left_ear is not None and features.right_ear is not None and features.ear is not None and features.mar is not None


def feature_vector_from_dict(
    feature_dict: Mapping[str, float],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray([float(feature_dict[name]) for name in feature_names], dtype=np.float32)


def feature_matrix_from_rows(
    rows: Sequence[Mapping[str, str | float | int]],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("Required feature value is missing")
    return float(value)
