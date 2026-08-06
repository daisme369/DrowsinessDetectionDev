from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_landmarker import FaceLandmarkSet
from .head_pose import HeadPose, estimate_head_pose


LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (61, 81, 13, 311, 291, 402, 14, 178)


@dataclass(slots=True)
class FaceGeometryFeatures:
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    mar: float | None
    head_pose: HeadPose | None

    @property
    def pitch_degrees(self) -> float | None:
        return None if self.head_pose is None else self.head_pose.pitch_degrees


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a[:2] - point_b[:2]))


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Compute the classic 6-point Eye Aspect Ratio."""

    if points.shape[0] != 6:
        raise ValueError("EAR requires exactly 6 eye points")
    horizontal = euclidean_distance(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    vertical_one = euclidean_distance(points[1], points[5])
    vertical_two = euclidean_distance(points[2], points[4])
    return (vertical_one + vertical_two) / (2.0 * horizontal)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """Compute mouth opening using three vertical distances and one width."""

    if points.shape[0] != 8:
        raise ValueError("MAR requires exactly 8 mouth points")
    width = euclidean_distance(points[0], points[4])
    if width <= 1e-6:
        return 0.0
    vertical_center = euclidean_distance(points[2], points[6])
    vertical_left = euclidean_distance(points[1], points[7])
    vertical_right = euclidean_distance(points[3], points[5])
    return (vertical_center + vertical_left + vertical_right) / (3.0 * width)


def _safe_index(points: np.ndarray, indices: tuple[int, ...]) -> np.ndarray | None:
    if points.shape[0] <= max(indices):
        return None
    return points[list(indices)]


def compute_face_geometry(landmarks: FaceLandmarkSet) -> FaceGeometryFeatures:
    left_points = _safe_index(landmarks.pixel, LEFT_EYE_INDICES)
    right_points = _safe_index(landmarks.pixel, RIGHT_EYE_INDICES)
    mouth_points = _safe_index(landmarks.pixel, MOUTH_INDICES)

    left_ear = eye_aspect_ratio(left_points) if left_points is not None else None
    right_ear = eye_aspect_ratio(right_points) if right_points is not None else None
    ear_values = [value for value in (left_ear, right_ear) if value is not None]
    ear = float(np.mean(ear_values)) if ear_values else None
    mar = mouth_aspect_ratio(mouth_points) if mouth_points is not None else None
    pose = estimate_head_pose(landmarks)

    return FaceGeometryFeatures(
        ear=ear,
        left_ear=left_ear,
        right_ear=right_ear,
        mar=mar,
        head_pose=pose,
    )

