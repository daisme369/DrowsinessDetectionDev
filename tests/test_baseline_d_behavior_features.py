from __future__ import annotations

import numpy as np

from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet
from drowsiness_detection.video.behavior_features import compute_behavior_features


def test_ear_mar_are_scale_invariant_for_synthetic_landmarks() -> None:
    landmarks = _landmarks(scale=1.0)
    scaled_landmarks = _landmarks(scale=2.0)

    result = compute_behavior_features(landmarks, _config())
    scaled_result = compute_behavior_features(scaled_landmarks, _config())

    assert result.is_valid
    assert scaled_result.is_valid
    assert result.ear == scaled_result.ear
    assert result.mar == scaled_result.mar
    assert result.ear == 0.5
    assert round(result.mar, 6) == 0.4


def test_zero_width_mouth_is_marked_invalid() -> None:
    landmarks = _landmarks(scale=1.0)
    landmarks.pixel[61, :2] = landmarks.pixel[291, :2]

    result = compute_behavior_features(landmarks, _config())

    assert not result.is_valid
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("mar_out_of_range")


def _landmarks(*, scale: float) -> FaceLandmarkSet:
    points = np.zeros((468, 3), dtype=np.float32)
    right_eye = [33, 160, 158, 133, 153, 144]
    left_eye = [362, 385, 387, 263, 373, 380]
    eye_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    points[right_eye] = eye_points * scale
    points[left_eye] = (eye_points + np.asarray([10.0, 0.0, 0.0], dtype=np.float32)) * scale
    points[61] = np.asarray([0.0, 10.0, 0.0], dtype=np.float32) * scale
    points[291] = np.asarray([10.0, 10.0, 0.0], dtype=np.float32) * scale
    points[13] = np.asarray([5.0, 8.0, 0.0], dtype=np.float32) * scale
    points[14] = np.asarray([5.0, 12.0, 0.0], dtype=np.float32) * scale
    return FaceLandmarkSet(normalized=points.copy(), pixel=points, image_width=100, image_height=100)


def _config() -> dict:
    return {
        "landmarks": {
            "right_eye": [33, 160, 158, 133, 153, 144],
            "left_eye": [362, 385, 387, 263, 373, 380],
            "mouth_left_corner": 61,
            "mouth_right_corner": 291,
            "upper_inner_lip": 13,
            "lower_inner_lip": 14,
        },
        "behavior_features": {
            "ear_valid_range": [0.05, 0.60],
            "mar_valid_range": [0.00, 1.50],
            "epsilon": 1e-6,
        },
    }
