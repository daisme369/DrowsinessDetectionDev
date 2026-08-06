from __future__ import annotations

import numpy as np

from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet
from drowsiness_detection.landmarks.geometric_features import LEFT_EYE_INDICES, MOUTH_INDICES, RIGHT_EYE_INDICES
from drowsiness_detection.training import build_frame_feature_dict, feature_vector_from_dict


def test_build_frame_feature_dict_contains_ear_mar_features() -> None:
    landmarks = _fake_landmarks()

    features = build_frame_feature_dict(landmarks)
    vector = feature_vector_from_dict(features, ["left_ear", "right_ear", "mean_ear", "mar", "mar_to_ear"])

    assert features["left_ear"] > 0
    assert features["right_ear"] > 0
    assert features["mean_ear"] > 0
    assert features["mar"] > 0
    assert vector.shape == (5,)


def _fake_landmarks() -> FaceLandmarkSet:
    points = np.zeros((478, 3), dtype=np.float32)
    left_eye = np.array(
        [
            [10.0, 10.0, 0.0],
            [12.0, 8.0, 0.0],
            [16.0, 8.0, 0.0],
            [20.0, 10.0, 0.0],
            [16.0, 12.0, 0.0],
            [12.0, 12.0, 0.0],
        ],
        dtype=np.float32,
    )
    right_eye = left_eye + np.array([30.0, 0.0, 0.0], dtype=np.float32)
    mouth = np.array(
        [
            [20.0, 45.0, 0.0],
            [24.0, 40.0, 0.0],
            [30.0, 38.0, 0.0],
            [36.0, 40.0, 0.0],
            [40.0, 45.0, 0.0],
            [36.0, 52.0, 0.0],
            [30.0, 54.0, 0.0],
            [24.0, 52.0, 0.0],
        ],
        dtype=np.float32,
    )
    for index, point in zip(LEFT_EYE_INDICES, left_eye):
        points[index] = point
    for index, point in zip(RIGHT_EYE_INDICES, right_eye):
        points[index] = point
    for index, point in zip(MOUTH_INDICES, mouth):
        points[index] = point
    return FaceLandmarkSet(
        normalized=points.copy(),
        pixel=points,
        image_width=100,
        image_height=100,
    )
