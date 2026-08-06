from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .face_landmarker import FaceLandmarkSet


HEAD_POSE_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class HeadPose:
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


def estimate_head_pose(landmarks: FaceLandmarkSet) -> HeadPose | None:
    if landmarks.pixel.shape[0] <= max(HEAD_POSE_INDICES.values()):
        return None

    image_points = np.array(
        [landmarks.pixel[index, :2] for index in HEAD_POSE_INDICES.values()],
        dtype=np.float64,
    )
    focal_length = float(landmarks.image_width)
    center = (landmarks.image_width / 2.0, landmarks.image_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return HeadPose(
        yaw_degrees=float(yaw),
        pitch_degrees=float(pitch),
        roll_degrees=float(roll),
    )

