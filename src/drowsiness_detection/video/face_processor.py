from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from drowsiness_detection.landmarks.face_landmarker import FaceLandmarkSet, MediaPipeFaceMeshLandmarker
from drowsiness_detection.preprocessing.roi_extraction import PixelBox, box_from_landmark_points, clip_pixel_box
from drowsiness_detection.utils.config import LandmarkConfig
from drowsiness_detection.video.behavior_features import compute_behavior_features


@dataclass(slots=True)
class FaceFrameResult:
    face_bbox: tuple[int, int, int, int] | None
    landmarks: np.ndarray | None
    aligned_face: np.ndarray | None
    ear: float | None
    mar: float | None
    is_valid: bool
    failure_reason: str | None = None


class MediaPipeFaceProcessor:
    """MediaPipe-backed face, landmark, alignment, EAR, and MAR processor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        face_config = config["face"]
        self.landmarker = MediaPipeFaceMeshLandmarker(
            LandmarkConfig(
                model_asset_path=face_config["model_asset_path"],
                max_num_faces=1,
                min_detection_confidence=float(face_config["minimum_detection_confidence"]),
                min_face_presence_confidence=float(face_config["minimum_landmark_confidence"]),
                min_tracking_confidence=float(face_config["minimum_landmark_confidence"]),
                static_image_mode=False,
            )
        )
        self._last_bbox: PixelBox | None = None
        self._consecutive_missing = 0

    def process(self, frame_bgr: np.ndarray) -> FaceFrameResult:
        faces = self.landmarker.detect(frame_bgr)
        recovered_from_previous_bbox = False
        if not faces:
            faces = self._detect_from_previous_bbox(frame_bgr)
            recovered_from_previous_bbox = bool(faces)

        if not faces:
            self._consecutive_missing += 1
            return FaceFrameResult(None, None, None, None, None, False, "face_not_detected")

        landmark_set = faces[0]
        self._consecutive_missing = 0
        bbox = self._face_bbox(landmark_set)
        self._last_bbox = bbox
        behavior = compute_behavior_features(landmark_set, self.config)
        if not behavior.is_valid:
            return FaceFrameResult(
                (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                landmark_set.pixel,
                None,
                behavior.ear,
                behavior.mar,
                False,
                behavior.failure_reason,
            )

        aligned_face = self._aligned_face(frame_bgr, landmark_set)
        if aligned_face is None:
            return FaceFrameResult(
                (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                landmark_set.pixel,
                None,
                behavior.ear,
                behavior.mar,
                False,
                "alignment_or_crop_failed",
            )
        reason = "recovered_from_previous_bbox" if recovered_from_previous_bbox else None
        return FaceFrameResult(
            (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            landmark_set.pixel,
            aligned_face,
            behavior.ear,
            behavior.mar,
            True,
            reason,
        )

    def close(self) -> None:
        self.landmarker.close()

    def _detect_from_previous_bbox(self, frame_bgr: np.ndarray) -> list[FaceLandmarkSet]:
        if not self.config["missing_data"].get("use_previous_bbox", True):
            return []
        max_missing = int(self.config["missing_data"]["max_consecutive_missing_frames"])
        if self._last_bbox is None or self._consecutive_missing >= max_missing:
            return []
        box = clip_pixel_box(self._last_bbox, frame_bgr.shape[1], frame_bgr.shape[0])
        crop = frame_bgr[box.y1 : box.y2, box.x1 : box.x2]
        if crop.size == 0:
            return []
        faces = self.landmarker.detect(crop)
        adjusted: list[FaceLandmarkSet] = []
        for face in faces:
            pixel = face.pixel.copy()
            pixel[:, 0] += float(box.x1)
            pixel[:, 1] += float(box.y1)
            normalized = pixel.copy()
            normalized[:, 0] /= float(frame_bgr.shape[1])
            normalized[:, 1] /= float(frame_bgr.shape[0])
            adjusted.append(
                FaceLandmarkSet(
                    normalized=normalized,
                    pixel=pixel,
                    image_width=frame_bgr.shape[1],
                    image_height=frame_bgr.shape[0],
                    confidence=face.confidence,
                )
            )
        return adjusted

    def _face_bbox(self, landmarks: FaceLandmarkSet) -> PixelBox:
        margin = float(self.config["face"]["bbox_margin_ratio"])
        return box_from_landmark_points(landmarks.pixel, landmarks.image_width, landmarks.image_height, padding=margin)

    def _aligned_face(self, frame_bgr: np.ndarray, landmarks: FaceLandmarkSet) -> np.ndarray | None:
        if not self.config["face"].get("align", True):
            bbox = self._face_bbox(landmarks)
            crop = frame_bgr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
            return _resize_bgr_to_rgb(crop, self.config)

        matrix = _eye_alignment_matrix(landmarks.pixel, self.config, frame_bgr.shape[1], frame_bgr.shape[0])
        if matrix is None:
            return None
        aligned_frame = cv2.warpAffine(
            frame_bgr,
            matrix,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        transformed_landmarks = _transform_points(landmarks.pixel, matrix)
        aligned_set = FaceLandmarkSet(
            normalized=landmarks.normalized,
            pixel=transformed_landmarks,
            image_width=landmarks.image_width,
            image_height=landmarks.image_height,
            confidence=landmarks.confidence,
        )
        bbox = self._face_bbox(aligned_set)
        crop = aligned_frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        return _resize_bgr_to_rgb(crop, self.config)


def _eye_alignment_matrix(points: np.ndarray, config: dict[str, Any], image_width: int, image_height: int) -> np.ndarray | None:
    landmark_config = config["landmarks"]
    right_eye = points[landmark_config["right_eye"], :2]
    left_eye = points[landmark_config["left_eye"], :2]
    right_center = right_eye.mean(axis=0)
    left_center = left_eye.mean(axis=0)
    dx = float(left_center[0] - right_center[0])
    dy = float(left_center[1] - right_center[1])
    if abs(dx) + abs(dy) <= 1e-6:
        return None
    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = (float((right_center[0] + left_center[0]) / 2.0), float((right_center[1] + left_center[1]) / 2.0))
    _ = image_width, image_height
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xy1 = np.concatenate([points[:, :2], np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed_xy = xy1 @ matrix.T
    output = points.copy()
    output[:, :2] = transformed_xy
    return output


def _resize_bgr_to_rgb(crop_bgr: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    image_config = config["image"]
    resized = cv2.resize(
        crop_bgr,
        (int(image_config["width"]), int(image_config["height"])),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
