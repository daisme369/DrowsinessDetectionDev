from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Protocol

import cv2
import numpy as np

from drowsiness_detection.utils.config import LandmarkConfig


@dataclass(slots=True)
class FaceLandmarkSet:
    """Normalized and pixel face landmarks for one detected face."""

    normalized: np.ndarray
    pixel: np.ndarray
    image_width: int
    image_height: int
    confidence: float | None = None


class FaceLandmarker(Protocol):
    def detect(self, frame_bgr: Any) -> list[FaceLandmarkSet]:
        ...

    def close(self) -> None:
        ...


class MediaPipeFaceMeshLandmarker:
    """MediaPipe Face Mesh adapter used by Baseline A."""

    def __init__(self, config: LandmarkConfig) -> None:
        legacy_face_mesh = _load_legacy_face_mesh_module()
        if legacy_face_mesh is not None:
            self._impl: FaceLandmarker = _LegacyFaceMeshLandmarker(config, legacy_face_mesh)
        else:
            self._impl = _TasksFaceLandmarker(config)

    def detect(self, frame_bgr: Any) -> list[FaceLandmarkSet]:
        return self._impl.detect(frame_bgr)

    def close(self) -> None:
        self._impl.close()


class _LegacyFaceMeshLandmarker:
    def __init__(self, config: LandmarkConfig, face_mesh_module: Any) -> None:
        self._mesh = face_mesh_module.FaceMesh(
            static_image_mode=config.static_image_mode,
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, frame_bgr: Any) -> list[FaceLandmarkSet]:
        if frame_bgr is None:
            return []
        image_height, image_width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        faces: list[FaceLandmarkSet] = []
        for face_landmarks in result.multi_face_landmarks:
            normalized = np.array(
                [(point.x, point.y, point.z) for point in face_landmarks.landmark],
                dtype=np.float32,
            )
            pixel = normalized.copy()
            pixel[:, 0] *= float(image_width)
            pixel[:, 1] *= float(image_height)
            faces.append(
                FaceLandmarkSet(
                    normalized=normalized,
                    pixel=pixel,
                    image_width=image_width,
                    image_height=image_height,
                    confidence=None,
                )
            )
        return faces

    def close(self) -> None:
        self._mesh.close()


class _TasksFaceLandmarker:
    def __init__(self, config: LandmarkConfig) -> None:
        mediapipe, tasks_python, vision = _load_tasks_modules()
        model_path = resolve_model_asset_path(config.model_asset_path)
        if not model_path.exists():
            raise RuntimeError(
                "MediaPipe Tasks Face Landmarker requires a model asset file. "
                f"Expected: {model_path}. Download it with "
                "`python scripts/download_face_landmarker_model.py`, or set "
                "`landmarks.model_asset_path` in configs/baseline_a.yaml."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=config.max_num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mediapipe
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: Any) -> list[FaceLandmarkSet]:
        if frame_bgr is None:
            return []
        image_height, image_width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return []

        faces: list[FaceLandmarkSet] = []
        for face_landmarks in result.face_landmarks:
            normalized = np.array(
                [(point.x, point.y, point.z) for point in face_landmarks],
                dtype=np.float32,
            )
            pixel = normalized.copy()
            pixel[:, 0] *= float(image_width)
            pixel[:, 1] *= float(image_height)
            faces.append(
                FaceLandmarkSet(
                    normalized=normalized,
                    pixel=pixel,
                    image_width=image_width,
                    image_height=image_height,
                    confidence=None,
                )
            )
        return faces

    def close(self) -> None:
        self._landmarker.close()


def resolve_model_asset_path(model_asset_path: str) -> Path:
    path = Path(model_asset_path).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_legacy_face_mesh_module() -> Any | None:
    for module_name in (
        "mediapipe.python.solutions.face_mesh",
        "mediapipe.solutions.face_mesh",
    ):
        try:
            return import_module(module_name)
        except ImportError:
            continue

    try:
        mediapipe = import_module("mediapipe")
        solutions = getattr(mediapipe, "solutions")
        return solutions.face_mesh
    except (ImportError, AttributeError):
        return None


def _load_tasks_modules() -> tuple[Any, Any, Any]:
    try:
        mediapipe = import_module("mediapipe")
        tasks_python = import_module("mediapipe.tasks.python")
        vision = import_module("mediapipe.tasks.python.vision")
        return mediapipe, tasks_python, vision
    except ImportError as tasks_error:
        python_info = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "MediaPipe Face Landmarker is required for Baseline A. This installed "
            "`mediapipe` package exposes neither the legacy `solutions.face_mesh` "
            "API nor the current `mediapipe.tasks.python.vision` API. Make sure "
            "there is no local file/folder named `mediapipe`, then reinstall a "
            "supported Google MediaPipe build with `python -m pip install --upgrade "
            "mediapipe`. "
            f"Current Python: {python_info} at {sys.executable}."
        ) from tasks_error
