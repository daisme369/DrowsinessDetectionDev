from __future__ import annotations

from pathlib import Path

from drowsiness_detection.landmarks.face_landmarker import resolve_model_asset_path


def test_resolve_relative_model_asset_path() -> None:
    path = resolve_model_asset_path("artifacts/models/face_landmarker.task")

    assert path == Path.cwd() / "artifacts" / "models" / "face_landmarker.task"

