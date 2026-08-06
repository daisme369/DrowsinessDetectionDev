from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import cv2


def _parse_source(source: int | str) -> int | str:
    if isinstance(source, int):
        return source
    value = str(source)
    if value.isdigit():
        return int(value)
    return value


def _is_stream_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https", "rtsp", "rtmp"}


@dataclass(slots=True)
class CapturedFrame:
    frame: Any
    timestamp_seconds: float
    frame_index: int


class OpenCVFrameSource:
    """Small OpenCV-backed frame source for cameras and video files."""

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        target_fps: int | None = None,
    ) -> None:
        self.source = _parse_source(source)
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0

    def open(self) -> None:
        if self._capture is not None:
            return
        if isinstance(self.source, str) and not _is_stream_url(self.source) and not Path(self.source).exists():
            raise FileNotFoundError(f"Video source does not exist: {self.source}")
        capture = cv2.VideoCapture(self.source)
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.target_fps:
            capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open OpenCV source: {self.source}")
        self._capture = capture

    def read(self) -> CapturedFrame | None:
        self.open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        captured = CapturedFrame(
            frame=frame,
            timestamp_seconds=time.monotonic(),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return captured

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCVFrameSource":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
