from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedYoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def parse_yolo_line(line: str) -> NormalizedYoloBox:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO columns, got {len(parts)}: {line!r}")
    class_id, x_center, y_center, width, height = parts
    return NormalizedYoloBox(
        class_id=int(float(class_id)),
        x_center=float(x_center),
        y_center=float(y_center),
        width=float(width),
        height=float(height),
    )


def normalized_yolo_to_pixel_box(box: NormalizedYoloBox, image_width: int, image_height: int) -> PixelBox:
    x1 = round((box.x_center - (box.width / 2.0)) * image_width)
    y1 = round((box.y_center - (box.height / 2.0)) * image_height)
    x2 = round((box.x_center + (box.width / 2.0)) * image_width)
    y2 = round((box.y_center + (box.height / 2.0)) * image_height)
    return PixelBox(x1=x1, y1=y1, x2=x2, y2=y2)


def box_from_landmark_points(points: Any, image_width: int, image_height: int, padding: float = 0.0) -> PixelBox:
    if points is None or len(points) == 0:
        return PixelBox(0, 0, 0, 0)
    x_values = points[:, 0]
    y_values = points[:, 1]
    box = PixelBox(
        x1=int(x_values.min()),
        y1=int(y_values.min()),
        x2=int(x_values.max()),
        y2=int(y_values.max()),
    )
    return expand_pixel_box(box, image_width, image_height, padding)


def expand_pixel_box(box: PixelBox, image_width: int, image_height: int, padding: float) -> PixelBox:
    if padding <= 0:
        return clip_pixel_box(box, image_width, image_height)
    pad_x = round(box.width * padding)
    pad_y = round(box.height * padding)
    return clip_pixel_box(
        PixelBox(
            x1=box.x1 - pad_x,
            y1=box.y1 - pad_y,
            x2=box.x2 + pad_x,
            y2=box.y2 + pad_y,
        ),
        image_width,
        image_height,
    )


def clip_pixel_box(box: PixelBox, image_width: int, image_height: int) -> PixelBox:
    return PixelBox(
        x1=max(0, min(image_width, box.x1)),
        y1=max(0, min(image_height, box.y1)),
        x2=max(0, min(image_width, box.x2)),
        y2=max(0, min(image_height, box.y2)),
    )


def eye_band_box(face_box: PixelBox, eye_band_y_min: float, eye_band_y_max: float) -> PixelBox:
    y_min = _clamp01(eye_band_y_min)
    y_max = _clamp01(eye_band_y_max)
    if y_max <= y_min:
        raise ValueError("eye_band_y_max must be greater than eye_band_y_min")
    y1 = face_box.y1 + round(face_box.height * y_min)
    y2 = face_box.y1 + round(face_box.height * y_max)
    return PixelBox(face_box.x1, y1, face_box.x2, y2)


def extract_roi_from_box(
    image: Any,
    box: PixelBox,
    *,
    crop_mode: str,
    padding: float = 0.0,
    eye_band_y_min: float = 0.18,
    eye_band_y_max: float = 0.55,
    min_box_size: int = 1,
) -> tuple[Any | None, PixelBox | None]:
    if image is None:
        return None, None
    image_height, image_width = image.shape[:2]
    face_box = expand_pixel_box(box, image_width, image_height, padding)
    mode = crop_mode.lower()
    if mode == "face":
        roi_box = face_box
    elif mode == "eye_band":
        roi_box = clip_pixel_box(eye_band_box(face_box, eye_band_y_min, eye_band_y_max), image_width, image_height)
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    if roi_box.width < min_box_size or roi_box.height < min_box_size:
        return None, None
    return image[roi_box.y1 : roi_box.y2, roi_box.x1 : roi_box.x2].copy(), roi_box


def extract_roi_from_yolo_box(
    image: Any,
    yolo_box: NormalizedYoloBox,
    *,
    crop_mode: str,
    padding: float = 0.0,
    eye_band_y_min: float = 0.18,
    eye_band_y_max: float = 0.55,
    min_box_size: int = 1,
) -> tuple[Any | None, PixelBox | None]:
    image_height, image_width = image.shape[:2]
    box = normalized_yolo_to_pixel_box(yolo_box, image_width, image_height)
    return extract_roi_from_box(
        image,
        box,
        crop_mode=crop_mode,
        padding=padding,
        eye_band_y_min=eye_band_y_min,
        eye_band_y_max=eye_band_y_max,
        min_box_size=min_box_size,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
