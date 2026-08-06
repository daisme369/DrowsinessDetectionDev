from __future__ import annotations

import numpy as np

from drowsiness_detection.preprocessing.roi_extraction import (
    NormalizedYoloBox,
    PixelBox,
    eye_band_box,
    extract_roi_from_box,
    normalized_yolo_to_pixel_box,
)


def test_normalized_yolo_to_pixel_box() -> None:
    box = normalized_yolo_to_pixel_box(
        NormalizedYoloBox(class_id=1, x_center=0.5, y_center=0.5, width=0.5, height=0.25),
        image_width=200,
        image_height=100,
    )

    assert box == PixelBox(x1=50, y1=38, x2=150, y2=62)


def test_eye_band_box_uses_upper_face_region() -> None:
    face = PixelBox(x1=10, y1=20, x2=110, y2=220)

    eye_box = eye_band_box(face, 0.2, 0.5)

    assert eye_box == PixelBox(x1=10, y1=60, x2=110, y2=120)


def test_extract_eye_band_roi_clips_to_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    roi, roi_box = extract_roi_from_box(
        image,
        PixelBox(x1=-20, y1=10, x2=80, y2=110),
        crop_mode="eye_band",
        padding=0.0,
        eye_band_y_min=0.2,
        eye_band_y_max=0.5,
        min_box_size=4,
    )

    assert roi is not None
    assert roi_box == PixelBox(x1=0, y1=28, x2=80, y2=55)
    assert roi.shape[:2] == (27, 80)
