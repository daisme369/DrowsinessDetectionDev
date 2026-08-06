from __future__ import annotations

import numpy as np

from drowsiness_detection.landmarks.geometric_features import eye_aspect_ratio, mouth_aspect_ratio


def test_eye_aspect_ratio_is_larger_for_open_eye() -> None:
    open_eye = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [8.0, 2.0],
            [10.0, 0.0],
            [8.0, -2.0],
            [2.0, -2.0],
        ]
    )
    closed_eye = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.3],
            [8.0, 0.3],
            [10.0, 0.0],
            [8.0, -0.3],
            [2.0, -0.3],
        ]
    )

    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed_eye)
    assert eye_aspect_ratio(open_eye) == 0.4


def test_mouth_aspect_ratio_increases_with_open_mouth() -> None:
    mostly_closed = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.3],
            [5.0, 0.4],
            [8.0, 0.3],
            [10.0, 0.0],
            [8.0, -0.3],
            [5.0, -0.4],
            [2.0, -0.3],
        ]
    )
    open_mouth = mostly_closed.copy()
    open_mouth[[1, 2, 3], 1] = 2.0
    open_mouth[[5, 6, 7], 1] = -2.0

    assert mouth_aspect_ratio(open_mouth) > mouth_aspect_ratio(mostly_closed)

