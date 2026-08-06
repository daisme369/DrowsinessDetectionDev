from __future__ import annotations

from typing import Any

import cv2

from drowsiness_detection.inference.baseline_a import PipelineResult
from drowsiness_detection.inference.baseline_b import BaselineBResult
from drowsiness_detection.landmarks.geometric_features import (
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
)
from drowsiness_detection.utils.profiling import PerformanceSnapshot


STATE_COLORS = {
    "alert": (80, 220, 80),
    "possibly_drowsy": (0, 220, 255),
    "drowsy": (0, 0, 255),
    "recovery": (255, 180, 0),
    "no_face": (180, 180, 180),
}


def draw_baseline_overlay(
    frame: Any,
    result: PipelineResult,
    performance: PerformanceSnapshot | None = None,
    show_landmarks: bool = True,
    show_metrics: bool = True,
) -> Any:
    output = frame.copy()
    state = result.state.value
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.rectangle(output, (0, 0), (output.shape[1], 90), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        f"State: {state.upper()}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    if show_metrics:
        metric_text = _format_metrics(result, performance)
        cv2.putText(
            output,
            metric_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    if show_landmarks and result.landmarks is not None:
        _draw_feature_points(output, result)
    return output


def draw_baseline_b_overlay(
    frame: Any,
    result: BaselineBResult,
    performance: PerformanceSnapshot | None = None,
    show_metrics: bool = True,
) -> Any:
    output = frame.copy()
    state = result.state.value
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.rectangle(output, (0, 0), (output.shape[1], 90), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        f"State: {state.upper()}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    if show_metrics:
        cv2.putText(
            output,
            _format_baseline_b_metrics(result, performance),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    _draw_baseline_b_regions(output, result)
    return output


def _format_metrics(result: PipelineResult, performance: PerformanceSnapshot | None) -> str:
    features = result.features
    ear = "n/a" if features is None or features.ear is None else f"{features.ear:.3f}"
    mar = "n/a" if features is None or features.mar is None else f"{features.mar:.3f}"
    pitch = "n/a" if features is None or features.pitch_degrees is None else f"{features.pitch_degrees:.1f}"
    fps = 0.0 if performance is None else performance.effective_fps
    return (
        f"EAR {ear} | MAR {mar} | pitch {pitch} deg | "
        f"PERCLOS {result.evidence.perclos:.2f} | FPS {fps:.1f}"
    )


def _format_baseline_b_metrics(result: BaselineBResult, performance: PerformanceSnapshot | None) -> str:
    features = result.features or {}
    ear = "n/a" if "mean_ear" not in features else f"{features['mean_ear']:.3f}"
    mar = "n/a" if "mar" not in features else f"{features['mar']:.3f}"
    predicted = "n/a" if result.prediction is None else str(result.prediction.class_index)
    confidence = 0.0 if result.prediction is None else result.prediction.confidence
    fps = 0.0 if performance is None else performance.effective_fps
    return f"EAR {ear} | MAR {mar} | pred {predicted} ({confidence:.2f}) | FPS {fps:.1f}"


def _draw_baseline_b_regions(frame: Any, result: BaselineBResult) -> None:
    for box, color in (
        (result.left_eye_box, (80, 220, 80)),
        (result.right_eye_box, (80, 220, 80)),
        (result.mouth_box, (80, 180, 255)),
    ):
        if box is None:
            continue
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)

    if result.landmarks is None:
        return
    points = result.landmarks.pixel
    for indices, color in (
        (LEFT_EYE_INDICES, (80, 220, 80)),
        (RIGHT_EYE_INDICES, (80, 220, 80)),
        (MOUTH_INDICES, (80, 180, 255)),
    ):
        for index in indices:
            if index >= points.shape[0]:
                continue
            x, y = points[index, :2].astype(int)
            cv2.circle(frame, (int(x), int(y)), 2, color, thickness=-1)


def _draw_feature_points(frame: Any, result: PipelineResult) -> None:
    assert result.landmarks is not None
    points = result.landmarks.pixel
    for indices, color in (
        (LEFT_EYE_INDICES, (80, 220, 80)),
        (RIGHT_EYE_INDICES, (80, 220, 80)),
        (MOUTH_INDICES, (80, 180, 255)),
    ):
        for index in indices:
            if index >= points.shape[0]:
                continue
            x, y = points[index, :2].astype(int)
            cv2.circle(frame, (int(x), int(y)), 2, color, thickness=-1)
