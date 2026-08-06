from __future__ import annotations

from drowsiness_detection.temporal import DriverState, EyeStateTemporalAggregator
from drowsiness_detection.utils.config import BaselineBTemporalConfig


def test_eye_classifier_hysteresis_keeps_brief_closure_from_drowsy() -> None:
    aggregator = EyeStateTemporalAggregator(
        BaselineBTemporalConfig(
            closed_probability_threshold=0.6,
            open_probability_threshold=0.4,
            eye_closed_duration_seconds=1.0,
            perclos_min_samples=10,
        )
    )

    aggregator.update(0.0, 0.8)
    assert aggregator.state == DriverState.POSSIBLY_DROWSY

    aggregator.update(0.3, 0.2)
    assert aggregator.state == DriverState.ALERT


def test_eye_classifier_prolonged_closure_triggers_drowsy() -> None:
    aggregator = EyeStateTemporalAggregator(
        BaselineBTemporalConfig(
            closed_probability_threshold=0.6,
            open_probability_threshold=0.4,
            eye_closed_duration_seconds=1.0,
            perclos_min_samples=10,
        )
    )

    aggregator.update(0.0, 0.9)
    evidence = aggregator.update(1.1, 0.7)

    assert evidence.prolonged_eye_closure
    assert aggregator.state == DriverState.DROWSY


def test_eye_classifier_missing_face_grace_period() -> None:
    aggregator = EyeStateTemporalAggregator(BaselineBTemporalConfig(no_face_grace_seconds=1.0))

    aggregator.update_no_face(2.0)
    assert aggregator.state == DriverState.ALERT

    aggregator.update_no_face(3.2)
    assert aggregator.state == DriverState.NO_FACE
