from __future__ import annotations

from drowsiness_detection.landmarks.geometric_features import FaceGeometryFeatures
from drowsiness_detection.temporal import DriverState, TemporalStateMachine
from drowsiness_detection.utils.config import ThresholdConfig


def features(ear: float, mar: float = 0.2) -> FaceGeometryFeatures:
    return FaceGeometryFeatures(
        ear=ear,
        left_ear=ear,
        right_ear=ear,
        mar=mar,
        head_pose=None,
    )


def test_brief_eye_closure_is_not_drowsy() -> None:
    machine = TemporalStateMachine(
        ThresholdConfig(
            ear_closed=0.21,
            ear_open=0.25,
            eye_closed_duration_seconds=1.0,
            perclos_min_samples=10,
        )
    )

    machine.update(0.0, features(0.18))
    assert machine.state == DriverState.ALERT

    machine.update(0.4, features(0.30))
    assert machine.state == DriverState.ALERT


def test_prolonged_eye_closure_triggers_drowsy_state() -> None:
    machine = TemporalStateMachine(
        ThresholdConfig(
            ear_closed=0.21,
            ear_open=0.25,
            eye_closed_duration_seconds=1.0,
            perclos_min_samples=10,
        )
    )

    machine.update(0.0, features(0.18))
    evidence = machine.update(1.2, features(0.18))

    assert evidence.prolonged_eye_closure
    assert machine.state == DriverState.DROWSY


def test_drowsy_state_requires_recovery_hold_before_alert() -> None:
    machine = TemporalStateMachine(
        ThresholdConfig(
            ear_closed=0.21,
            ear_open=0.25,
            eye_closed_duration_seconds=1.0,
            recovery_hold_seconds=2.0,
            perclos_min_samples=10,
            ema_alpha=1.0,
        )
    )

    machine.update(0.0, features(0.18))
    machine.update(1.2, features(0.18))
    assert machine.state == DriverState.DROWSY

    machine.update(1.3, features(0.30))
    assert machine.state == DriverState.RECOVERY

    machine.update(2.2, features(0.30))
    assert machine.state == DriverState.RECOVERY

    machine.update(3.4, features(0.30))
    assert machine.state == DriverState.ALERT


def test_missing_face_has_grace_period() -> None:
    machine = TemporalStateMachine(ThresholdConfig(no_face_grace_seconds=1.0))

    machine.update_no_face(10.0)
    assert machine.state == DriverState.ALERT

    machine.update_no_face(11.1)
    assert machine.state == DriverState.NO_FACE

