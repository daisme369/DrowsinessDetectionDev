from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from drowsiness_detection.landmarks.geometric_features import FaceGeometryFeatures
from drowsiness_detection.utils.config import ThresholdConfig


class DriverState(str, Enum):
    ALERT = "alert"
    POSSIBLY_DROWSY = "possibly_drowsy"
    DROWSY = "drowsy"
    RECOVERY = "recovery"
    NO_FACE = "no_face"


@dataclass(slots=True)
class StateEvidence:
    eye_closed: bool = False
    eye_closed_duration_seconds: float = 0.0
    prolonged_eye_closure: bool = False
    yawn: bool = False
    yawn_duration_seconds: float = 0.0
    prolonged_yawn: bool = False
    abnormal_head_pitch: bool = False
    head_pitch_duration_seconds: float = 0.0
    perclos: float = 0.0
    perclos_window_count: int = 0
    perclos_high: bool = False
    face_missing_duration_seconds: float = 0.0
    evidence_score: int = 0


@dataclass(slots=True)
class _WindowSample:
    timestamp_seconds: float
    eye_closed: bool


class TemporalStateMachine:
    """Duration and sliding-window decision logic for Baseline A."""

    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds
        self.state = DriverState.ALERT
        self._eye_closed = False
        self._yawn = False
        self._abnormal_pitch = False
        self._eye_closed_since: float | None = None
        self._yawn_since: float | None = None
        self._abnormal_pitch_since: float | None = None
        self._face_missing_since: float | None = None
        self._recovery_since: float | None = None
        self._samples: deque[_WindowSample] = deque()
        self._ema_ear: float | None = None
        self._ema_mar: float | None = None

    def update_no_face(self, timestamp_seconds: float) -> StateEvidence:
        if self._face_missing_since is None:
            self._face_missing_since = timestamp_seconds
        missing_duration = max(0.0, timestamp_seconds - self._face_missing_since)
        if missing_duration >= self.thresholds.no_face_grace_seconds:
            self.state = DriverState.NO_FACE
        return StateEvidence(face_missing_duration_seconds=missing_duration)

    def update(self, timestamp_seconds: float, features: FaceGeometryFeatures) -> StateEvidence:
        self._face_missing_since = None
        self._update_eye_state(timestamp_seconds, features.ear)
        self._update_yawn_state(timestamp_seconds, features.mar)
        self._update_head_state(timestamp_seconds, features.pitch_degrees)

        self._samples.append(_WindowSample(timestamp_seconds, self._eye_closed))
        self._trim_window(timestamp_seconds)

        evidence = self._build_evidence(timestamp_seconds)
        self._transition(timestamp_seconds, evidence)
        return evidence

    def _update_eye_state(self, timestamp_seconds: float, ear: float | None) -> None:
        if ear is None:
            return
        self._ema_ear = ear if self._ema_ear is None else self._smooth(self._ema_ear, ear)
        ear_value = self._ema_ear
        if self._eye_closed:
            if ear_value >= self.thresholds.ear_open:
                self._eye_closed = False
                self._eye_closed_since = None
        elif ear_value <= self.thresholds.ear_closed:
            self._eye_closed = True
            self._eye_closed_since = timestamp_seconds

    def _update_yawn_state(self, timestamp_seconds: float, mar: float | None) -> None:
        if mar is None:
            return
        self._ema_mar = mar if self._ema_mar is None else self._smooth(self._ema_mar, mar)
        mar_value = self._ema_mar
        if self._yawn:
            if mar_value <= self.thresholds.mar_recovery:
                self._yawn = False
                self._yawn_since = None
        elif mar_value >= self.thresholds.mar_yawn:
            self._yawn = True
            self._yawn_since = timestamp_seconds

    def _update_head_state(self, timestamp_seconds: float, pitch_degrees: float | None) -> None:
        if pitch_degrees is None:
            return
        is_abnormal = (
            pitch_degrees >= self.thresholds.head_pitch_down_degrees
            or pitch_degrees <= self.thresholds.head_pitch_up_degrees
        )
        if is_abnormal and not self._abnormal_pitch:
            self._abnormal_pitch = True
            self._abnormal_pitch_since = timestamp_seconds
        elif not is_abnormal:
            self._abnormal_pitch = False
            self._abnormal_pitch_since = None

    def _trim_window(self, timestamp_seconds: float) -> None:
        cutoff = timestamp_seconds - self.thresholds.perclos_window_seconds
        while self._samples and self._samples[0].timestamp_seconds < cutoff:
            self._samples.popleft()

    def _build_evidence(self, timestamp_seconds: float) -> StateEvidence:
        eye_duration = self._duration(timestamp_seconds, self._eye_closed_since, self._eye_closed)
        yawn_duration = self._duration(timestamp_seconds, self._yawn_since, self._yawn)
        head_duration = self._duration(timestamp_seconds, self._abnormal_pitch_since, self._abnormal_pitch)
        closed_count = sum(1 for sample in self._samples if sample.eye_closed)
        sample_count = len(self._samples)
        perclos = closed_count / sample_count if sample_count else 0.0

        prolonged_eye = eye_duration >= self.thresholds.eye_closed_duration_seconds
        prolonged_yawn = yawn_duration >= self.thresholds.yawn_duration_seconds
        prolonged_head = head_duration >= self.thresholds.head_pose_duration_seconds
        perclos_high = (
            sample_count >= self.thresholds.perclos_min_samples
            and perclos >= self.thresholds.perclos_threshold
        )
        score = int(prolonged_eye) + int(prolonged_yawn) + int(prolonged_head) + int(perclos_high)
        return StateEvidence(
            eye_closed=self._eye_closed,
            eye_closed_duration_seconds=eye_duration,
            prolonged_eye_closure=prolonged_eye,
            yawn=self._yawn,
            yawn_duration_seconds=yawn_duration,
            prolonged_yawn=prolonged_yawn,
            abnormal_head_pitch=self._abnormal_pitch,
            head_pitch_duration_seconds=head_duration,
            perclos=perclos,
            perclos_window_count=sample_count,
            perclos_high=perclos_high,
            evidence_score=score,
        )

    def _transition(self, timestamp_seconds: float, evidence: StateEvidence) -> None:
        high_confidence_pair = evidence.prolonged_eye_closure and evidence.abnormal_head_pitch
        if evidence.prolonged_eye_closure or high_confidence_pair or evidence.evidence_score >= 2:
            self.state = DriverState.DROWSY
            self._recovery_since = None
            return
        if evidence.evidence_score == 1:
            self.state = DriverState.POSSIBLY_DROWSY
            self._recovery_since = None
            return

        if self.state == DriverState.DROWSY:
            self.state = DriverState.RECOVERY
            self._recovery_since = timestamp_seconds
            return
        if self.state == DriverState.RECOVERY:
            assert self._recovery_since is not None
            if timestamp_seconds - self._recovery_since >= self.thresholds.recovery_hold_seconds:
                self.state = DriverState.ALERT
            return

        self.state = DriverState.ALERT
        self._recovery_since = None

    def _smooth(self, previous: float, current: float) -> float:
        alpha = self.thresholds.ema_alpha
        return (alpha * current) + ((1.0 - alpha) * previous)

    @staticmethod
    def _duration(timestamp_seconds: float, start: float | None, active: bool) -> float:
        if not active or start is None:
            return 0.0
        return max(0.0, timestamp_seconds - start)

