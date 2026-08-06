from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drowsiness_detection.temporal.state_machine import DriverState
from drowsiness_detection.utils.config import BaselineBTemporalConfig


@dataclass(slots=True)
class EyeClassifierEvidence:
    eye_closed_probability: float = 0.0
    eye_closed: bool = False
    uncertain: bool = False
    eye_closed_duration_seconds: float = 0.0
    prolonged_eye_closure: bool = False
    perclos: float = 0.0
    perclos_window_count: int = 0
    perclos_high: bool = False
    face_missing_duration_seconds: float = 0.0
    evidence_score: int = 0


@dataclass(slots=True)
class _EyeWindowSample:
    timestamp_seconds: float
    eye_closed: bool


class EyeStateTemporalAggregator:
    """Temporal logic for Baseline B eye-state probabilities."""

    def __init__(self, config: BaselineBTemporalConfig) -> None:
        self.config = config
        self.state = DriverState.ALERT
        self._eye_closed = False
        self._eye_closed_since: float | None = None
        self._face_missing_since: float | None = None
        self._recovery_since: float | None = None
        self._samples: deque[_EyeWindowSample] = deque()

    def update_no_face(self, timestamp_seconds: float) -> EyeClassifierEvidence:
        if self._face_missing_since is None:
            self._face_missing_since = timestamp_seconds
        missing_duration = max(0.0, timestamp_seconds - self._face_missing_since)
        if missing_duration >= self.config.no_face_grace_seconds:
            self.state = DriverState.NO_FACE
        return EyeClassifierEvidence(face_missing_duration_seconds=missing_duration)

    def update(self, timestamp_seconds: float, eye_closed_probability: float) -> EyeClassifierEvidence:
        self._face_missing_since = None
        probability = max(0.0, min(1.0, float(eye_closed_probability)))
        uncertain = self.config.open_probability_threshold < probability < self.config.closed_probability_threshold

        if self._eye_closed:
            if probability <= self.config.open_probability_threshold:
                self._eye_closed = False
                self._eye_closed_since = None
        elif probability >= self.config.closed_probability_threshold:
            self._eye_closed = True
            self._eye_closed_since = timestamp_seconds

        self._samples.append(_EyeWindowSample(timestamp_seconds, self._eye_closed))
        self._trim_window(timestamp_seconds)
        evidence = self._build_evidence(timestamp_seconds, probability, uncertain)
        self._transition(timestamp_seconds, evidence)
        return evidence

    def _trim_window(self, timestamp_seconds: float) -> None:
        cutoff = timestamp_seconds - self.config.perclos_window_seconds
        while self._samples and self._samples[0].timestamp_seconds < cutoff:
            self._samples.popleft()

    def _build_evidence(
        self,
        timestamp_seconds: float,
        eye_closed_probability: float,
        uncertain: bool,
    ) -> EyeClassifierEvidence:
        duration = self._duration(timestamp_seconds, self._eye_closed_since, self._eye_closed)
        sample_count = len(self._samples)
        closed_count = sum(1 for sample in self._samples if sample.eye_closed)
        perclos = closed_count / sample_count if sample_count else 0.0
        prolonged = duration >= self.config.eye_closed_duration_seconds
        perclos_high = sample_count >= self.config.perclos_min_samples and perclos >= self.config.perclos_threshold
        score = int(prolonged) + int(perclos_high)
        return EyeClassifierEvidence(
            eye_closed_probability=eye_closed_probability,
            eye_closed=self._eye_closed,
            uncertain=uncertain,
            eye_closed_duration_seconds=duration,
            prolonged_eye_closure=prolonged,
            perclos=perclos,
            perclos_window_count=sample_count,
            perclos_high=perclos_high,
            evidence_score=score,
        )

    def _transition(self, timestamp_seconds: float, evidence: EyeClassifierEvidence) -> None:
        if evidence.prolonged_eye_closure or evidence.perclos_high:
            self.state = DriverState.DROWSY
            self._recovery_since = None
            return
        if evidence.eye_closed or evidence.uncertain:
            self.state = DriverState.POSSIBLY_DROWSY
            self._recovery_since = None
            return
        if self.state == DriverState.DROWSY:
            self.state = DriverState.RECOVERY
            self._recovery_since = timestamp_seconds
            return
        if self.state == DriverState.RECOVERY:
            assert self._recovery_since is not None
            if timestamp_seconds - self._recovery_since >= self.config.recovery_hold_seconds:
                self.state = DriverState.ALERT
            return
        self.state = DriverState.ALERT
        self._recovery_since = None

    @staticmethod
    def _duration(timestamp_seconds: float, start: float | None, active: bool) -> float:
        if not active or start is None:
            return 0.0
        return max(0.0, timestamp_seconds - start)
