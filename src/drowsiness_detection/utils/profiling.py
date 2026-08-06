from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class PerformanceSnapshot:
    effective_fps: float
    mean_latency_ms: float
    last_latency_ms: float
    sample_count: int


class RollingPerformanceMeter:
    def __init__(self, window_size: int = 120) -> None:
        self._latencies_ms: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def update(self, timestamp_seconds: float, latency_ms: float) -> PerformanceSnapshot:
        self._timestamps.append(timestamp_seconds)
        self._latencies_ms.append(latency_ms)
        return self.snapshot()

    def snapshot(self) -> PerformanceSnapshot:
        sample_count = len(self._latencies_ms)
        if sample_count == 0:
            return PerformanceSnapshot(0.0, 0.0, 0.0, 0)
        elapsed = self._timestamps[-1] - self._timestamps[0] if len(self._timestamps) > 1 else 0.0
        fps = (len(self._timestamps) - 1) / elapsed if elapsed > 1e-6 else 0.0
        mean_latency = sum(self._latencies_ms) / sample_count
        return PerformanceSnapshot(
            effective_fps=fps,
            mean_latency_ms=mean_latency,
            last_latency_ms=self._latencies_ms[-1],
            sample_count=sample_count,
        )

