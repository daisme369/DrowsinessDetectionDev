from __future__ import annotations

import numpy as np


def target_timestamps(duration_seconds: float, target_fps: float) -> np.ndarray:
    if duration_seconds <= 0:
        return np.asarray([], dtype=np.float32)
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    step = 1.0 / target_fps
    return np.arange(0.0, duration_seconds, step, dtype=np.float32)
