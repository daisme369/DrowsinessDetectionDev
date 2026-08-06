from __future__ import annotations

import numpy as np
from pathlib import Path

from drowsiness_detection.video.clips import build_clip_records
from drowsiness_detection.video.preprocess import ProcessedVideoSummary
from drowsiness_detection.video.splits import SplitManifest


def test_clip_generation_uses_sequence_length_stride_and_discards_tail() -> None:
    npz_path = _artifact_path("S01_alert_full.npz")
    np.savez_compressed(
        npz_path,
        face_frames=np.zeros((96, 224, 224, 3), dtype=np.uint8),
        timestamps=np.arange(96, dtype=np.float32) / 5.0,
        ear=np.full(96, 0.3, dtype=np.float32),
        mar=np.full(96, 0.2, dtype=np.float32),
        valid_mask=np.ones(96, dtype=bool),
    )
    summary = ProcessedVideoSummary(
        video_id="S01_alert",
        subject_id="01",
        label_id=0,
        npz_path=str(npz_path),
        sampled_frame_count=96,
        valid_frame_count=96,
        valid_frame_ratio=1.0,
        maximum_missing_run=0,
    )

    clips, rejected = build_clip_records([summary], _split_manifest(), _config())

    assert rejected == []
    assert [clip.start_frame for clip in clips] == [0, 28, 56]
    assert [clip.end_frame for clip in clips] == [39, 67, 95]
    assert all(clip.label_id == 0 and clip.subject_id == "01" and clip.split == "train" for clip in clips)


def test_clip_generation_rejects_long_missing_run() -> None:
    valid_mask = np.ones(40, dtype=bool)
    valid_mask[10:15] = False
    npz_path = _artifact_path("S01_alert_missing.npz")
    np.savez_compressed(
        npz_path,
        face_frames=np.zeros((40, 224, 224, 3), dtype=np.uint8),
        timestamps=np.arange(40, dtype=np.float32) / 5.0,
        ear=np.full(40, 0.3, dtype=np.float32),
        mar=np.full(40, 0.2, dtype=np.float32),
        valid_mask=valid_mask,
    )
    summary = ProcessedVideoSummary("S01_alert", "01", 0, str(npz_path), 40, 35, 0.875, 5)

    clips, rejected = build_clip_records([summary], _split_manifest(), _config())

    assert clips == []
    assert rejected[0].reason == "missing_run_above_threshold"


def _config() -> dict:
    return {
        "clips": {"sequence_length": 40, "stride_frames": 28, "overlap_ratio": 0.30},
        "missing_data": {"minimum_valid_frame_ratio": 0.80, "max_consecutive_missing_frames": 3},
    }


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        train_subjects=["01"],
        validation_subjects=[],
        test_subjects=[],
        train_video_ids=["S01_alert"],
        validation_video_ids=[],
        test_video_ids=[],
    )


def _artifact_path(name: str) -> Path:
    path = Path("artifacts") / "test_baseline_d" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
