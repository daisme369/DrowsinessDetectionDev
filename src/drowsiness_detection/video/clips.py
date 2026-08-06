from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drowsiness_detection.video.preprocess import ProcessedVideoSummary, maximum_missing_run
from drowsiness_detection.video.splits import SplitManifest, split_for_video_id


@dataclass(frozen=True, slots=True)
class ClipRecord:
    clip_id: str
    video_id: str
    subject_id: str
    label_id: int
    split: str
    npz_path: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    valid_frame_ratio: float


@dataclass(frozen=True, slots=True)
class RejectedClipRecord:
    clip_id: str
    video_id: str
    subject_id: str
    reason: str
    valid_frame_ratio: float
    maximum_missing_run: int


def build_clip_records(
    summaries: list[ProcessedVideoSummary],
    split_manifest: SplitManifest,
    config: dict[str, Any],
) -> tuple[list[ClipRecord], list[RejectedClipRecord]]:
    clip_config = config["clips"]
    missing_config = config["missing_data"]
    sequence_length = int(clip_config["sequence_length"])
    stride = int(clip_config.get("stride_frames") or round(sequence_length * (1.0 - float(clip_config["overlap_ratio"]))))
    minimum_valid_ratio = float(missing_config["minimum_valid_frame_ratio"])
    max_missing_allowed = int(missing_config["max_consecutive_missing_frames"])

    clips: list[ClipRecord] = []
    rejected: list[RejectedClipRecord] = []
    for summary in summaries:
        data = np.load(summary.npz_path)
        timestamps = data["timestamps"]
        valid_mask = data["valid_mask"].astype(bool)
        frame_count = len(timestamps)
        for start in range(0, max(0, frame_count - sequence_length + 1), stride):
            end = start + sequence_length
            clip_valid = valid_mask[start:end]
            valid_ratio = float(clip_valid.mean()) if clip_valid.size else 0.0
            max_missing_run = maximum_missing_run(clip_valid)
            clip_id = f"{summary.video_id}_{start:06d}"
            if valid_ratio < minimum_valid_ratio:
                rejected.append(
                    RejectedClipRecord(
                        clip_id=clip_id,
                        video_id=summary.video_id,
                        subject_id=summary.subject_id,
                        reason="valid_frame_ratio_below_threshold",
                        valid_frame_ratio=valid_ratio,
                        maximum_missing_run=max_missing_run,
                    )
                )
                continue
            if max_missing_run > max_missing_allowed:
                rejected.append(
                    RejectedClipRecord(
                        clip_id=clip_id,
                        video_id=summary.video_id,
                        subject_id=summary.subject_id,
                        reason="missing_run_above_threshold",
                        valid_frame_ratio=valid_ratio,
                        maximum_missing_run=max_missing_run,
                    )
                )
                continue
            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    video_id=summary.video_id,
                    subject_id=summary.subject_id,
                    label_id=summary.label_id,
                    split=split_for_video_id(split_manifest, summary.video_id),
                    npz_path=summary.npz_path,
                    start_time=float(timestamps[start]),
                    end_time=float(timestamps[end - 1]),
                    start_frame=start,
                    end_frame=end - 1,
                    valid_frame_ratio=valid_ratio,
                )
            )
    return clips, rejected


def write_clip_manifest(clips: list[ClipRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(clips[0]).keys()) if clips else [
        "clip_id",
        "video_id",
        "subject_id",
        "label_id",
        "split",
        "npz_path",
        "start_time",
        "end_time",
        "start_frame",
        "end_frame",
        "valid_frame_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for clip in clips:
            writer.writerow(asdict(clip))


def read_clip_manifest(path: str | Path) -> list[ClipRecord]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        ClipRecord(
            clip_id=row["clip_id"],
            video_id=row["video_id"],
            subject_id=row["subject_id"],
            label_id=int(row["label_id"]),
            split=row["split"],
            npz_path=row["npz_path"],
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            valid_frame_ratio=float(row["valid_frame_ratio"]),
        )
        for row in rows
    ]


def write_rejected_clips(rejected: list[RejectedClipRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "video_id",
        "subject_id",
        "reason",
        "valid_frame_ratio",
        "maximum_missing_run",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rejected:
            writer.writerow(asdict(item))
