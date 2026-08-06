from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drowsiness_detection.video.config import config_hash
from drowsiness_detection.video.face_processor import MediaPipeFaceProcessor
from drowsiness_detection.video.manifest import VideoRecord
from drowsiness_detection.video.sampling import target_timestamps


@dataclass(frozen=True, slots=True)
class ProcessedVideoSummary:
    video_id: str
    subject_id: str
    label_id: int
    npz_path: str
    sampled_frame_count: int
    valid_frame_count: int
    valid_frame_ratio: float
    maximum_missing_run: int


FRAME_COLUMNS = [
    "video_id",
    "subject_id",
    "label_id",
    "frame_index",
    "source_frame_index",
    "timestamp_seconds",
    "face_detected",
    "landmarks_detected",
    "valid_frame",
    "failure_reason",
]


def preprocess_videos(
    records: list[VideoRecord],
    config: dict[str, Any],
    *,
    max_videos: int = 0,
    max_frames_per_video: int = 0,
) -> tuple[list[ProcessedVideoSummary], list[dict[str, object]]]:
    selected_records = records[:max_videos] if max_videos > 0 else records
    processor = MediaPipeFaceProcessor(config)
    summaries: list[ProcessedVideoSummary] = []
    frame_rows: list[dict[str, object]] = []
    try:
        for record in selected_records:
            summary, rows = preprocess_one_video(
                record,
                config,
                processor,
                max_frames=max_frames_per_video,
            )
            summaries.append(summary)
            frame_rows.extend(rows)
            print(
                f"preprocessed {record.video_id}: frames={summary.sampled_frame_count} "
                f"valid={summary.valid_frame_count} ratio={summary.valid_frame_ratio:.3f}",
                flush=True,
            )
    finally:
        processor.close()
    return summaries, frame_rows


def preprocess_one_video(
    record: VideoRecord,
    config: dict[str, Any],
    processor: MediaPipeFaceProcessor,
    *,
    max_frames: int = 0,
) -> tuple[ProcessedVideoSummary, list[dict[str, object]]]:
    processed_dir = Path(config["dataset"]["processed_dir"])
    feature_dir = processed_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    timestamps = target_timestamps(record.duration_seconds, float(config["sampling"]["target_fps"]))
    if max_frames > 0:
        timestamps = timestamps[:max_frames]

    capture = cv2.VideoCapture(record.video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for preprocessing: {record.video_path}")

    image_height = int(config["image"]["height"])
    image_width = int(config["image"]["width"])
    face_frames: list[np.ndarray] = []
    ears: list[float] = []
    mars: list[float] = []
    raw_valid_mask: list[bool] = []
    source_frame_indices: list[int] = []
    failure_reasons: list[str] = []
    frame_rows: list[dict[str, object]] = []

    try:
        for frame_index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            source_frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
            source_frame_indices.append(max(0, source_frame_index))
            if not ok or frame is None:
                result = None
                face_frames.append(np.zeros((image_height, image_width, 3), dtype=np.uint8))
                ears.append(np.nan)
                mars.append(np.nan)
                raw_valid_mask.append(False)
                reason = "decode_failed"
            else:
                result = processor.process(frame)
                if result.is_valid and result.aligned_face is not None:
                    face_frames.append(result.aligned_face.astype(np.uint8))
                    ears.append(float(result.ear))
                    mars.append(float(result.mar))
                    raw_valid_mask.append(True)
                    reason = result.failure_reason or ""
                else:
                    face_frames.append(np.zeros((image_height, image_width, 3), dtype=np.uint8))
                    ears.append(np.nan if result.ear is None else float(result.ear))
                    mars.append(np.nan if result.mar is None else float(result.mar))
                    raw_valid_mask.append(False)
                    reason = result.failure_reason or "invalid_frame"
            failure_reasons.append(reason)
            frame_rows.append(
                {
                    "video_id": record.video_id,
                    "subject_id": record.subject_id,
                    "label_id": record.label_id,
                    "frame_index": frame_index,
                    "source_frame_index": source_frame_indices[-1],
                    "timestamp_seconds": f"{float(timestamp):.6f}",
                    "face_detected": result.face_bbox is not None if result is not None else False,
                    "landmarks_detected": result.landmarks is not None if result is not None else False,
                    "valid_frame": raw_valid_mask[-1],
                    "failure_reason": reason,
                }
            )
    finally:
        capture.release()

    ear_array = np.asarray(ears, dtype=np.float32)
    mar_array = np.asarray(mars, dtype=np.float32)
    raw_valid = np.asarray(raw_valid_mask, dtype=bool)
    recovered_ear, recovered_mar, recovered_valid = _recover_behavior_features(
        ear_array,
        mar_array,
        raw_valid,
        max_gap=int(config["missing_data"]["max_consecutive_missing_frames"]),
        interpolate=bool(config["missing_data"]["interpolate_behavior_features"]),
    )
    valid_ratio = float(recovered_valid.mean()) if recovered_valid.size else 0.0
    max_missing = maximum_missing_run(recovered_valid)
    npz_path = feature_dir / f"{record.video_id}.npz"
    np.savez_compressed(
        npz_path,
        face_frames=np.asarray(face_frames, dtype=np.uint8),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        source_frame_indices=np.asarray(source_frame_indices, dtype=np.int64),
        ear=recovered_ear.astype(np.float32),
        mar=recovered_mar.astype(np.float32),
        raw_ear=ear_array,
        raw_mar=mar_array,
        valid_mask=recovered_valid.astype(bool),
        raw_valid_mask=raw_valid,
        preprocessing_hash=config_hash(config),
    )
    summary = ProcessedVideoSummary(
        video_id=record.video_id,
        subject_id=record.subject_id,
        label_id=record.label_id,
        npz_path=str(npz_path),
        sampled_frame_count=len(face_frames),
        valid_frame_count=int(recovered_valid.sum()),
        valid_frame_ratio=valid_ratio,
        maximum_missing_run=max_missing,
    )
    return summary, frame_rows


def write_frame_manifest(rows: list[dict[str, object]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_processed_video_summary(summaries: list[ProcessedVideoSummary], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            for summary in summaries:
                writer.writerow(asdict(summary))


def maximum_missing_run(valid_mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in valid_mask:
        if bool(value):
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _recover_behavior_features(
    ear: np.ndarray,
    mar: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap: int,
    interpolate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not interpolate or valid.size == 0:
        return ear, mar, valid
    recovered_valid = valid.copy()
    recovered_ear = _fill_short_gaps(ear, valid, max_gap=max_gap)
    recovered_mar = _fill_short_gaps(mar, valid, max_gap=max_gap)
    recovered_valid &= np.isfinite(recovered_ear) & np.isfinite(recovered_mar)
    return recovered_ear, recovered_mar, recovered_valid


def _fill_short_gaps(values: np.ndarray, valid: np.ndarray, *, max_gap: int) -> np.ndarray:
    output = values.copy()
    indices = np.arange(len(values))
    valid_indices = indices[valid & np.isfinite(values)]
    if len(valid_indices) == 0:
        return output
    interp_values = np.interp(indices, valid_indices, values[valid_indices]).astype(np.float32)
    missing_runs = _missing_runs(valid & np.isfinite(values))
    for start, end in missing_runs:
        if end - start <= max_gap:
            output[start:end] = interp_values[start:end]
    return output


def _missing_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid and start is None:
            start = index
        elif is_valid and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs
