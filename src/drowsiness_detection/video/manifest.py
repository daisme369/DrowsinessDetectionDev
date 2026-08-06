from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import cv2

from drowsiness_detection.video.labels import raw_label_to_canonical_id, raw_label_to_label_name


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    video_path: str
    subject_id: str
    label_name: str
    label_id: int
    raw_label: int
    fold_id: str
    duration_seconds: float
    original_fps: float
    frame_count: int
    width: int
    height: int


VIDEO_COLUMNS = [
    "video_id",
    "video_path",
    "subject_id",
    "label_name",
    "label_id",
    "raw_label",
    "fold_id",
    "duration_seconds",
    "original_fps",
    "frame_count",
    "width",
    "height",
]


def discover_uta_rldd_videos(config: dict[str, Any], *, probe_video: bool = True) -> list[VideoRecord]:
    dataset_config = config["dataset"]
    root = Path(dataset_config["root_dir"])
    extensions = {extension.lower() for extension in dataset_config.get("video_extensions", [])}
    if not root.exists():
        raise FileNotFoundError(f"UTA-RLDD video root does not exist: {root}")

    records: list[VideoRecord] = []
    label_counts_by_subject: dict[tuple[str, int], int] = {}
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        subject_id = subject_dir.name
        for video_path in sorted(path for path in subject_dir.iterdir() if path.is_file()):
            if extensions and video_path.suffix.lower() not in extensions:
                continue
            raw_label = _raw_label_from_filename(video_path)
            if raw_label is None:
                continue
            label_name = raw_label_to_label_name(raw_label)
            label_id = raw_label_to_canonical_id(raw_label)
            key = (subject_id, raw_label)
            label_counts_by_subject[key] = label_counts_by_subject.get(key, 0) + 1
            duplicate_index = label_counts_by_subject[key]
            duplicate_suffix = "" if duplicate_index == 1 else f"_{duplicate_index}"
            video_id = f"S{subject_id}_{label_name}{duplicate_suffix}"
            duration, fps, frame_count, width, height = _probe_video(video_path) if probe_video else (0.0, 0.0, 0, 0, 0)
            records.append(
                VideoRecord(
                    video_id=video_id,
                    video_path=str(video_path),
                    subject_id=subject_id,
                    label_name=label_name,
                    label_id=label_id,
                    raw_label=raw_label,
                    fold_id="",
                    duration_seconds=duration,
                    original_fps=fps,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                )
            )
    if not records:
        raise RuntimeError(f"No UTA-RLDD videos discovered under {root}")
    return records


def read_video_manifest(path: str | Path) -> list[VideoRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Video manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: list[VideoRecord] = []
    for row in rows:
        records.append(
            VideoRecord(
                video_id=row["video_id"],
                video_path=row["video_path"],
                subject_id=row["subject_id"],
                label_name=row["label_name"],
                label_id=int(row["label_id"]),
                raw_label=int(row["raw_label"]),
                fold_id=row.get("fold_id", ""),
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                original_fps=float(row.get("original_fps") or 0.0),
                frame_count=int(float(row.get("frame_count") or 0)),
                width=int(float(row.get("width") or 0)),
                height=int(float(row.get("height") or 0)),
            )
        )
    return records


def write_video_manifest(records: Iterable[VideoRecord], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _raw_label_from_filename(video_path: Path) -> int | None:
    match = re.match(r"^(0|5|10)(?:\D.*)?$", video_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _probe_video(video_path: Path) -> tuple[float, float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return 0.0, 0.0, 0, 0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 1e-6 and frame_count > 0 else 0.0
    capture.release()
    return duration, fps, frame_count, width, height
