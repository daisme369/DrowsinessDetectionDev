from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2

from drowsiness_detection.video.labels import RAW_LABEL_TO_NAME
from drowsiness_detection.video.manifest import VideoRecord


@dataclass(frozen=True, slots=True)
class InvalidVideoRecord:
    video_id: str
    video_path: str
    subject_id: str
    reason: str


def validate_video_records(
    records: Iterable[VideoRecord],
    config: dict[str, Any],
    *,
    decode_one_frame: bool = True,
) -> tuple[dict[str, Any], list[InvalidVideoRecord]]:
    records = list(records)
    invalid: list[InvalidVideoRecord] = []
    video_ids = [record.video_id for record in records]
    duplicate_ids = sorted(video_id for video_id, count in Counter(video_ids).items() if count > 1)
    for duplicate_id in duplicate_ids:
        matching = [record for record in records if record.video_id == duplicate_id]
        for record in matching:
            invalid.append(_invalid(record, "duplicate_video_id"))

    expected_raw_labels = set(int(value) for value in config["dataset"].get("expected_raw_labels", [0, 5, 10]))
    labels_by_subject: dict[str, list[int]] = defaultdict(list)
    for record in records:
        labels_by_subject[record.subject_id].append(record.raw_label)
        _validate_record(record, expected_raw_labels, invalid, decode_one_frame=decode_one_frame)

    missing_expected: dict[str, list[int]] = {}
    duplicate_labels: dict[str, dict[str, int]] = {}
    for subject_id, raw_labels in labels_by_subject.items():
        counts = Counter(raw_labels)
        missing = sorted(expected_raw_labels.difference(counts))
        if missing:
            missing_expected[subject_id] = missing
        duplicated = {str(label): count for label, count in counts.items() if count > 1}
        if duplicated:
            duplicate_labels[subject_id] = duplicated

    label_counts = Counter(record.label_name for record in records)
    subject_count = len(labels_by_subject)
    report = {
        "video_count": len(records),
        "subject_count": subject_count,
        "label_counts": dict(label_counts),
        "duplicate_video_ids": duplicate_ids,
        "subjects_missing_expected_labels": missing_expected,
        "subjects_with_duplicate_labels": duplicate_labels,
        "invalid_video_count": len(invalid),
        "valid_video_count": max(0, len(records) - len({item.video_id for item in invalid})),
    }
    return report, invalid


def write_validation_outputs(
    report: dict[str, Any],
    invalid: list[InvalidVideoRecord],
    *,
    report_path: str | Path,
    invalid_path: str | Path,
) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    invalid_csv = Path(invalid_path)
    invalid_csv.parent.mkdir(parents=True, exist_ok=True)
    with invalid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "video_path", "subject_id", "reason"])
        writer.writeheader()
        for item in invalid:
            writer.writerow(asdict(item))


def _validate_record(
    record: VideoRecord,
    expected_raw_labels: set[int],
    invalid: list[InvalidVideoRecord],
    *,
    decode_one_frame: bool,
) -> None:
    if not record.subject_id:
        invalid.append(_invalid(record, "empty_subject_id"))
    if record.raw_label not in expected_raw_labels or record.raw_label not in RAW_LABEL_TO_NAME:
        invalid.append(_invalid(record, f"unrecognized_raw_label_{record.raw_label}"))
    if not Path(record.video_path).exists():
        invalid.append(_invalid(record, "missing_video_file"))
        return
    if record.original_fps <= 0 or record.original_fps > 240:
        invalid.append(_invalid(record, f"implausible_fps_{record.original_fps:.3f}"))
    if record.duration_seconds <= 0:
        invalid.append(_invalid(record, "non_positive_duration"))
    if record.frame_count <= 0:
        invalid.append(_invalid(record, "non_positive_frame_count"))
    if record.width <= 0 or record.height <= 0:
        invalid.append(_invalid(record, "invalid_frame_size"))
    if decode_one_frame and not _can_decode_one_frame(record.video_path):
        invalid.append(_invalid(record, "cannot_decode_first_frame"))


def _can_decode_one_frame(video_path: str) -> bool:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        return False
    ok, frame = capture.read()
    capture.release()
    return bool(ok and frame is not None)


def _invalid(record: VideoRecord, reason: str) -> InvalidVideoRecord:
    return InvalidVideoRecord(
        video_id=record.video_id,
        video_path=record.video_path,
        subject_id=record.subject_id,
        reason=reason,
    )
