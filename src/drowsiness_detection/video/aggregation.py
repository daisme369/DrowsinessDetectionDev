from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClipPrediction:
    clip_id: str
    video_id: str
    subject_id: str
    true_label: int
    p_alert: float
    p_low_vigilant: float
    p_drowsy: float
    predicted_label: int
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class VideoPrediction:
    video_id: str
    subject_id: str
    true_label: int
    p_alert: float
    p_low_vigilant: float
    p_drowsy: float
    predicted_label: int
    num_clips: int


def clip_predictions_from_probabilities(
    batches: Iterable[dict[str, object]],
    probabilities: np.ndarray,
) -> list[ClipPrediction]:
    rows = list(batches)
    if len(rows) != len(probabilities):
        raise ValueError("Metadata rows and probability rows must have the same length")
    predictions: list[ClipPrediction] = []
    for row, probs in zip(rows, probabilities):
        normalized = _normalize_probability(probs)
        predictions.append(
            ClipPrediction(
                clip_id=str(row["clip_id"]),
                video_id=str(row["video_id"]),
                subject_id=str(row["subject_id"]),
                true_label=int(row["true_label"]),
                p_alert=float(normalized[0]),
                p_low_vigilant=float(normalized[1]),
                p_drowsy=float(normalized[2]),
                predicted_label=int(np.argmax(normalized)),
                start_time=float(row.get("start_time", 0.0)),
                end_time=float(row.get("end_time", 0.0)),
            )
        )
    return predictions


def aggregate_video_predictions(clip_predictions: Sequence[ClipPrediction]) -> list[VideoPrediction]:
    grouped: dict[str, list[ClipPrediction]] = {}
    for prediction in clip_predictions:
        grouped.setdefault(prediction.video_id, []).append(prediction)
    if not grouped:
        raise ValueError("Cannot aggregate video predictions from an empty clip prediction set")

    videos: list[VideoPrediction] = []
    for video_id, items in sorted(grouped.items()):
        true_labels = {item.true_label for item in items}
        subject_ids = {item.subject_id for item in items}
        if len(true_labels) != 1:
            raise ValueError(f"Video {video_id} has mixed true labels: {sorted(true_labels)}")
        if len(subject_ids) != 1:
            raise ValueError(f"Video {video_id} has mixed subject IDs: {sorted(subject_ids)}")
        probabilities = np.asarray(
            [[item.p_alert, item.p_low_vigilant, item.p_drowsy] for item in items],
            dtype=np.float64,
        )
        mean_probability = _normalize_probability(probabilities.mean(axis=0))
        videos.append(
            VideoPrediction(
                video_id=video_id,
                subject_id=items[0].subject_id,
                true_label=items[0].true_label,
                p_alert=float(mean_probability[0]),
                p_low_vigilant=float(mean_probability[1]),
                p_drowsy=float(mean_probability[2]),
                predicted_label=int(np.argmax(mean_probability)),
                num_clips=len(items),
            )
        )
    return videos


def write_clip_predictions(predictions: Sequence[ClipPrediction], path: str | Path) -> None:
    _write_dataclass_rows(predictions, path, ClipPrediction)


def write_video_predictions(predictions: Sequence[VideoPrediction], path: str | Path) -> None:
    _write_dataclass_rows(predictions, path, VideoPrediction)


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid probability vector: {values}")
    return array / total


def _write_dataclass_rows(rows: Sequence[object], path: str | Path, row_type: type) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
