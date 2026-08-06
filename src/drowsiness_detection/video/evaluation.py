from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np

from drowsiness_detection.video.aggregation import (
    ClipPrediction,
    VideoPrediction,
    aggregate_video_predictions,
    clip_predictions_from_probabilities,
)
from drowsiness_detection.video.clips import ClipRecord, read_clip_manifest
from drowsiness_detection.video.dataset import TemporalClipDataset
from drowsiness_detection.video.metrics import MulticlassMetrics, compute_multiclass_metrics


def load_behavior_stats(path: str | Path) -> dict[str, float]:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Behavior normalization stats do not exist: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "ear_mean": float(payload.get("ear_mean", 0.0)),
        "ear_std": float(max(float(payload.get("ear_std", 1.0)), 1e-6)),
        "mar_mean": float(payload.get("mar_mean", 0.0)),
        "mar_std": float(max(float(payload.get("mar_std", 1.0)), 1e-6)),
    }


def default_clips_path(config: dict[str, Any]) -> Path:
    return Path(config["dataset"]["processed_dir"]) / "manifests" / "clips.csv"


def default_behavior_stats_path(config: dict[str, Any], *, fold_index: int | None = None) -> Path:
    preprocessing_dir = Path(config["project"]["output_dir"]) / "preprocessing"
    split_mode = str(config.get("split", {}).get("mode", "fixed"))
    if split_mode in {"five_fold", "cross_validation"}:
        if fold_index is None:
            fold_index = 1
        return preprocessing_dir / f"fold_{fold_index}_behavior_stats.json"
    return preprocessing_dir / "fixed_behavior_stats.json"


def load_clip_dataset(
    config: dict[str, Any],
    *,
    split: str,
    clips_path: str | Path | None = None,
    behavior_stats_path: str | Path | None = None,
    behavior_stats: dict[str, float] | None = None,
    training: bool = False,
) -> TemporalClipDataset:
    path = Path(clips_path) if clips_path is not None else default_clips_path(config)
    clips = read_clip_manifest(path)
    stats = behavior_stats if behavior_stats is not None else load_behavior_stats(
        behavior_stats_path or default_behavior_stats_path(config)
    )
    dataset = TemporalClipDataset(clips, config, split=split, behavior_stats=stats, training=training)
    if len(dataset) == 0:
        raise ValueError(f"No clips found for split `{split}` in {path}")
    return dataset


def make_dataloader(dataset: TemporalClipDataset, config: dict[str, Any], *, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(config.get("training", {}).get("random_seed", 42)))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(torch.cuda.is_available()),
        generator=generator,
    )


def evaluate_model_on_loader(model, dataloader, device, criterion=None) -> dict[str, Any]:
    import torch

    model.eval()
    losses: list[float] = []
    metadata: list[dict[str, object]] = []
    probabilities: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in dataloader:
            frames = batch["face_frames"].to(device, non_blocking=True)
            behavior = batch["behavior"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(frames, behavior, valid_mask)
            if criterion is not None:
                loss = criterion(logits, labels)
                losses.append(float(loss.detach().cpu()))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities.append(probs)
            for index in range(labels.shape[0]):
                metadata.append(
                    {
                        "clip_id": batch["clip_id"][index],
                        "video_id": batch["video_id"][index],
                        "subject_id": batch["subject_id"][index],
                        "true_label": int(labels[index].detach().cpu()),
                        "start_time": float(batch["start_time"][index]),
                        "end_time": float(batch["end_time"][index]),
                    }
                )

    if not probabilities:
        raise ValueError("Cannot evaluate an empty dataloader")
    clip_predictions = clip_predictions_from_probabilities(metadata, np.concatenate(probabilities, axis=0))
    video_predictions = aggregate_video_predictions(clip_predictions)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "clip_predictions": clip_predictions,
        "video_predictions": video_predictions,
        "clip_metrics": metrics_from_clip_predictions(clip_predictions),
        "video_metrics": metrics_from_video_predictions(video_predictions),
    }


def metrics_from_clip_predictions(predictions: list[ClipPrediction]) -> MulticlassMetrics:
    return compute_multiclass_metrics(
        [prediction.true_label for prediction in predictions],
        [prediction.predicted_label for prediction in predictions],
    )


def metrics_from_video_predictions(predictions: list[VideoPrediction]) -> MulticlassMetrics:
    return compute_multiclass_metrics(
        [prediction.true_label for prediction in predictions],
        [prediction.predicted_label for prediction in predictions],
    )


def class_weights_from_clips(clips: list[ClipRecord], config: dict[str, Any], *, split: str = "train"):
    import torch

    counts = {label: 0 for label in [0, 1, 2]}
    for clip in clips:
        if clip.split == split:
            counts[int(clip.label_id)] += 1
    nonzero = [count for count in counts.values() if count > 0]
    if len(nonzero) != len(counts):
        return None
    imbalance = max(nonzero) / max(min(nonzero), 1)
    threshold = float(config.get("training", {}).get("class_weight_imbalance_threshold", 1.2))
    if imbalance < threshold:
        return None
    total = float(sum(nonzero))
    weights = [total / (len(counts) * float(counts[label])) for label in [0, 1, 2]]
    return torch.tensor(weights, dtype=torch.float32)
