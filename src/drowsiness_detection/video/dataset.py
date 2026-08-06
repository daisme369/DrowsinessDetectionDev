from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from drowsiness_detection.video.clips import ClipRecord


class TemporalClipDataset:
    """PyTorch-compatible dataset backed by cached Baseline D `.npz` files."""

    def __init__(
        self,
        clips: Iterable[ClipRecord],
        config: dict[str, Any],
        *,
        split: str | None = None,
        behavior_stats: dict[str, float] | None = None,
        training: bool = False,
    ) -> None:
        self.config = config
        self.split = split
        self.training = training
        self.clips = [clip for clip in clips if split is None or clip.split == split]
        self.behavior_stats = behavior_stats or {
            "ear_mean": 0.0,
            "ear_std": 1.0,
            "mar_mean": 0.0,
            "mar_std": 1.0,
        }
        image_config = config["image"]
        self.mean = np.asarray(image_config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.asarray(image_config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, 1, 3)
        self.rng = np.random.default_rng(int(config.get("training", {}).get("random_seed", 42)))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        clip = self.clips[index]
        data = np.load(Path(clip.npz_path))
        start = int(clip.start_frame)
        end = int(clip.end_frame) + 1

        frames = data["face_frames"][start:end].astype(np.uint8)
        ear = data["ear"][start:end].astype(np.float32)
        mar = data["mar"][start:end].astype(np.float32)
        valid_mask = data["valid_mask"][start:end].astype(bool)

        expected_length = int(self.config["clips"]["sequence_length"])
        if frames.shape[0] != expected_length:
            raise ValueError(f"Clip {clip.clip_id} has {frames.shape[0]} frames, expected {expected_length}")

        if self.training and bool(self.config.get("augmentation", {}).get("enabled", False)):
            frames = _augment_clip_temporally_consistent(frames, self.config, self.rng)

        frames_float = frames.astype(np.float32) / 255.0
        frames_float = (frames_float - self.mean) / self.std
        frames_chw = np.transpose(frames_float, (0, 3, 1, 2)).astype(np.float32)

        behavior = np.stack([ear, mar], axis=1).astype(np.float32)
        if self.config.get("behavior_features", {}).get("normalize", "zscore") == "zscore":
            behavior[:, 0] = (behavior[:, 0] - float(self.behavior_stats.get("ear_mean", 0.0))) / float(
                max(self.behavior_stats.get("ear_std", 1.0), 1e-6)
            )
            behavior[:, 1] = (behavior[:, 1] - float(self.behavior_stats.get("mar_mean", 0.0))) / float(
                max(self.behavior_stats.get("mar_std", 1.0), 1e-6)
            )
        behavior = np.nan_to_num(behavior, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return {
            "face_frames": torch.from_numpy(frames_chw),
            "behavior": torch.from_numpy(behavior),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32)),
            "label": torch.tensor(int(clip.label_id), dtype=torch.long),
            "clip_id": clip.clip_id,
            "video_id": clip.video_id,
            "subject_id": clip.subject_id,
            "start_time": float(clip.start_time),
            "end_time": float(clip.end_time),
        }


def count_clip_labels(clips: Iterable[ClipRecord], *, split: str | None = None) -> dict[int, int]:
    counts: dict[int, int] = {}
    for clip in clips:
        if split is not None and clip.split != split:
            continue
        counts[int(clip.label_id)] = counts.get(int(clip.label_id), 0) + 1
    return counts


def _augment_clip_temporally_consistent(
    frames_rgb: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    aug = config.get("augmentation", {})
    output = frames_rgb.copy()

    if rng.random() < float(aug.get("horizontal_flip_probability", 0.0)):
        output = output[:, :, ::-1, :]

    max_rotation = float(aug.get("rotation_degrees", 0.0))
    if max_rotation > 0:
        angle = float(rng.uniform(-max_rotation, max_rotation))
        output = _rotate_clip(output, angle)

    brightness = 1.0 + float(rng.uniform(-float(aug.get("brightness_jitter", 0.0)), float(aug.get("brightness_jitter", 0.0))))
    contrast = 1.0 + float(rng.uniform(-float(aug.get("contrast_jitter", 0.0)), float(aug.get("contrast_jitter", 0.0))))
    output_float = output.astype(np.float32)
    output_float = (output_float - 127.5) * contrast + 127.5
    output_float *= brightness

    saturation_jitter = float(aug.get("saturation_jitter", 0.0))
    if saturation_jitter > 0:
        saturation = 1.0 + float(rng.uniform(-saturation_jitter, saturation_jitter))
        gray = np.dot(output_float[..., :3], np.asarray([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        output_float = gray + (output_float - gray) * saturation

    output = np.clip(output_float, 0, 255).astype(np.uint8)
    if rng.random() < float(aug.get("gaussian_blur_probability", 0.0)):
        output = np.stack([cv2.GaussianBlur(frame, (3, 3), 0) for frame in output], axis=0)
    return output


def _rotate_clip(frames_rgb: np.ndarray, angle: float) -> np.ndarray:
    height, width = frames_rgb.shape[1:3]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = [
        cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        for frame in frames_rgb
    ]
    return np.stack(rotated, axis=0)
