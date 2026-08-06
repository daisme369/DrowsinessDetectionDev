from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import cv2
import numpy as np

from drowsiness_detection.preprocessing.roi_extraction import (
    NormalizedYoloBox,
    extract_roi_from_yolo_box,
    parse_yolo_line,
)
from drowsiness_detection.utils.config import BaselineBDatasetConfig, BaselineBModelConfig


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True, slots=True)
class RoiSample:
    image_path: Path
    label_path: Path
    yolo_box: NormalizedYoloBox
    label: int
    source_class_id: int


def normalize_class_mapping(mapping: dict[int | str, int]) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for key, value in mapping.items():
        normalized[int(key)] = int(value)
    return normalized


def collect_yolo_roi_samples(config: BaselineBDatasetConfig, split: str) -> list[RoiSample]:
    root = Path(config.root)
    image_root = root / split / config.image_dir
    label_root = root / split / config.label_dir
    if not image_root.exists():
        raise FileNotFoundError(f"Image split directory does not exist: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label split directory does not exist: {label_root}")

    class_mapping = normalize_class_mapping(config.class_id_to_label)
    samples: list[RoiSample] = []
    for label_path in sorted(label_root.glob("*.txt")):
        image_path = _find_image_for_label(image_root, label_path.stem)
        if image_path is None:
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            yolo_box = parse_yolo_line(line)
            if yolo_box.class_id not in class_mapping:
                continue
            samples.append(
                RoiSample(
                    image_path=image_path,
                    label_path=label_path,
                    yolo_box=yolo_box,
                    label=class_mapping[yolo_box.class_id],
                    source_class_id=yolo_box.class_id,
                )
            )
    if not samples:
        raise RuntimeError(f"No ROI samples found for split '{split}' under {root}")
    return samples


def count_labels(samples: Sequence[RoiSample]) -> Counter[int]:
    return Counter(sample.label for sample in samples)


class YoloRoiDataset:
    """Torch-compatible dataset for YOLO face boxes converted to classifier ROIs."""

    def __init__(
        self,
        samples: Sequence[RoiSample],
        dataset_config: BaselineBDatasetConfig,
        model_config: BaselineBModelConfig,
        *,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample.image_path}")
        roi, _roi_box = extract_roi_from_yolo_box(
            image,
            sample.yolo_box,
            crop_mode=self.dataset_config.crop_mode,
            padding=self.dataset_config.bbox_padding,
            eye_band_y_min=self.dataset_config.eye_band_y_min,
            eye_band_y_max=self.dataset_config.eye_band_y_max,
            min_box_size=self.dataset_config.min_box_size,
        )
        if roi is None:
            raise RuntimeError(f"Invalid ROI for {sample.image_path}")
        if self.augment:
            roi = _augment_roi(roi)
        tensor = preprocess_roi_for_torch(
            roi,
            input_size=self.model_config.input_size,
            grayscale=self.model_config.grayscale,
            normalize_mean=self.model_config.normalize_mean,
            normalize_std=self.model_config.normalize_std,
        )
        return tensor, sample.label


def preprocess_roi_for_torch(
    roi_bgr: np.ndarray,
    *,
    input_size: Sequence[int],
    grayscale: bool,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
) -> object:
    import torch

    return torch.from_numpy(
        preprocess_roi_to_numpy(
            roi_bgr,
            input_size=input_size,
            grayscale=grayscale,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    )


def preprocess_roi_to_numpy(
    roi_bgr: np.ndarray,
    *,
    input_size: Sequence[int],
    grayscale: bool,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
) -> np.ndarray:
    height, width = _parse_input_size(input_size)
    if grayscale:
        resized = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (width, height))
        array = resized.astype(np.float32)[None, :, :] / 255.0
        mean = np.array([normalize_mean[0] if normalize_mean else 0.5], dtype=np.float32)[:, None, None]
        std = np.array([normalize_std[0] if normalize_std else 0.5], dtype=np.float32)[:, None, None]
    else:
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array(_three_values(normalize_mean, 0.0), dtype=np.float32)[:, None, None]
        std = np.array(_three_values(normalize_std, 1.0), dtype=np.float32)[:, None, None]
    array = (array - mean) / np.maximum(std, 1e-6)
    return array.astype(np.float32, copy=False)


def _find_image_for_label(image_root: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _augment_roi(roi: np.ndarray) -> np.ndarray:
    output = roi.copy()
    if random.random() < 0.5:
        output = cv2.flip(output, 1)
    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-18.0, 18.0)
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    if random.random() < 0.2:
        output = cv2.GaussianBlur(output, (3, 3), sigmaX=0.0)
    return output


def _parse_input_size(input_size: Sequence[int]) -> tuple[int, int]:
    if len(input_size) != 2:
        raise ValueError(f"input_size must contain [height, width], got: {input_size}")
    height = int(input_size[0])
    width = int(input_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"input_size values must be positive, got: {input_size}")
    return height, width


def _three_values(values: Sequence[float], fallback: float) -> list[float]:
    if len(values) >= 3:
        return [float(values[0]), float(values[1]), float(values[2])]
    if len(values) == 1:
        return [float(values[0])] * 3
    return [fallback, fallback, fallback]
