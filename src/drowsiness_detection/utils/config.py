from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(slots=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30


@dataclass(slots=True)
class LandmarkConfig:
    detector: str = "mediapipe"
    model_asset_path: str = "artifacts/models/face_landmarker.task"
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False


@dataclass(slots=True)
class ThresholdConfig:
    ear_closed: float = 0.21
    ear_open: float = 0.25
    mar_yawn: float = 0.65
    mar_recovery: float = 0.55
    head_pitch_down_degrees: float = 18.0
    head_pitch_up_degrees: float = -20.0
    eye_closed_duration_seconds: float = 1.5
    yawn_duration_seconds: float = 1.0
    head_pose_duration_seconds: float = 1.0
    perclos_window_seconds: float = 30.0
    perclos_threshold: float = 0.40
    perclos_min_samples: int = 10
    recovery_hold_seconds: float = 2.0
    no_face_grace_seconds: float = 1.0
    ema_alpha: float = 0.35


@dataclass(slots=True)
class DisplayConfig:
    show_landmarks: bool = True
    show_metrics: bool = True
    window_name: str = "Baseline A Drowsiness Detection"


@dataclass(slots=True)
class BaselineAConfig:
    camera: CameraConfig
    landmarks: LandmarkConfig
    thresholds: ThresholdConfig
    display: DisplayConfig


@dataclass(slots=True)
class BaselineBDatasetConfig:
    root: str = "data"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"
    image_dir: str = "images"
    label_dir: str = "labels"
    crop_mode: str = "face"
    bbox_padding: float = 0.08
    eye_band_y_min: float = 0.18
    eye_band_y_max: float = 0.55
    min_box_size: int = 12
    class_names: list[str] = field(default_factory=lambda: ["open_or_alert", "closed_or_drowsy"])
    class_id_to_label: dict[int | str, int] = field(default_factory=lambda: {0: 0, 1: 1})


@dataclass(slots=True)
class BaselineBModelConfig:
    architecture: str = "random_forest"
    feature_names: list[str] = field(
        default_factory=lambda: [
            "left_ear",
            "right_ear",
            "mean_ear",
            "ear_diff",
            "mar",
            "mar_to_ear",
            "pitch_degrees",
            "yaw_degrees",
            "roll_degrees",
        ]
    )
    random_forest_n_estimators: int = 200
    random_forest_max_depth: int | None = 8
    logistic_max_iter: int = 1000
    svm_c: float = 1.0
    input_size: list[int] = field(default_factory=lambda: [96, 96])
    num_classes: int = 2
    grayscale: bool = False
    pretrained: bool = False
    dropout: float = 0.2
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass(slots=True)
class BaselineBTrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 0
    seed: int = 42
    use_class_weights: bool = True
    feature_path: str = "artifacts/features/baseline_b_geometric_features.csv"
    output_dir: str = "artifacts/models/baseline_b"
    report_dir: str = "artifacts/reports/baseline_b"


@dataclass(slots=True)
class BaselineBInferenceConfig:
    backend: str = "sklearn"
    model_path: str = "artifacts/models/baseline_b/geometric_feature_classifier.pkl"
    checkpoint_path: str = "artifacts/models/baseline_b/best.pt"
    onnx_path: str = "artifacts/models/baseline_b/eye_classifier.onnx"
    closed_class_index: int = 1
    confidence_threshold: float = 0.6


@dataclass(slots=True)
class BaselineBTemporalConfig:
    closed_probability_threshold: float = 0.6
    open_probability_threshold: float = 0.4
    eye_closed_duration_seconds: float = 1.5
    perclos_window_seconds: float = 30.0
    perclos_threshold: float = 0.4
    perclos_min_samples: int = 10
    recovery_hold_seconds: float = 2.0
    no_face_grace_seconds: float = 1.0


@dataclass(slots=True)
class BaselineBConfig:
    camera: CameraConfig
    landmarks: LandmarkConfig
    dataset: BaselineBDatasetConfig
    model: BaselineBModelConfig
    training: BaselineBTrainingConfig
    inference: BaselineBInferenceConfig
    temporal: BaselineBTemporalConfig
    display: DisplayConfig



T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any] | None) -> T:
    valid_names = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in (values or {}).items() if key in valid_names}
    return cls(**kwargs)


def load_baseline_a_config(path: str | Path) -> BaselineAConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return BaselineAConfig(
        camera=_coerce_dataclass(CameraConfig, raw.get("camera")),
        landmarks=_coerce_dataclass(LandmarkConfig, raw.get("landmarks")),
        thresholds=_coerce_dataclass(ThresholdConfig, raw.get("thresholds")),
        display=_coerce_dataclass(DisplayConfig, raw.get("display")),
    )


def load_baseline_b_config(path: str | Path) -> BaselineBConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return BaselineBConfig(
        camera=_coerce_dataclass(CameraConfig, raw.get("camera")),
        landmarks=_coerce_dataclass(LandmarkConfig, raw.get("landmarks")),
        dataset=_coerce_dataclass(BaselineBDatasetConfig, raw.get("dataset")),
        model=_coerce_dataclass(BaselineBModelConfig, raw.get("model")),
        training=_coerce_dataclass(BaselineBTrainingConfig, raw.get("training")),
        inference=_coerce_dataclass(BaselineBInferenceConfig, raw.get("inference")),
        temporal=_coerce_dataclass(BaselineBTemporalConfig, raw.get("temporal")),
        display=_coerce_dataclass(DisplayConfig, raw.get("display")),
    )