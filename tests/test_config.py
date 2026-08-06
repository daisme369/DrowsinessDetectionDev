from __future__ import annotations

from drowsiness_detection.utils import load_baseline_a_config, load_baseline_b_config


def test_load_baseline_a_config() -> None:
    config = load_baseline_a_config("configs/baseline_a.yaml")

    assert config.landmarks.detector == "mediapipe"
    assert config.landmarks.model_asset_path.endswith("face_landmarker.task")
    assert config.thresholds.eye_closed_duration_seconds > 0


def test_load_baseline_b_config() -> None:
    config = load_baseline_b_config("configs/baseline_b.yaml")

    assert config.dataset.root == "data"
    assert config.dataset.crop_mode == "face"
    assert config.model.architecture == "random_forest"
    assert "mean_ear" in config.model.feature_names
    assert "mar" in config.model.feature_names
    assert config.model.num_classes == 2
    assert config.inference.closed_class_index == 1
    assert config.inference.model_path.endswith(".pkl")
