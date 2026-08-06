from __future__ import annotations

from drowsiness_detection.training import collect_yolo_roi_samples, count_labels
from drowsiness_detection.utils import load_baseline_b_config


def test_collect_yawdd_train_samples_from_yolo_export() -> None:
    config = load_baseline_b_config("configs/baseline_b.yaml")

    samples = collect_yolo_roi_samples(config.dataset, config.dataset.train_split)
    counts = count_labels(samples)

    assert len(samples) == 1512
    assert counts[0] == 758
    assert counts[1] == 754
