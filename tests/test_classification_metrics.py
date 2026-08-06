from __future__ import annotations

from drowsiness_detection.training.metrics import compute_classification_metrics


def test_compute_classification_metrics_reports_positive_fnr() -> None:
    metrics = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        class_names=["open", "closed"],
        positive_label=1,
    )

    assert metrics.accuracy == 0.5
    assert metrics.positive_false_negative_rate == 0.5
    assert metrics.confusion_matrix == [[1, 1], [1, 1]]
    assert metrics.per_class["closed"]["support"] == 2.0
