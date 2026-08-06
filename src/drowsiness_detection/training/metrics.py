from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    positive_false_negative_rate: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_names: Sequence[str] | None = None,
    positive_label: int = 1,
) -> ClassificationMetrics:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")

    labels = sorted(set(true.tolist()) | set(pred.tolist()))
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, actual in zip(true, pred):
        confusion[label_to_index[int(expected)], label_to_index[int(actual)]] += 1

    per_class: dict[str, dict[str, float]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        index = label_to_index[label]
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        name = _class_name(label, class_names)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(confusion[index, :].sum()),
        }

    accuracy = float((true == pred).mean())
    if positive_label in label_to_index:
        positive_index = label_to_index[positive_label]
        tp = int(confusion[positive_index, positive_index])
        fn = int(confusion[positive_index, :].sum() - tp)
        positive_fnr = _safe_divide(fn, tp + fn)
    else:
        positive_fnr = 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(np.mean(precision_values)),
        macro_recall=float(np.mean(recall_values)),
        macro_f1=float(np.mean(f1_values)),
        positive_false_negative_rate=positive_fnr,
        confusion_matrix=confusion.tolist(),
        per_class=per_class,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_name(label: int, class_names: Sequence[str] | None) -> str:
    if class_names and 0 <= label < len(class_names):
        return str(class_names[label])
    return str(label)
