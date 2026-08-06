from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from drowsiness_detection.video.labels import CANONICAL_ID_TO_LABEL


DEFAULT_LABELS = [0, 1, 2]
DEFAULT_CLASS_NAMES = [CANONICAL_ID_TO_LABEL[label] for label in DEFAULT_LABELS]


@dataclass(frozen=True, slots=True)
class MulticlassMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    class_support: dict[str, int]
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> MulticlassMetrics:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        average="weighted",
        zero_division=0,
    )[2]
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(true, pred, labels=list(labels))
    per_class: dict[str, dict[str, float]] = {}
    class_support: dict[str, int] = {}
    for index, label in enumerate(labels):
        name = str(class_names[index]) if index < len(class_names) else str(label)
        class_support[name] = int(per_support[index])
        per_class[name] = {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": float(per_support[index]),
        }
    return MulticlassMetrics(
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(true, pred)),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion_matrix=matrix.astype(int).tolist(),
        class_support=class_support,
        sample_count=int(true.size),
    )
